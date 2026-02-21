# GreenAI Sustainability Diagnostics — Implementation Plan

## Context

The Observer collects rich per-epoch profiler, loss, throughput, and architecture data, but none of it is analyzed through a sustainability lens. The diagnostics engine has 16+ checks for training health but nothing that identifies **wasted compute, over-parameterized layers, dead neurons, or diminishing returns** — the core GreenAI concerns.

From the sample CNN report: `fc1` holds 97% of parameters but only ~23% of compute; `conv1` has 0.08% of parameters but ~20% of compute; epochs 3-4 yield <4% marginal improvement for the same compute cost. None of this is surfaced today.

**Goal**: (1) Add runtime layer-health hooks to the Observer that capture activation, gradient, and weight statistics every batch. (2) Add derived sustainability metrics to the epoch report. (3) Add heuristic-based GreenAI checks to the diagnostics engine covering both profiler-level efficiency and new tensor-level health data.

---

## Files to Modify

| File | Change |
|------|--------|
| `neural_network/observer.py` | Add layer-health hooks, `_compute_sustainability()`, `_compute_layer_health()`, update `ObserverConfig`, `register_model()`, `_start_epoch()`, `end_epoch()`, `_build_summary()`, `close()` |
| `backend/models.py` | Add `sustainability` to `IssueCategory` enum |
| `backend/diagnostics/engine.py` | Add 10 new GreenAI check functions (5 profiler-level + 5 tensor-level), wire into `run_diagnostics()` |
| `backend/diagnostics/schemas.py` | Add `SustainabilityInsight` model, update `DiagnosticRunOut` |
| `backend/routers/diagnostics.py` | Populate `sustainability` field in responses |

---

## Part 1A: Observer — Runtime Layer Health Hooks

**File:** `neural_network/observer.py`

### Config additions (`ObserverConfig`, ~line 34)

```python
# GreenAI / layer health
track_sustainability: bool = True      # master switch for sustainability metrics
track_layer_health: bool = True        # persistent activation/gradient hooks (some overhead per batch)
layer_health_zero_threshold: float = 1e-6  # threshold for "near-zero" weights/activations
```

### New ephemeral state (`__init__`, ~line 162)

```python
# -- Layer health (per-epoch, reset in _start_epoch) --
self._layer_activation_stats: Dict[str, List[Dict]] = defaultdict(list)
self._layer_gradient_stats: Dict[str, List[Dict]] = defaultdict(list)
self._health_hooks: List[Any] = []  # persistent hook handles, removed in close()
```

### Hook registration in `register_model()` (~line 255)

When `config.track_layer_health` is True, install **persistent forward hooks and backward hooks** on every parameter layer (`_get_parameter_layers`). Hooks capture lightweight scalar statistics only (no tensor storage):

**Forward hook** — captures per-batch activation stats:
- `output.mean()`, `output.std()`, `output.var()`, sparsity (% below threshold), L2 norm

**Backward hook** — captures per-batch gradient flow:
- `grad_output.norm()`, `grad_output.mean()`, `grad_output.std()`, gradient sparsity

### Reset in `_start_epoch()` (~line 487)

Clear `_layer_activation_stats` and `_layer_gradient_stats`.

### Remove hooks in `close()` (~line 1064)

Remove all handles in `_health_hooks` and clear the list.

---

## Part 1B: Observer — Derived Sustainability Metrics

### New method `_compute_layer_health()` (after `_snapshot_system`)

Aggregates per-batch activation/gradient stats into per-layer summaries. Also snapshots current weight tensors.

Per-layer output:

**Activation health:**
- `activation_mean`, `activation_std`, `activation_var_of_means` (output variability across batches)
- `activation_sparsity` (% near-zero activations), `num_batches`

**Gradient health:**
- `gradient_norm_mean`, `gradient_norm_std`, `gradient_sparsity`

**Weight health** (snapshot from model):
- `weight_sparsity` (% near-zero weights), `weight_mean`, `weight_std`, `weight_norm`

**Derived boolean flags:**
- `is_dead`: `activation_var_of_means < 1e-8` AND `weight_sparsity > 0.9`
- `has_frozen_output`: `activation_var_of_means < 1e-8`
- `has_vanishing_gradients`: `gradient_norm_mean < 1e-7`
- `has_near_zero_weights`: `weight_sparsity > 0.5`
- `has_low_activation_variance`: `activation_std < 1e-6`

**Activation correlations** (between consecutive layers in sequential path):
- Pearson correlation of per-batch activation means between layer pairs
- Stored as `activation_correlations: [{layer_a, layer_b, correlation}]`

### New method `_compute_sustainability()` (after `_compute_layer_health`)

Derives profiler-level sustainability metrics:

- **`layer_efficiency`**: `compute_to_param_ratio` per layer (profiler pct_total / arch pct_of_total)
- **`marginal_loss`**: `absolute_improvement`, `pct_improvement`, `cumulative_improvement`, `marginal_over_cumulative`
- **`epoch_compute_cost`**: duration, CPU/CUDA time, samples processed
- **`cumulative_compute`**: running totals across all epochs

### Call from `end_epoch()` (~line 627)

```python
rec["layer_health"] = self._compute_layer_health()
rec["sustainability"] = self._compute_sustainability(rec, epoch)
```

### Update `_build_summary()` (~line 1056)

Add sustainability summary:
- `optimal_stop_epoch`, `wasted_epochs`, `wasted_compute_pct`, `wasted_duration_seconds`
- `parameter_efficiency_score` (0-100, based on log2 deviation of compute/param ratios)
- `dead_layers`, `vanishing_gradient_layers`, `frozen_output_layers`

---

## Part 2: Backend Engine — GreenAI Heuristic Checks

### 2a. Add `sustainability` to `IssueCategory` (`backend/models.py`)

### 2b. Profiler-level checks (5 functions)

| Check | Trigger | Severity |
|-------|---------|----------|
| `check_diminishing_returns` | marginal/cumulative < 5% after epoch 2 | warning |
| `check_over_parameterized_layer` | pct_params / pct_compute > 10x | warning |
| `check_compute_inefficient_layer` | pct_compute / pct_params > 10x, compute > 5% | info |
| `check_device_underutilization` | total CUDA time = 0 or CUDA/CPU < 10% | info/warning |
| `check_early_stop_opportunity` | Quantifies wasted compute past optimal stop | warning |

### 2c. Tensor-level checks (5 functions, using `layer_health` data)

| Check | Trigger | Severity |
|-------|---------|----------|
| `check_dead_neurons` | weight_sparsity > 0.5 or is_dead flag | warning/critical |
| `check_vanishing_gradients` | gradient_norm < 1e-7 across 2+ epochs | warning |
| `check_frozen_output` | activation_var_of_means < 1e-8 across 2+ epochs | warning |
| `check_activation_collapse` | activation_std < 1e-6 across 2+ epochs | warning |
| `check_redundant_layers` | activation correlation > 0.95 across 2+ epochs | info |

All tensor-level checks gracefully return `[]` when `layer_health` data is absent.

### 2d. Wire into `run_diagnostics()` (~line 800)

Add all 10 checks between profiler checks and architecture-specific checks.

---

## Part 3: Schema & Router Updates

### New schema (`backend/diagnostics/schemas.py`)

```python
class SustainabilityInsight(BaseModel):
    optimal_stop_epoch: int | None = None
    wasted_epochs: int | None = None
    wasted_compute_pct: float | None = None
    wasted_duration_seconds: float | None = None
    parameter_efficiency_score: float | None = None
    total_training_duration_seconds: float | None = None
    total_samples_processed: int | None = None
    dead_layers: list[str] = []
    vanishing_gradient_layers: list[str] = []
    frozen_output_layers: list[str] = []
    redundant_layer_pairs: list[dict] = []
    sustainability_issue_count: int = 0
```

Add to `DiagnosticRunOut` and populate in router endpoints.

---

## Expected Results on Sample Data

| Check | Fires? | Detail |
|-------|--------|--------|
| diminishing_returns | Yes | Epoch 4 (marginal/cumulative = 3.6%) |
| over_parameterized_layer | No | fc1 ratio = 4.2x (< 10x threshold) |
| compute_inefficient_layer | Yes | conv1 (250x ratio), conv2 (13.4x ratio) |
| device_underutilization | Yes | All CUDA time = 0 |
| early_stop_opportunity | Yes | Optimal stop epoch 3, ~20% compute wasted |
| dead_neurons | Depends | Requires layer_health data |
| vanishing_gradients | Depends | Requires layer_health data |
| frozen_output | Depends | Requires layer_health data |
| activation_collapse | Depends | Requires layer_health data |
| redundant_layers | Depends | Requires layer_health data |

## Part 4: CodeCarbon Integration (Carbon Footprint)

Integrated [CodeCarbon](https://github.com/mlco2/codecarbon) for real CO2 emissions and energy tracking.

### Observer (`neural_network/observer.py`)
- **Config**: `track_carbon_emissions=True`, `carbon_tracker_mode="online"`, `carbon_country_iso="IRL"`
- **Lazy init**: `_init_carbon_tracker()` starts CodeCarbon on first epoch (gracefully degrades if not installed)
- **Per-epoch**: `flush()` in `end_epoch()` captures delta CO2/energy since last flush → `carbon_emissions` block
- **Summary**: `total_co2_kg`, `total_energy_kwh`, `co2_per_epoch_avg_kg`, `avg_power_draw_watts`, `wasted_co2_kg`
- **Cleanup**: `close()` calls `tracker.stop()`

### Per-epoch `carbon_emissions` fields
- `epoch_co2_kg`, `epoch_energy_kwh`, `cumulative_co2_kg`, `cumulative_energy_kwh`
- `co2_per_sample_kg`, `co2_per_second_kg`, `power_draw_watts`, `country_iso_code`

### Backend Diagnostics
- **`check_high_carbon_intensity()`**: Reports total footprint (info) + flags epochs with 10x+ average carbon intensity (warning)
- **`check_wasted_carbon()`**: Quantifies CO2 wasted on epochs past optimal stop (warning)
- **Schema**: 6 new carbon fields on `SustainabilityInsight`
- **Router**: Extracts carbon data into insight from epoch data and issue metric_values

---

## Verification

1. Run small_cnn sample notebook → verify `layer_health` + `sustainability` + `carbon_emissions` in JSON report
2. Call `run_diagnostics()` with sample data → verify expected issue firing
3. POST `/diagnostics/sessions/{id}/run` → verify `sustainability` in API response with carbon fields
4. Uninstall codecarbon → verify graceful degradation (warning logged, no crash)
