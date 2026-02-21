# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two-part project:
1. **`neural_network/`** — Python/PyTorch Mini GPT transformer trained on child speech data, plus a training observer that exports JSON diagnostics.
2. **`visualization/`** — Three.js/Vite frontend that renders a 3D interactive visualization of the trained model's architecture.

## Commands

### Visualization (JavaScript / Vite)

```bash
cd visualization
npm install
npm run dev       # Vite dev server with hot reload
npm run build     # production build (use to verify 0 errors before committing)
npm run preview   # preview production build
```

No linter or test runner is configured. `npm run build` is the correctness check.

### Neural Network (Python)

```bash
cd neural_network
pip install -r requirements.txt   # torch, psutil, pydantic
```

Training runs cell-by-cell in `neural_network/sample/mini_gpt.ipynb`. `observer.py` is imported inside the notebook. Reports are written to `neural_network/observer_reports/`.

## Architecture

### Data Flow (end-to-end)

```
mini_gpt.ipynb  ──trains──►  Observer  ──writes──►  observer_reports/*.json
                                                          │
                                          visualization/mock-report.json  (checked-in copy)
                                                          │
                                        api.js: parseObserverReport()
                                                          │
                                     typed layer list: [{ id, type, params }]
                                                          │
                               main.js: LAYER_RENDERERS[layer.type](layer)
                                                          │
                          models/*.js factory functions → THREE.Group
                                                          │
                                         Three.js scene (top-to-bottom layout)
```

The live fetch from the Python server is currently commented out in `api.js`. Swap the return statement there to re-enable it.

### Visualization — Key Concepts

**Layer type system.** `main.js` maintains a `LAYER_RENDERERS` map from string type names to factory functions imported from `visualization/src/models/`. Supported types: `Embedding`, `Linear`, `LayerNorm`, `Dropout`, `ReLU`, `Head`, `MultiHeadAttention`, `FeedForward`, `FeedFoward` (the latter is a typo preserved from the PyTorch model — both must exist in the map), `Block`. Unknown types fall through to `createFallbackGroup`.

**Anchor contract.** Every factory in `models/` must set:
```js
group.userData.inputAnchor  = new THREE.Vector3(−halfW, 0, 0);
group.userData.outputAnchor = new THREE.Vector3(+halfW, 0, 0);
```
Anchors **must be at `y = 0`** (X-direction only). `main.js` does not apply any auto-orient rotation; it places each group at `position.y = cursorY` and decrements `cursorY` by `STAGE_SPACING`. Breaking the `y = 0` constraint causes groups to be mis-positioned.

**3D coordinate transform.** Every group receives `group.rotation.x = Math.PI / 2` in `buildNetwork`. This maps local coordinates to world coordinates as:
```
local (x, y, z)  →  world (x, −z, y)
```
Consequences for `models/` files:
- Local Y positions become world Z depth (perspective foreground/background when camera is at positive Z).
- A component at local `y = +2.0` appears **in front** of the spine (`y = 0`) and closer to the camera.
- A component at local `y = −1.4` appears **behind** the spine (further from camera).
- This is intentional: in `block.js`, sublayers at `y = +2.0` float in the foreground; skip connections at `y = −1.4` recede into the background.

**`dim()` helper.** All model files inline the same log-scaled dimension sizing function:
```js
function dim(n) { return Math.max(0.7, Math.min(3.2, Math.log2(n) * 0.48)); }
```
Do not introduce a shared module for this — it is intentionally inlined per file.

**Emissive pulse.** The forward-pass animation in `main.js` ramps `emissiveIntensity` from `0.08` (rest) up and then back down. All `MeshPhongMaterial` instances in model groups will respond automatically. `LineSegments` materials do not participate (no `emissiveIntensity`).

**Raycasting.** `main.js` calls `collectMeshes(group)` after adding each group and tags every `Mesh` with `mesh.userData.stageIndex`. Hover highlighting and tooltip display rely on this. `LineSegments` objects are excluded from raycasting.

### Visualization — File Map

- **`src/api.js`** — Parses the Observer JSON report into `{ name, layers: [{id, type, params}] }`. `paramsFor()` derives renderer-specific params (e.g. `vocabSize`, `nEmbd`, `headSize`) from hyperparameters + the flat layer dict.
- **`src/main.js`** — Scene lifecycle, `buildNetwork()`, raycasting, hover tooltips, forward-pass pulse animation, camera framing, menu controls.
- **`src/models/`** — One factory per layer type. Each is a standalone ES module with no cross-imports. Shared helpers (`dim`, `addGrid`, `slabMat`) are inlined in each file.
- **`mock-report.json`** — A real Observer export from a `128-emb / 4-layer / 4-head` GPT run on child speech data. This is the default data source.
- **`network-schema.json`** — JSON Schema for the simpler `{ name, layers: [{id, size}] }` format (used if the server returns plain architecture data instead of a full Observer report).

### Neural Network — Key Concepts

**Observer API.** The `Observer` class wraps the training loop:
```python
obs = Observer(config=ObserverConfig(...), project_id="...", run_name="...")
obs.register_model(model)           # one-time: hooks in gradient/activation tracking
obs.start_epoch(epoch)
for x, y in dataloader:
    if first_batch: obs.profile_step(model, x, y)
    loss = ...
    obs.log_batch(step, loss, batch_size, seq_length)
obs.end_epoch(epoch, val_metrics={"val_loss": ...})
obs.close()                         # writes JSON to observer_reports/
```

**Observer report schema.** Defined as Pydantic models in `neural_network/schema.py`. Root model: `ObserverReport`. Key sub-structures:
- `session` — run metadata and config snapshot
- `hyperparameters` — training HP dict (batch_size, block_size, n_embd, n_head, n_layer, dropout, …)
- `model_architecture.module_tree` — recursive `nn.Module` nesting (`type`, `children`)
- `model_architecture.layers` — flat dict of all parametric leaf modules → `{type, parameters, pct_of_total}`
- `model_architecture.layer_graph` — present when `track_layer_graph=True`; contains `nodes`, `edges`, `sequential_path`, `dimension_flow`
- `epochs[]` — per-epoch telemetry (loss, gradients, activations, weights, memory, throughput, profiler, attention_entropy)

**Expensive channels** (disable for fast runs): `track_profiler`, `track_activations`, `track_attention_entropy`.

**`FeedFoward` typo.** The PyTorch model class is literally named `FeedFoward` (missing 'r'). This typo propagates into Observer output and into `api.js` / `main.js` where both spellings are handled.
