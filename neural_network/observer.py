"""
Observer - PyTorch Training Pipeline Monitor

Lightweight training observer that captures session metadata, hyperparameters,
model architecture, and per-step metrics (loss, throughput, memory, system, profiler).

Collects:
  - Session (run id, device, config snapshot)
  - Hyperparameters (user-supplied)
  - Model architecture (layer map, module tree, optional layer graph)
  - Per-step: loss, throughput, memory, system, profiler (when profile_step used), log_counts
"""

import json
import logging
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ObserverConfig:
    """Controls which telemetry channels the Observer records."""

    # Core tracking
    track_profiler: bool = True
    track_memory: bool = True
    track_throughput: bool = True
    track_loss: bool = True

    # Logging
    track_console_logs: bool = True
    track_error_logs: bool = True
    track_hyperparameters: bool = True

    # Architecture visualization
    track_layer_graph: bool = True

    # System
    track_system_resources: bool = True

    # Profiler tuning
    profile_at_step: Optional[int] = 0  # Which step to profile (None = never)
    profile_every_n_steps: Optional[int] = None  # Profile every N steps (overrides profile_at_step when set)
    profiler_record_shapes: bool = True
    profiler_profile_memory: bool = True
    profiler_with_stack: bool = False
    profiler_top_n_ops: int = 20
    # Per-stack grouping: set group_by_stack_n > 0 to see which call stacks use the most compute.
    # with_stack is auto-enabled when group_by_stack_n > 0. Stack depth is often 5 (PyTorch limit).
    profiler_group_by_stack_n: int = 0
    profiler_top_n_stacks: int = 20

    # GreenAI / layer health
    track_sustainability: bool = True       # sustainability metrics (layer efficiency, marginal loss, compute costs)
    track_layer_health: bool = True         # persistent activation/gradient hooks (some overhead per batch)
    layer_health_zero_threshold: float = 1e-6  # threshold for "near-zero" weights/activations

    # Carbon emissions (CodeCarbon)
    track_carbon_emissions: bool = True          # enable CO2/energy tracking via CodeCarbon
    carbon_tracker_mode: str = "online"          # "online" (real-time grid data) or "offline" (static country data)
    carbon_country_iso: str = "IRL"              # ISO 3166-1 alpha-3, used as fallback or for offline mode

    # Pending timeout (exponential backoff)
    pending_timeout: float = 39.0  # seconds to wait before auto-stopping when session is pending

    # Log level for the observer's own logger
    log_level: int = logging.INFO


# ---------------------------------------------------------------------------
# Log capture handler
# ---------------------------------------------------------------------------

class _LogCaptureHandler(logging.Handler):
    """Lightweight handler that appends log records into a list and optionally calls on_emit for real-time push."""

    def __init__(
        self,
        store: list,
        min_level: int = logging.DEBUG,
        on_emit: Optional[Callable[[Dict[str, Any], str], None]] = None,
        kind: Literal["console", "error"] = "console",
    ):
        super().__init__(min_level)
        self.store = store
        self.on_emit = on_emit
        self.kind = kind

    def emit(self, record):
        try:
            entry = {
                "ts": datetime.now().isoformat(),
                "level": record.levelname,
                "msg": self.format(record),
                "module": getattr(record, "module", ""),
                "lineno": getattr(record, "lineno", 0),
            }
            self.store.append(entry)
            if self.on_emit:
                try:
                    self.on_emit(entry, self.kind)
                except Exception:
                    pass  # never break the training loop
        except Exception:
            pass  # never break the training loop


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class Observer:
    """
    Step-based training observer that collects rich diagnostics.

    Quickstart
    ----------
    >>> config = ObserverConfig(track_profiler=True, profile_at_step=0)
    >>> obs = Observer(project_id="proj-abc", config=config)
    >>> obs.log_hyperparameters({"lr": 3e-4, "batch_size": 64, ...})
    >>> obs.register_model(model)
    >>>
    >>> for step, (x, y) in enumerate(loader):
    ...     if obs.should_profile(step):
    ...         logits, loss = obs.profile_step(model, x, y)
    ...     else:
    ...         logits, loss = model(x, y)
    ...         loss.backward()
    ...     optimizer.step(); optimizer.zero_grad()
    ...     obs.step(step, loss, batch_size=x.size(0), seq_length=x.size(1))
    ... obs.flush(val_metrics=val_metrics)
    >>>
    >>> obs.export("observer_reports/run.json")
    >>> obs.close()
    """

    def __init__(
        self,
        project_id: int,
        config: Optional[ObserverConfig] = None,
        run_name: Optional[str] = None,
        *,
        backend_base_url: str = "http://localhost:8000",
    ):
        self.project_id = project_id
        self.config = config or ObserverConfig()
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_id = f"{project_id}_{self.run_name}"

        # ── Backend sync (session id stored after create) ──
        self._backend_base_url = backend_base_url.rstrip("/")
        self._backend_session_id: Optional[int] = None

        # ── Session metadata ──
        self.session: Dict[str, Any] = {
            "project_id": project_id,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "device": str(
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
            "cuda_available": torch.cuda.is_available(),
            "pytorch_version": torch.__version__,
            "config": asdict(self.config),
        }

        # ── Persistent stores ──
        self.hyperparameters: Dict[str, Any] = {}
        self.model_architecture: Dict[str, Any] = {}
        self.step_data: List[Dict[str, Any]] = []
        self.console_logs: List[Dict] = []
        self.error_logs: List[Dict] = []

        # ── Ephemeral (per-interval) state, reset on flush() ──
        self._interval_started: bool = False
        self._step_start_time: float = 0.0
        self._step_batch_losses: List[float] = []
        self._step_batch_times: List[float] = []
        self._step_tokens_processed: int = 0
        self._step_samples_processed: int = 0
        self._profiler_snapshot: Optional[Dict] = None

        self._model: Optional[nn.Module] = None

        # ── Layer health (per-step, reset in _start_interval) ──
        self._layer_activation_stats: Dict[str, List[Dict]] = defaultdict(list)
        self._layer_gradient_stats: Dict[str, List[Dict]] = defaultdict(list)
        self._health_hooks: List[Any] = []  # persistent hook handles, removed in close()

        # ── Carbon emissions (CodeCarbon) ──
        self._carbon_tracker: Optional[Any] = None       # lazily initialized
        self._carbon_prev_emissions: float = 0.0         # cumulative CO2 at previous flush
        self._carbon_prev_energy: float = 0.0            # cumulative energy at previous flush

        # ── Observer logger ──
        self._log = logging.getLogger(f"observer.{self.run_id}")
        self._log.setLevel(self.config.log_level)
        if not self._log.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(logging.Formatter("[Observer] %(message)s"))
            self._log.addHandler(_h)

        # ── Attach log capture ──
        self._capture_handlers: list = []
        if self.config.track_console_logs:
            h = _LogCaptureHandler(
                self.console_logs,
                logging.INFO,
                on_emit=lambda e, k: self._push_single_log_to_backend(e, k),
                kind="console",
            )
            h.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger().addHandler(h)
            self._capture_handlers.append(h)
        if self.config.track_error_logs:
            h = _LogCaptureHandler(
                self.error_logs,
                logging.WARNING,
                on_emit=lambda e, k: self._push_single_log_to_backend(e, k),
                kind="error",
            )
            h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logging.getLogger().addHandler(h)
            self._capture_handlers.append(h)

        self._log.info(
            f"Initialized | project={project_id} | run={self.run_name} | "
            f"device={self.session['device']}"
        )

        self._create_backend_session()

    # ==================================================================
    # Hyperparameters
    # ==================================================================

    def log_hyperparameters(self, params: dict):
        """Record training hyperparameters (call before training starts)."""
        self.hyperparameters.update(params)
        self._log.info(f"Hyperparameters logged: {list(params.keys())}")

    def _create_backend_session(self) -> None:
        """Create a train session in the backend and store the returned session id."""
        url = f"{self._backend_base_url}/sessions/project/{self.project_id}"
        payload = {
            "run_id": self.session["run_id"],
            "run_name": self.session["run_name"],
            "started_at": self.session["started_at"],
            "device": self.session["device"],
            "cuda_available": self.session["cuda_available"],
            "pytorch_version": self.session["pytorch_version"],
            "config": self.session["config"],
            "status": "running",
        }
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            self._backend_session_id = data.get("id")
            if self._backend_session_id is not None:
                self._log.info(f"Backend session created | session_id={self._backend_session_id}")
            else:
                self._log.warning("Backend did not return session id")
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as e:
            self._log.warning(f"Failed to create backend session: {e}")
            raise e

    def _register_backend_model(self, architecture: Dict[str, Any], hyperparameters: Dict[str, Any]) -> None:
        """Register the model (architecture + hyperparameters) with the backend session."""
        if self._backend_session_id is None:
            return
        url = f"{self._backend_base_url}/sessions/{self._backend_session_id}/model"
        payload = {"architecture": architecture, "hyperparameters": hyperparameters}
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            self._log.info(f"Model registered in backend | model_id={data.get('id', '?')}")
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as e:
            self._log.warning(f"Failed to register model in backend: {e}")
            raise e

    def _push_single_log_to_backend(self, entry: Dict[str, Any], kind: Literal["console", "error"]) -> None:
        """Push a single log entry to the backend in real time via POST /sessions/{id}/log."""
        if self._backend_session_id is None:
            return
        if kind == "console" and not self.config.track_console_logs:
            return
        if kind == "error" and not self.config.track_error_logs:
            return
        payload = {
            "ts": entry["ts"],
            "level": entry["level"],
            "msg": entry["msg"],
            "module": entry.get("module", ""),
            "lineno": entry.get("lineno", 0),
            "kind": kind,
        }
        url = f"{self._backend_base_url}/sessions/{self._backend_session_id}/log"
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10):
                pass
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as e:
            self._log.warning(f"Failed to push log to backend: {e}")

    def _register_backend_step(self, rec: Dict[str, Any]) -> None:
        """Register the current step record with the backend via POST /sessions/{id}/step."""
        if self._backend_session_id is None:
            return
        payload = {
            "step_index": rec["step"],
            "timestamp": rec["timestamp"],
            "duration_seconds": rec["duration_seconds"],
            "loss": rec.get("loss") or {},
            "throughput": rec.get("throughput") or {},
            "profiler": rec.get("profiler") or {},
            "memory": rec.get("memory") or {},
            "system": rec.get("system") or {},
        }
        url = f"{self._backend_base_url}/sessions/{self._backend_session_id}/step"
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            self._log.info(f"Step {rec['step']} registered in backend | step_id={data.get('id', '?')}")
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as e:
            self._log.warning(f"Failed to register step in backend: {e}")

    # ==================================================================
    # Model registration & architecture analysis
    # ==================================================================

    def register_model(self, model: nn.Module):
        """
        Register a model for observation.

        This captures the full architecture map (layer types, param counts),
        detailed layer graph (neurons, weight shapes, hidden dimensions,
        connections between layers), and attaches forward hooks for
        activation / attention monitoring.
        """
        self._model = model

        # ── Architecture map ──
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        layer_map: Dict[str, Dict] = {}
        for name, module in model.named_modules():
            if name == "":
                continue
            own_params = sum(p.numel() for p in module.parameters(recurse=False))
            if own_params > 0:
                layer_map[name] = {
                    "type": type(module).__name__,
                    "parameters": own_params,
                    "pct_of_total": round(100 * own_params / max(total_params, 1), 2),
                }

        self.model_architecture = {
            "total_parameters": total_params,
            "trainable_parameters": trainable,
            "frozen_parameters": total_params - trainable,
            "num_parameter_layers": len(layer_map),
            "layers": layer_map,
            "module_tree": self._module_tree(model),
        }

        # ── Full layer graph for visualization ──
        if self.config.track_layer_graph:
            self.model_architecture["layer_graph"] = self._build_layer_graph(model)

        # ── Layer health hooks ──
        if self.config.track_layer_health:
            param_layers = self._get_parameter_layers(model)
            for name, module in param_layers:
                h1 = module.register_forward_hook(self._make_activation_hook(name))
                h2 = module.register_full_backward_hook(self._make_gradient_hook(name))
                self._health_hooks.extend([h1, h2])
            self._log.info(
                f"Layer health hooks registered on {len(param_layers)} layers"
            )

        self._log.info(
            f"Model registered | {total_params:,} params "
            f"({total_params / 1e6:.2f}M) | {len(layer_map)} param layers"
        )

        if self._backend_session_id is not None:
            self._register_backend_model(self.model_architecture, self.hyperparameters)

    @staticmethod
    def _module_tree(model: nn.Module, prefix: str = "") -> Dict:
        """Build a recursive tree of module types (lightweight architecture snapshot)."""
        tree: Dict[str, Any] = {"type": type(model).__name__}
        children = {}
        for name, child in model.named_children():
            children[name] = Observer._module_tree(child, prefix=f"{prefix}.{name}")
        if children:
            tree["children"] = children
        return tree

    # ------------------------------------------------------------------
    # Layer health hooks (persistent, fire every batch)
    # ------------------------------------------------------------------

    def _make_activation_hook(self, layer_name: str):
        """Create a forward hook that captures per-batch activation statistics."""
        threshold = self.config.layer_health_zero_threshold

        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                out = output.detach().float()
                self._layer_activation_stats[layer_name].append({
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "var": out.var().item(),
                    "sparsity": (out.abs() < threshold).float().mean().item(),
                    "norm": out.norm(2).item(),
                })

        return hook

    def _make_gradient_hook(self, layer_name: str):
        """Create a backward hook that captures per-batch gradient flow statistics."""
        threshold = self.config.layer_health_zero_threshold

        def hook(module, grad_input, grad_output):
            g = grad_output[0] if grad_output and grad_output[0] is not None else None
            if g is None:
                return
            with torch.no_grad():
                g = g.detach().float()
                self._layer_gradient_stats[layer_name].append({
                    "norm": g.norm(2).item(),
                    "mean": g.mean().item(),
                    "std": g.std().item(),
                    "sparsity": (g.abs() < threshold).float().mean().item(),
                })

        return hook

    # ------------------------------------------------------------------
    # Layer graph for architecture visualization
    # ------------------------------------------------------------------

    def _build_layer_graph(self, model: nn.Module) -> Dict[str, Any]:
        """
        Build a full layer graph describing every neuron dimension, weight
        matrix shape, bias presence, hidden layers, and connections.

        Returns a dict with:
          - nodes: ordered list of layers with full metadata
          - edges: connections between layers (source -> target)
          - sequential_path: the forward-pass order of compute layers
          - summary: high-level dimension flow
        """
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        sequential_path: List[str] = []

        # Walk every module and extract detailed info
        for name, module in model.named_modules():
            if name == "":
                continue

            node: Dict[str, Any] = {
                "id": name,
                "type": type(module).__name__,
                "is_container": len(list(module.children())) > 0,
            }

            # ── Per-layer-type details ──
            if isinstance(module, nn.Linear):
                node["in_features"] = module.in_features
                node["out_features"] = module.out_features
                node["neurons"] = module.out_features
                node["has_bias"] = module.bias is not None
                node["weight_shape"] = list(module.weight.shape)
                if module.bias is not None:
                    node["bias_shape"] = list(module.bias.shape)
                node["category"] = "linear"

            elif isinstance(module, nn.Embedding):
                node["num_embeddings"] = module.num_embeddings
                node["embedding_dim"] = module.embedding_dim
                node["neurons"] = module.embedding_dim
                node["weight_shape"] = list(module.weight.shape)
                node["padding_idx"] = module.padding_idx
                node["category"] = "embedding"

            elif isinstance(module, nn.LayerNorm):
                ns = list(module.normalized_shape)
                node["normalized_shape"] = ns
                node["neurons"] = ns[0] if len(ns) == 1 else ns
                node["has_weight"] = module.weight is not None
                node["has_bias"] = module.bias is not None
                node["eps"] = module.eps
                if module.weight is not None:
                    node["weight_shape"] = list(module.weight.shape)
                node["category"] = "normalization"

            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                node["num_features"] = module.num_features
                node["neurons"] = module.num_features
                node["eps"] = module.eps
                node["momentum"] = module.momentum
                node["affine"] = module.affine
                if module.weight is not None:
                    node["weight_shape"] = list(module.weight.shape)
                node["category"] = "normalization"

            elif isinstance(module, nn.Dropout):
                node["p"] = module.p
                node["category"] = "regularization"

            elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid)):
                node["category"] = "activation"
                if hasattr(module, "inplace"):
                    node["inplace"] = module.inplace

            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                node["in_channels"] = module.in_channels
                node["out_channels"] = module.out_channels
                node["kernel_size"] = list(module.kernel_size)
                node["stride"] = list(module.stride)
                node["padding"] = list(module.padding)
                node["has_bias"] = module.bias is not None
                node["weight_shape"] = list(module.weight.shape)
                node["neurons"] = module.out_channels
                node["category"] = "convolution"

            elif isinstance(module, nn.MultiheadAttention):
                node["embed_dim"] = module.embed_dim
                node["num_heads"] = module.num_heads
                node["head_dim"] = module.head_dim
                node["neurons"] = module.embed_dim
                node["category"] = "attention"

            elif isinstance(module, nn.Sequential):
                node["num_children"] = len(module)
                node["category"] = "container"

            elif isinstance(module, nn.ModuleList):
                node["num_children"] = len(module)
                node["category"] = "container"

            else:
                node["category"] = "custom"

            # ── Collect all weight tensors for this module ──
            own_params = {}
            for pname, param in module.named_parameters(recurse=False):
                own_params[pname] = {
                    "shape": list(param.shape),
                    "numel": param.numel(),
                    "dtype": str(param.dtype),
                    "requires_grad": param.requires_grad,
                    "mean": round(param.data.float().mean().item(), 6),
                    "std": round(param.data.float().std().item(), 6),
                    "norm": round(param.data.float().norm(2).item(), 6),
                }
            if own_params:
                node["parameters"] = own_params
                node["total_params"] = sum(p["numel"] for p in own_params.values())

            # ── Collect buffers (e.g. causal masks, running stats) ──
            own_buffers = {}
            for bname, buf in module.named_buffers(recurse=False):
                own_buffers[bname] = {
                    "shape": list(buf.shape),
                    "dtype": str(buf.dtype),
                }
            if own_buffers:
                node["buffers"] = own_buffers

            nodes.append(node)

            # Track compute layers for sequential path
            if not node["is_container"] and node["category"] != "container":
                sequential_path.append(name)

        # ── Build edges from parent-child relationships ──
        for name, module in model.named_modules():
            if name == "":
                continue
            for child_name, _ in module.named_children():
                full_child = f"{name}.{child_name}" if name else child_name
                edges.append({
                    "source": name,
                    "target": full_child,
                    "relation": "contains",
                })

        # ── Build data-flow edges for Sequential containers ──
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                children = list(module.named_children())
                for i in range(len(children) - 1):
                    src = f"{name}.{children[i][0]}" if name else children[i][0]
                    tgt = f"{name}.{children[i+1][0]}" if name else children[i+1][0]
                    edges.append({
                        "source": src,
                        "target": tgt,
                        "relation": "data_flow",
                    })

        # ── Build dimension flow summary ──
        dimension_flow: List[Dict[str, Any]] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                dimension_flow.append({
                    "layer": name,
                    "type": "Linear",
                    "in": module.in_features,
                    "out": module.out_features,
                })
            elif isinstance(module, nn.Embedding):
                dimension_flow.append({
                    "layer": name,
                    "type": "Embedding",
                    "vocab": module.num_embeddings,
                    "dim": module.embedding_dim,
                })
            elif isinstance(module, nn.LayerNorm):
                dimension_flow.append({
                    "layer": name,
                    "type": "LayerNorm",
                    "shape": list(module.normalized_shape),
                })
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                dimension_flow.append({
                    "layer": name,
                    "type": type(module).__name__,
                    "in_ch": module.in_channels,
                    "out_ch": module.out_channels,
                    "kernel": list(module.kernel_size),
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "sequential_path": sequential_path,
            "dimension_flow": dimension_flow,
            "total_compute_layers": len(sequential_path),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    # ==================================================================
    # Carbon emissions (CodeCarbon)
    # ==================================================================

    def _init_carbon_tracker(self):
        """Lazily initialize CodeCarbon emissions tracker."""
        if self._carbon_tracker is not None or not self.config.track_carbon_emissions:
            return
        try:
            from codecarbon import OfflineEmissionsTracker, EmissionsTracker
            if self.config.carbon_tracker_mode == "online":
                self._carbon_tracker = EmissionsTracker(
                    log_level="error",
                    save_to_file=False,
                    save_to_api=False,
                )
            else:
                self._carbon_tracker = OfflineEmissionsTracker(
                    country_iso_code=self.config.carbon_country_iso,
                    log_level="error",
                    save_to_file=False,
                    save_to_api=False,
                )
            self._carbon_tracker.start()
            self._log.info(
                f"CodeCarbon tracker started ({self.config.carbon_tracker_mode} mode)"
            )
        except ImportError:
            self._log.warning(
                "codecarbon not installed; carbon tracking disabled. "
                "pip install codecarbon"
            )
            self.config.track_carbon_emissions = False
        except Exception as e:
            self._log.warning(f"CodeCarbon init failed: {e}; carbon tracking disabled")
            self.config.track_carbon_emissions = False

    # ==================================================================
    # Step lifecycle
    # ==================================================================

    def _start_interval(self):
        """Internal: reset per-interval state (on first step or after flush)."""
        self._interval_started = True
        self._step_start_time = time.time()
        self._step_batch_losses.clear()
        self._step_batch_times.clear()
        self._step_tokens_processed = 0
        self._step_samples_processed = 0
        self._profiler_snapshot = None
        self._layer_activation_stats.clear()
        self._layer_gradient_stats.clear()

        # Lazy-init carbon tracker on first step/interval
        if self.config.track_carbon_emissions and self._carbon_tracker is None:
            self._init_carbon_tracker()

        if torch.cuda.is_available() and self.config.track_memory:
            torch.cuda.reset_peak_memory_stats()

    def should_profile(self, step: int) -> bool:
        """Return True if profiling should run on this step.

        When ``profile_every_n_steps`` is set, profiles every N-th step
        (i.e. steps 0, N, 2N, …).  Otherwise falls back to the single
        ``profile_at_step`` behaviour.
        """
        if not self.config.track_profiler:
            return False
        if self.config.profile_every_n_steps is not None:
            return step % self.config.profile_every_n_steps == 0
        return (
            self.config.profile_at_step is not None
            and step == self.config.profile_at_step
        )

    def step(
        self,
        step: int,
        loss,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ):
        """
        Record a single training step. On first call, initialises the interval buffer.

        Parameters
        ----------
        step : int
            Batch / step index.
        loss : torch.Tensor | float
            Loss value for this step.
        batch_size : int, optional
            Number of samples in this batch.
        seq_length : int, optional
            Sequence length per sample (for token throughput).
        """
        if not self._interval_started:
            self._start_interval()

        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        self._step_batch_losses.append(loss_val)
        self._step_batch_times.append(time.time())
        if batch_size:
            self._step_samples_processed += batch_size
        if batch_size and seq_length:
            self._step_tokens_processed += batch_size * seq_length

        # Per-step log to verify backend log push (each step → one log entry on the backend)
        self._log.info(
            f"[observer] step={step} loss={loss_val:.6f}"
            + (f" batch_size={batch_size}" if batch_size else "")
            + (f" seq_length={seq_length}" if seq_length else "")
        )

    def flush(
        self,
        val_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Finalise the current interval: collect all channel data and return the record.
        Call this when you want to commit the current batch of steps (e.g. after each
        validation or at the end of training).

        Parameters
        ----------
        val_metrics : dict, optional
            Validation metrics, e.g. {"val_loss": 1.23, "val_acc": 0.87}.

        Returns
        -------
        dict
            The complete step data record (also appended to self.step_data).
        """
        if not self._interval_started:
            self._start_interval()
        duration = time.time() - self._step_start_time
        step_index = len(self.step_data)

        rec: Dict[str, Any] = {
            "step": step_index,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration, 4),
        }

        # ── Loss ──
        if self.config.track_loss and self._step_batch_losses:
            losses = self._step_batch_losses
            rec["loss"] = {
                "train_mean": round(sum(losses) / len(losses), 6),
                "train_min": round(min(losses), 6),
                "train_max": round(max(losses), 6),
                "train_first": round(losses[0], 6),
                "train_last": round(losses[-1], 6),
                "train_std": round(
                    (sum((l - sum(losses) / len(losses)) ** 2 for l in losses) / len(losses)) ** 0.5,
                    6,
                ) if len(losses) > 1 else 0.0,
                "num_batches": len(losses),
            }
            if val_metrics:
                rec["loss"]["val"] = {
                    k: round(v.item() if isinstance(v, torch.Tensor) else float(v), 6)
                    for k, v in val_metrics.items()
                }

        # ── Throughput ──
        if self.config.track_throughput and duration > 0:
            rec["throughput"] = {
                "samples_processed": self._step_samples_processed,
                "tokens_processed": self._step_tokens_processed,
                "samples_per_second": round(self._step_samples_processed / duration, 2),
                "tokens_per_second": round(self._step_tokens_processed / duration, 2),
                "batches_per_second": round(len(self._step_batch_losses) / duration, 2),
                "seconds_per_batch": round(
                    duration / max(len(self._step_batch_losses), 1), 4
                ),
            }

        # ── Memory ──
        if self.config.track_memory:
            rec["memory"] = self._snapshot_memory()

        # ── System ──
        if self.config.track_system_resources:
            rec["system"] = self._snapshot_system()

        # ── Profiler (attached by profile_step) ──
        if self._profiler_snapshot is not None:
            rec["profiler"] = self._profiler_snapshot

        # ── Log counts ──
        rec["log_counts"] = {
            "console": len(self.console_logs),
            "error": len(self.error_logs),
        }

        # ── Layer health ──
        if self.config.track_layer_health and self._model:
            rec["layer_health"] = self._compute_layer_health()

        # ── Sustainability ──
        if self.config.track_sustainability:
            rec["sustainability"] = self._compute_sustainability(rec, step_index)

        # ── Carbon emissions (CodeCarbon) ──
        if self.config.track_carbon_emissions and self._carbon_tracker:
            try:
                cumulative_co2 = self._carbon_tracker.flush() or 0.0
                # Access tracker's internal output for energy data
                data = self._carbon_tracker._output
                cumulative_energy = (
                    getattr(data, "energy_consumed", 0.0) if data else 0.0
                )

                step_co2 = cumulative_co2 - self._carbon_prev_emissions
                step_energy = cumulative_energy - self._carbon_prev_energy

                samples = rec.get("throughput", {}).get("samples_processed", 0)

                rec["carbon_emissions"] = {
                    "step_co2_kg": round(step_co2, 10),
                    "step_energy_kwh": round(step_energy, 10),
                    "cumulative_co2_kg": round(cumulative_co2, 10),
                    "cumulative_energy_kwh": round(cumulative_energy, 10),
                    "co2_per_sample_kg": round(
                        step_co2 / max(samples, 1), 12
                    ),
                    "co2_per_second_kg": round(
                        step_co2 / max(duration, 1e-9), 12
                    ),
                    "power_draw_watts": round(
                        step_energy * 3.6e6 / max(duration, 1e-9), 2
                    ),  # kWh -> J / s = W
                    "country_iso_code": self.config.carbon_country_iso,
                }

                self._carbon_prev_emissions = cumulative_co2
                self._carbon_prev_energy = cumulative_energy
            except Exception as e:
                self._log.warning(f"Carbon snapshot failed: {e}")

        self.step_data.append(rec)
        self._log.info(f"Registering step {step_index} in backend")
        self._register_backend_step(rec)
        self._await_backend_status(rec) # Poll the backend for the status of the session 
        self._interval_started = False
        self._start_interval()

        loss_str = rec.get("loss", {}).get("train_mean", "N/A")
        self._log.info(
            f"--- Step {step_index} done | {duration:.2f}s | loss={loss_str} ---"
        )
        return rec


    def _poll_backend_status(self, rec: Dict[str, Any]) -> Optional[str]:
        """Poll the backend for the status of the session."""
        url = f"{self._backend_base_url}/sessions/{self._backend_session_id}/status"
        req = Request(url, method="GET")
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return data

    def _save_checkpoint(self) -> None:
        """Save model state_dict to a checkpoint file."""
        if self._model is None:
            self._log.warning("No model registered, skipping checkpoint save.")
            return
        checkpoint_dir = os.path.join("observer_reports", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(
            checkpoint_dir,
            f"{self.run_name}_timeout_checkpoint.pt",
        )
        torch.save(self._model.state_dict(), path)
        self._log.info(f"Checkpoint saved -> {path}")

    def _stop_backend_session(self) -> None:
        """Tell backend this session is stopped (timeout auto-stop)."""
        url = f"{self._backend_base_url}/sessions/{self._backend_session_id}/action"
        body = json.dumps({"action": "stop"}).encode()
        req = Request(url, method="POST", data=body,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10) as resp:
                resp.read()
        except Exception as e:
            self._log.warning(f"Failed to stop backend session: {e}")

        
    def _await_backend_status(self, rec: Dict[str, Any]) -> None:
        """Await the status of the session with exponential backoff timeout."""
        self._log.info(f"Awaiting backend status for session {self._backend_session_id}...")
        elapsed = 0.0
        delay = 1.0

        while True:
            status = self._poll_backend_status(rec)
            self._log.info(f"Session status: {status}")
            if status == "running":
                return
            if status in ("completed", "stopped"):
                self._log.info(f"Training {status}")
                self.close()
                return
            if status == "failed":
                self._log.info("Training failed")
                self.close()
                return
            if status in ("paused", "pending"):
                if elapsed >= self.config.pending_timeout:
                    self._log.warning(
                        f"Pending timeout ({self.config.pending_timeout}s) reached. "
                        "Saving model and stopping."
                    )
                    self._save_checkpoint()
                    self._stop_backend_session()
                    self.close()
                    return
                sleep_time = min(delay, self.config.pending_timeout - elapsed)
                time.sleep(sleep_time)
                elapsed += sleep_time
                delay = min(delay * 2, 32.0)

    # ==================================================================
    # PyTorch Profiler
    # ==================================================================

    def profile_step(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor):
        """
        Run one forward + backward pass under the PyTorch profiler.

        Per-layer timing uses PyTorch's record_function(): each parameter
        layer's forward/backward is wrapped in record_function(layer_name)
        so the profiler natively attributes time to layers (see per_layer
        in the snapshot). with_modules=True exists but only works for
        TorchScript; for eager mode, record_function + hooks is the
        recommended approach.

        Call this *instead of* a normal training step (e.g. first batch).
        Returns (logits, loss) like model(x, y).
        """
        param_layers = self._get_parameter_layers(model)
        layer_names = [name for name, _ in param_layers]
        self._profiler_layer_names = layer_names

        # Wrap each layer's forward/backward in record_function so the
        # profiler attributes time to layer names (PyTorch's per-layer story for eager mode).
        fwd_ctx: Dict[int, Any] = {}
        bwd_ctx: Dict[int, Any] = {}

        def make_forward_hooks(layer_name: str, mod_id: int):
            def pre_hook(module, input):
                ctx = record_function(layer_name)
                fwd_ctx[mod_id] = ctx
                ctx.__enter__()

            def post_hook(module, input, output):
                fwd_ctx[mod_id].__exit__(None, None, None)
                del fwd_ctx[mod_id]

            return pre_hook, post_hook

        def make_backward_hooks(layer_name: str, bwd_name: str, mod_id: int):
            def bwd_pre_hook(module, grad_output):
                ctx = record_function(bwd_name)
                bwd_ctx[mod_id] = ctx
                ctx.__enter__()

            def bwd_hook(module, grad_input, grad_output):
                bwd_ctx[mod_id].__exit__(None, None, None)
                del bwd_ctx[mod_id]

            return bwd_pre_hook, bwd_hook

        handles: List[Any] = []
        for name, module in param_layers:
            mid = id(module)
            bwd_name = f"{name}.backward"
            pre, post = make_forward_hooks(name, mid)
            handles.append(module.register_forward_pre_hook(pre))
            handles.append(module.register_forward_hook(post))
            bwd_pre, bwd_post = make_backward_hooks(name, bwd_name, mid)
            handles.append(module.register_full_backward_pre_hook(bwd_pre))
            handles.append(module.register_full_backward_hook(bwd_post))

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        use_stack = (
            self.config.profiler_with_stack
            or self.config.profiler_group_by_stack_n > 0
        )

        try:
            with profile(
                activities=activities,
                record_shapes=self.config.profiler_record_shapes,
                profile_memory=self.config.profiler_profile_memory,
                with_stack=use_stack,
            ) as prof:
                with record_function("model_forward"):
                    logits, loss = model(x, y)
                with record_function("model_backward"):
                    if loss is not None:
                        loss.backward()
            self._profiler_snapshot = self._parse_profiler(prof)
        finally:
            for h in handles:
                h.remove()

        return logits, loss

    @staticmethod
    def _get_parameter_layers(model: nn.Module) -> List[tuple]:
        """Return (name, module) for every module that has its own parameters."""
        return [
            (name, module)
            for name, module in model.named_modules()
            if name and sum(p.numel() for p in module.parameters(recurse=False)) > 0
        ]

    def _parse_profiler(self, prof) -> Dict[str, Any]:
        """Extract structured data from a PyTorch profiler run."""
        avgs = prof.key_averages()

        # ── Top operations by CPU time ──
        top_ops = []
        for evt in sorted(avgs, key=lambda e: e.cpu_time_total, reverse=True)[
            : self.config.profiler_top_n_ops
        ]:
            op: Dict[str, Any] = {
                "name": evt.key,
                "calls": evt.count,
                "cpu_time_us": evt.cpu_time_total,
                "cpu_time_ms": round(evt.cpu_time_total / 1000, 3),
                "avg_cpu_us": round(evt.cpu_time_total / max(evt.count, 1), 1),
            }
            if torch.cuda.is_available():
                op["cuda_time_us"] = getattr(evt, "cuda_time_total", 0) or 0
                op["cuda_time_ms"] = round(op["cuda_time_us"] / 1000, 3)
            if evt.cpu_memory_usage:
                op["cpu_mem_bytes"] = evt.cpu_memory_usage
            if hasattr(evt, "cuda_memory_usage") and evt.cuda_memory_usage:
                op["cuda_mem_bytes"] = evt.cuda_memory_usage
            if hasattr(evt, "input_shapes") and evt.input_shapes:
                op["input_shapes"] = [str(s) for s in evt.input_shapes]
            top_ops.append(op)

        total_cpu = sum(e.cpu_time_total for e in avgs)
        total_cuda = (
            sum(getattr(e, "cuda_time_total", 0) or 0 for e in avgs) if torch.cuda.is_available() else 0
        )

        # ── Forward / backward breakdown ──
        fwd_time = bwd_time = 0
        for evt in avgs:
            kl = evt.key.lower()
            if "forward" in kl:
                fwd_time += evt.cpu_time_total
            elif "backward" in kl or "autograd" in kl or "grad" in kl:
                bwd_time += evt.cpu_time_total

        # ── Operation categories ──
        cats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"cpu_us": 0, "cuda_us": 0, "calls": 0}
        )
        cat_map = [
            (["matmul", "mm", "addmm", "bmm", "linear"], "matrix_ops"),
            (["softmax"], "softmax"),
            (["layer_norm", "norm", "batch_norm"], "normalization"),
            (["dropout"], "dropout"),
            (["embedding"], "embedding"),
            (["cross_entropy", "nll_loss", "loss"], "loss"),
            (["relu", "gelu", "silu", "tanh", "sigmoid"], "activation_fn"),
            (["conv"], "convolution"),
            (["adam", "sgd", "optim"], "optimizer"),
            (["forward"], "forward_pass"),
            (["backward", "autograd"], "backward_pass"),
        ]
        for evt in avgs:
            kl = evt.key.lower()
            matched = "other"
            for keywords, category in cat_map:
                if any(kw in kl for kw in keywords):
                    matched = category
                    break
            cats[matched]["cpu_us"] += evt.cpu_time_total
            cats[matched]["cuda_us"] += (getattr(evt, "cuda_time_total", 0) or 0) if torch.cuda.is_available() else 0
            cats[matched]["calls"] += evt.count

        categories = {}
        for cat, stats in cats.items():
            categories[cat] = {
                "cpu_time_ms": round(stats["cpu_us"] / 1000, 3),
                "cuda_time_ms": round(stats["cuda_us"] / 1000, 3),
                "calls": stats["calls"],
                "pct_cpu": round(100 * stats["cpu_us"] / max(total_cpu, 1), 2),
            }

        # ── Per-stack grouping (which call stacks use the most compute) ──
        top_by_stack: List[Dict[str, Any]] = []
        use_stack = (
            self.config.profiler_with_stack
            or self.config.profiler_group_by_stack_n > 0
        )
        if self.config.profiler_group_by_stack_n > 0 and use_stack:
            try:
                avgs_stack = prof.key_averages(
                    group_by_stack_n=self.config.profiler_group_by_stack_n
                )
            except Exception:
                avgs_stack = []
            for evt in sorted(
                avgs_stack,
                key=lambda e: e.cpu_time_total,
                reverse=True,
            )[: self.config.profiler_top_n_stacks]:
                key_str = (
                    evt.key
                    if isinstance(evt.key, str)
                    else str(evt.key)
                )
                lines = key_str.strip().split("\n")
                op_name = lines[0].strip() if lines else key_str
                stack_frames = [ln.strip() for ln in lines[1:] if ln.strip()]
                entry: Dict[str, Any] = {
                    "op": op_name,
                    "cpu_time_us": evt.cpu_time_total,
                    "cpu_time_ms": round(evt.cpu_time_total / 1000, 3),
                    "calls": evt.count,
                    "avg_cpu_us": round(
                        evt.cpu_time_total / max(evt.count, 1), 1
                    ),
                    "pct_cpu": round(
                        100 * evt.cpu_time_total / max(total_cpu, 1), 2
                    ),
                }
                if stack_frames:
                    entry["stack_frames"] = stack_frames
                if torch.cuda.is_available():
                    entry["cuda_time_us"] = getattr(evt, "cuda_time_total", 0) or 0
                    entry["cuda_time_ms"] = round(
                        entry["cuda_time_us"] / 1000, 3
                    )
                top_by_stack.append(entry)

        result: Dict[str, Any] = {
            "total_cpu_time_ms": round(total_cpu / 1000, 3),
            "total_cuda_time_ms": round(total_cuda / 1000, 3),
            "forward_time_ms": round(fwd_time / 1000, 3),
            "backward_time_ms": round(bwd_time / 1000, 3),
            "fwd_bwd_ratio": round(fwd_time / max(bwd_time, 1), 4),
            "num_unique_ops": len(avgs),
            "top_operations": top_ops,
            "operation_categories": categories,
        }
        if top_by_stack:
            result["top_by_stack"] = top_by_stack

        # ── Per-layer from record_function() events (PyTorch native for eager mode) ──
        layer_names = getattr(self, "_profiler_layer_names", None)
        if layer_names:
            key_to_evt = {evt.key: evt for evt in avgs}
            total_fwd_us = total_bwd_us = 0.0
            per_layer: List[Dict[str, Any]] = []
            for name in layer_names:
                evt_fwd = key_to_evt.get(name)
                evt_bwd = key_to_evt.get(f"{name}.backward")
                fwd_us = evt_fwd.cpu_time_total if evt_fwd else 0.0
                bwd_us = evt_bwd.cpu_time_total if evt_bwd else 0.0
                if torch.cuda.is_available():
                    if evt_fwd:
                        fwd_us = fwd_us + (getattr(evt_fwd, "cuda_time_total", 0) or 0)
                    if evt_bwd:
                        bwd_us = bwd_us + (getattr(evt_bwd, "cuda_time_total", 0) or 0)
                total_fwd_us += fwd_us
                total_bwd_us += bwd_us
                row_us = fwd_us + bwd_us
                per_layer.append({
                    "layer": name,
                    "forward_us": round(fwd_us, 1),
                    "forward_ms": round(fwd_us / 1000, 4),
                    "backward_us": round(bwd_us, 1),
                    "backward_ms": round(bwd_us / 1000, 4),
                    "total_us": round(row_us, 1),
                    "total_ms": round(row_us / 1000, 4),
                })
            total_layer_us = total_fwd_us + total_bwd_us
            for row in per_layer:
                fu, bu = row["forward_us"], row["backward_us"]
                tu = fu + bu
                row["pct_forward"] = round(100 * fu / max(total_fwd_us, 1), 2)
                row["pct_backward"] = round(100 * bu / max(total_bwd_us, 1), 2)
                row["pct_total"] = round(100 * tu / max(total_layer_us, 1), 2)
            per_layer.sort(key=lambda r: r["total_us"], reverse=True)
            result["per_layer"] = per_layer
            result["num_layers"] = len(per_layer)

        return result

    # ==================================================================
    # Snapshot helpers
    # ==================================================================

    def _snapshot_memory(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info()
            stats["process_rss_mb"] = round(mem.rss / 1024 / 1024, 2)
            stats["process_vms_mb"] = round(mem.vms / 1024 / 1024, 2)
        except ImportError:
            stats["process_rss_mb"] = None
            stats["_note"] = "install psutil for process memory tracking"

        if torch.cuda.is_available():
            stats["cuda_allocated_mb"] = round(
                torch.cuda.memory_allocated() / 1024 / 1024, 2
            )
            stats["cuda_reserved_mb"] = round(
                torch.cuda.memory_reserved() / 1024 / 1024, 2
            )
            stats["cuda_peak_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / 1024 / 1024, 2
            )
            stats["cuda_peak_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024, 2
            )
        return stats

    def _snapshot_system(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        try:
            import psutil
            vm = psutil.virtual_memory()
            stats["cpu_percent"] = psutil.cpu_percent(interval=None)
            stats["ram_total_gb"] = round(vm.total / 1024 ** 3, 2)
            stats["ram_used_gb"] = round(vm.used / 1024 ** 3, 2)
            stats["ram_percent"] = vm.percent
        except ImportError:
            stats["_note"] = "install psutil for system resource tracking"

        if torch.cuda.is_available():
            stats["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            stats["gpu_total_mem_mb"] = round(props.total_mem / 1024 / 1024, 2)

        return stats

    # ==================================================================
    # Layer health & sustainability
    # ==================================================================

    def _compute_layer_health(self) -> Dict[str, Any]:
        """
        Aggregate per-batch activation/gradient stats into per-layer summaries.
        Also snapshot current weight tensor statistics from the model.
        """
        result: Dict[str, Any] = {"layers": {}, "activation_correlations": []}
        if not self._model:
            return result

        threshold = self.config.layer_health_zero_threshold
        param_layers = self._get_parameter_layers(self._model)
        sequential_names = [name for name, _ in param_layers]

        for name, module in param_layers:
            layer_data: Dict[str, Any] = {}

            # ── Activation health ──
            act_stats = self._layer_activation_stats.get(name, [])
            if act_stats:
                means = [s["mean"] for s in act_stats]
                stds = [s["std"] for s in act_stats]
                sparsities = [s["sparsity"] for s in act_stats]
                n = len(means)
                act_mean = sum(means) / n
                act_var_of_means = (
                    sum((m - act_mean) ** 2 for m in means) / n if n > 1 else 0.0
                )
                layer_data["activation_mean"] = round(act_mean, 6)
                layer_data["activation_std"] = round(sum(stds) / n, 6)
                layer_data["activation_var_of_means"] = round(act_var_of_means, 10)
                layer_data["activation_sparsity"] = round(sum(sparsities) / n, 6)
                layer_data["num_batches"] = n
            else:
                layer_data["activation_mean"] = None
                layer_data["num_batches"] = 0

            # ── Gradient health ──
            grad_stats = self._layer_gradient_stats.get(name, [])
            if grad_stats:
                norms = [s["norm"] for s in grad_stats]
                sparsities_g = [s["sparsity"] for s in grad_stats]
                gn = len(norms)
                grad_norm_mean = sum(norms) / gn
                grad_norm_std = (
                    (sum((n_ - grad_norm_mean) ** 2 for n_ in norms) / gn) ** 0.5
                    if gn > 1 else 0.0
                )
                layer_data["gradient_norm_mean"] = round(grad_norm_mean, 8)
                layer_data["gradient_norm_std"] = round(grad_norm_std, 8)
                layer_data["gradient_sparsity"] = round(sum(sparsities_g) / gn, 6)
            else:
                layer_data["gradient_norm_mean"] = None

            # ── Weight health (snapshot) ──
            try:
                weight = None
                for pname, param in module.named_parameters(recurse=False):
                    if "weight" in pname:
                        weight = param
                        break
                if weight is not None:
                    with torch.no_grad():
                        w = weight.detach().float()
                        layer_data["weight_sparsity"] = round(
                            (w.abs() < threshold).float().mean().item(), 6
                        )
                        layer_data["weight_mean"] = round(w.abs().mean().item(), 6)
                        layer_data["weight_std"] = round(w.std().item(), 6)
                        layer_data["weight_norm"] = round(w.norm(2).item(), 6)
            except Exception:
                pass

            # ── Derived flags ──
            act_var = layer_data.get("activation_var_of_means")
            w_sparsity = layer_data.get("weight_sparsity", 0)
            g_norm = layer_data.get("gradient_norm_mean")
            act_std = layer_data.get("activation_std")

            layer_data["is_dead"] = (
                act_var is not None and act_var < 1e-8
                and w_sparsity > 0.9
            )
            layer_data["has_frozen_output"] = (
                act_var is not None and act_var < 1e-8
            )
            layer_data["has_vanishing_gradients"] = (
                g_norm is not None and g_norm < 1e-7
            )
            layer_data["has_near_zero_weights"] = w_sparsity > 0.5
            layer_data["has_low_activation_variance"] = (
                act_std is not None and act_std < 1e-6
            )

            result["layers"][name] = layer_data

        # ── Activation correlations between consecutive layers ──
        for i in range(len(sequential_names) - 1):
            name_a = sequential_names[i]
            name_b = sequential_names[i + 1]
            means_a = [s["mean"] for s in self._layer_activation_stats.get(name_a, [])]
            means_b = [s["mean"] for s in self._layer_activation_stats.get(name_b, [])]
            n = min(len(means_a), len(means_b))
            if n < 5:
                continue
            means_a, means_b = means_a[:n], means_b[:n]
            avg_a = sum(means_a) / n
            avg_b = sum(means_b) / n
            cov = sum((a - avg_a) * (b - avg_b) for a, b in zip(means_a, means_b)) / n
            std_a = (sum((a - avg_a) ** 2 for a in means_a) / n) ** 0.5
            std_b = (sum((b - avg_b) ** 2 for b in means_b) / n) ** 0.5
            denom = std_a * std_b
            corr = round(cov / denom, 6) if denom > 1e-12 else 0.0
            if abs(corr) > 0.8:
                result["activation_correlations"].append({
                    "layer_a": name_a,
                    "layer_b": name_b,
                    "correlation": corr,
                })

        return result

    def _compute_sustainability(self, rec: Dict[str, Any], step_index: int) -> Dict[str, Any]:
        """
        Derive sustainability metrics from existing step data:
        layer efficiency, marginal loss, compute cost, cumulative compute.
        """
        sus: Dict[str, Any] = {}

        # ── Per-layer compute efficiency ──
        per_layer = (rec.get("profiler") or {}).get("per_layer", [])
        arch_layers = self.model_architecture.get("layers", {})
        if per_layer and arch_layers:
            layer_efficiency = []
            for pl in per_layer:
                layer_name = pl.get("layer", "")
                arch_info = arch_layers.get(layer_name)
                if arch_info:
                    pct_compute = pl.get("pct_total", 0)
                    pct_params = arch_info.get("pct_of_total", 0)
                    ratio = round(pct_compute / max(pct_params, 0.01), 4)
                    layer_efficiency.append({
                        "layer": layer_name,
                        "pct_compute": pct_compute,
                        "pct_parameters": pct_params,
                        "compute_to_param_ratio": ratio,
                    })
            sus["layer_efficiency"] = layer_efficiency

        # ── Marginal loss improvement ──
        curr_loss = (rec.get("loss") or {}).get("train_mean")
        if curr_loss is not None and self.step_data:
            prev_loss = (self.step_data[-1].get("loss") or {}).get("train_mean")
            if prev_loss is not None and prev_loss > 0:
                abs_imp = round(prev_loss - curr_loss, 6)
                pct_imp = round(100 * abs_imp / prev_loss, 4)
                first_loss = (self.step_data[0].get("loss") or {}).get(
                    "train_mean", prev_loss
                )
                cum_imp = round(first_loss - curr_loss, 6)
                moc = (
                    round(abs_imp / max(cum_imp, 1e-9), 4)
                    if cum_imp > 0 else None
                )
                sus["marginal_loss"] = {
                    "previous_loss": prev_loss,
                    "current_loss": curr_loss,
                    "absolute_improvement": abs_imp,
                    "pct_improvement": pct_imp,
                    "cumulative_improvement": cum_imp,
                    "marginal_over_cumulative": moc,
                }
        elif curr_loss is not None:
            sus["marginal_loss"] = {
                "previous_loss": None,
                "current_loss": curr_loss,
                "absolute_improvement": None,
                "pct_improvement": None,
                "cumulative_improvement": 0.0,
                "marginal_over_cumulative": None,
            }

        # ── Step compute cost ──
        sus["step_compute_cost"] = {
            "duration_seconds": rec.get("duration_seconds", 0),
            "profiler_cpu_time_ms": (rec.get("profiler") or {}).get("total_cpu_time_ms"),
            "profiler_cuda_time_ms": (rec.get("profiler") or {}).get("total_cuda_time_ms"),
            "samples_processed": (rec.get("throughput") or {}).get("samples_processed", 0),
            "samples_per_second": (rec.get("throughput") or {}).get("samples_per_second"),
        }

        # ── Cumulative compute ──
        cum_duration = sum(
            s.get("duration_seconds", 0) for s in self.step_data
        ) + rec.get("duration_seconds", 0)
        cum_samples = sum(
            (s.get("throughput") or {}).get("samples_processed", 0)
            for s in self.step_data
        ) + (rec.get("throughput") or {}).get("samples_processed", 0)
        sus["cumulative_compute"] = {
            "total_duration_seconds": round(cum_duration, 4),
            "total_samples_processed": cum_samples,
            "steps_completed": len(self.step_data) + 1,
        }

        return sus

    # ==================================================================
    # Export
    # ==================================================================

    def get_step_report(self, step_index: int) -> Optional[Dict]:
        """Retrieve the data for a specific step (interval) index."""
        for rec in self.step_data:
            if rec["step"] == step_index:
                return rec
        return None

    def export(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Export all collected data to a JSON file and return the full report dict.

        Parameters
        ----------
        filepath : str, optional
            Destination path. Defaults to observer_reports/<run_id>.json.
        """
        report = {
            "session": self.session,
            "hyperparameters": self.hyperparameters,
            "model_architecture": self.model_architecture,
            "steps": self.step_data,
            "console_logs": self.console_logs if self.config.track_console_logs else [],
            "error_logs": self.error_logs if self.config.track_error_logs else [],
            "summary": self._build_summary(),
        }

        if filepath is None:
            filepath = os.path.join("observer_reports", f"{self.run_id}.json")

        dirn = os.path.dirname(filepath)
        if dirn:
            os.makedirs(dirn, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        self._log.info(f"Report exported -> {filepath}")
        return report

    def _build_summary(self) -> Dict[str, Any]:
        if not self.step_data:
            return {"status": "no_data"}

        summary: Dict[str, Any] = {
            "total_steps": len(self.step_data),
            "total_duration_s": round(
                sum(s["duration_seconds"] for s in self.step_data), 2
            ),
        }

        # Loss trend
        means = [
            s["loss"]["train_mean"]
            for s in self.step_data
            if "loss" in s and "train_mean" in s.get("loss", {})
        ]
        if means:
            summary["loss_trend"] = {
                "first": means[0],
                "last": means[-1],
                "best": min(means),
                "worst": max(means),
                "delta": round(means[0] - means[-1], 6),
                "improved": means[-1] < means[0],
            }

        # Throughput averages
        tps = [
            s["throughput"]["tokens_per_second"]
            for s in self.step_data
            if "throughput" in s
        ]
        if tps:
            summary["avg_tokens_per_sec"] = round(sum(tps) / len(tps), 2)

        # Profiler highlights (from last profiled step)
        for s in reversed(self.step_data):
            if "profiler" in s:
                p = s["profiler"]
                pl = p.get("per_layer") or []
                top_layer = pl[0] if pl else None
                summary["profiler_highlight"] = {
                    "fwd_bwd_ratio": p.get("fwd_bwd_ratio"),
                    "top_op": p["top_operations"][0]["name"] if p.get("top_operations") else None,
                    "top_op_pct": round(
                        100 * p["top_operations"][0]["cpu_time_us"] / max(p.get("total_cpu_time_ms", 0) * 1000, 1), 2
                    ) if p.get("top_operations") else None,
                    "top_layer": top_layer["layer"] if top_layer else None,
                    "top_layer_pct": top_layer["pct_total"] if top_layer else None,
                }
                break

        # ── Sustainability summary ──
        sustainability_summary: Dict[str, Any] = {}

        # Optimal stop & wasted compute
        if len(means) > 2:
            first_loss = means[0]
            optimal_stop = len(means) - 1
            for i in range(2, len(means)):
                cum_imp = first_loss - means[i]
                marginal_imp = means[i - 1] - means[i]
                if cum_imp > 0 and marginal_imp / cum_imp < 0.05:
                    optimal_stop = i - 1
                    break

            total_dur = sum(s.get("duration_seconds", 0) for s in self.step_data)
            wasted_dur = sum(
                s.get("duration_seconds", 0)
                for s in self.step_data[optimal_stop + 1:]
            )
            wasted_pct = round(100 * wasted_dur / max(total_dur, 1e-9), 2)
            sustainability_summary["optimal_stop_step"] = optimal_stop
            sustainability_summary["wasted_steps"] = len(self.step_data) - 1 - optimal_stop
            sustainability_summary["wasted_compute_pct"] = wasted_pct
            sustainability_summary["wasted_duration_seconds"] = round(wasted_dur, 2)

        # Parameter efficiency score
        last_sus = None
        for s in reversed(self.step_data):
            le = (s.get("sustainability") or {}).get("layer_efficiency")
            if le:
                last_sus = le
                break
        if last_sus:
            ratios = [le_item["compute_to_param_ratio"] for le_item in last_sus]
            deviations = [abs(math.log2(max(r, 0.001))) for r in ratios]
            avg_dev = sum(deviations) / len(deviations) if deviations else 0
            sustainability_summary["parameter_efficiency_score"] = max(
                0, round(100 - avg_dev * 20, 1)
            )

        # Dead / vanishing / frozen layers (from last step's layer_health)
        last_health = None
        for s in reversed(self.step_data):
            lh = s.get("layer_health")
            if lh:
                last_health = lh
                break
        if last_health:
            layers_data = last_health.get("layers", {})
            sustainability_summary["dead_layers"] = [
                name for name, d in layers_data.items() if d.get("is_dead")
            ]
            sustainability_summary["vanishing_gradient_layers"] = [
                name for name, d in layers_data.items()
                if d.get("has_vanishing_gradients")
            ]
            sustainability_summary["frozen_output_layers"] = [
                name for name, d in layers_data.items()
                if d.get("has_frozen_output")
            ]

        # Carbon totals
        carbon_steps = [
            s["carbon_emissions"]
            for s in self.step_data
            if s.get("carbon_emissions")
        ]
        if carbon_steps:
            total_co2 = sum(c["step_co2_kg"] for c in carbon_steps)
            total_energy = sum(c["step_energy_kwh"] for c in carbon_steps)
            total_samples = sum(
                s.get("throughput", {}).get("samples_processed", 0)
                for s in self.step_data
            )
            sustainability_summary["total_co2_kg"] = round(total_co2, 8)
            sustainability_summary["total_energy_kwh"] = round(total_energy, 8)
            sustainability_summary["co2_per_step_avg_kg"] = round(
                total_co2 / len(carbon_steps), 10
            )
            sustainability_summary["co2_per_1k_samples_kg"] = (
                round(total_co2 / max(total_samples / 1000, 1e-9), 10)
                if total_samples
                else None
            )
            sustainability_summary["avg_power_draw_watts"] = round(
                sum(c["power_draw_watts"] for c in carbon_steps)
                / len(carbon_steps),
                2,
            )
            # Wasted carbon (steps past optimal stop)
            if "optimal_stop_step" in sustainability_summary:
                opt = sustainability_summary["optimal_stop_step"]
                wasted_carbon = sum(
                    s.get("carbon_emissions", {}).get("step_co2_kg", 0)
                    for s in self.step_data[opt + 1 :]
                )
                sustainability_summary["wasted_co2_kg"] = round(wasted_carbon, 10)

        if sustainability_summary:
            summary["sustainability"] = sustainability_summary

        return summary

    # ==================================================================
    # Cleanup
    # ==================================================================

    def close(self):
        """Finalize session metadata and remove log handlers. Flushes any pending steps."""
        if self._interval_started and (
            self._step_batch_losses or self._step_samples_processed or self._step_tokens_processed
        ):
            self.flush()
        for h in self._capture_handlers:
            logging.getLogger().removeHandler(h)
        self._capture_handlers.clear()
        for h in self._health_hooks:
            h.remove()
        self._health_hooks.clear()
        # Stop CodeCarbon tracker
        if self._carbon_tracker:
            try:
                self._carbon_tracker.stop()
            except Exception:
                pass
            self._carbon_tracker = None
        self.session["ended_at"] = datetime.now().isoformat()
        self._log.info("Observer closed.")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (
            f"Observer(project={self.project_id!r}, run={self.run_name!r}, "
            f"steps={len(self.step_data)})"
        )
