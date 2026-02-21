"""
Observer - PyTorch Training Pipeline Monitor

Lightweight training observer that captures session metadata, hyperparameters,
model architecture, and per-epoch metrics (loss, throughput, memory, system, profiler).

Collects:
  - Session (run id, device, config snapshot)
  - Hyperparameters (user-supplied)
  - Model architecture (layer map, module tree, optional layer graph)
  - Per-epoch: loss, throughput, memory, system, profiler (when profile_step used), log_counts
"""

import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    profile_at_step: Optional[int] = 0  # Which step within each epoch to profile (None = never)
    profiler_record_shapes: bool = True
    profiler_profile_memory: bool = True
    profiler_with_stack: bool = False
    profiler_top_n_ops: int = 20
    # Per-stack grouping: set group_by_stack_n > 0 to see which call stacks use the most compute.
    # with_stack is auto-enabled when group_by_stack_n > 0. Stack depth is often 5 (PyTorch limit).
    profiler_group_by_stack_n: int = 0
    profiler_top_n_stacks: int = 20

    # Log level for the observer's own logger
    log_level: int = logging.INFO


# ---------------------------------------------------------------------------
# Log capture handler
# ---------------------------------------------------------------------------

class _LogCaptureHandler(logging.Handler):
    """Lightweight handler that appends log records into a list."""

    def __init__(self, store: list, min_level: int = logging.DEBUG):
        super().__init__(min_level)
        self.store = store

    def emit(self, record):
        try:
            self.store.append({
                "ts": datetime.now().isoformat(),
                "level": record.levelname,
                "msg": self.format(record),
                "module": getattr(record, "module", ""),
                "lineno": getattr(record, "lineno", 0),
            })
        except Exception:
            pass  # never break the training loop


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class Observer:
    """
    Epoch-by-epoch training observer that collects rich diagnostics.

    Quickstart
    ----------
    >>> config = ObserverConfig(track_profiler=True, profile_at_step=0)
    >>> obs = Observer(api_key="key-123", project_id="proj-abc", config=config)
    >>> obs.log_hyperparameters({"lr": 3e-4, "batch_size": 64, ...})
    >>> obs.register_model(model)
    >>>
    >>> for epoch in range(num_epochs):
    ...     for step, (x, y) in enumerate(loader):
    ...         if obs.should_profile(step):
    ...             logits, loss = obs.profile_step(model, x, y)
    ...         else:
    ...             logits, loss = model(x, y)
    ...             loss.backward()
    ...         optimizer.step(); optimizer.zero_grad()
    ...         obs.step(epoch, step, loss, batch_size=x.size(0), seq_length=x.size(1))
    ...     val_metrics = evaluate()
    ...     report = obs.end_epoch(epoch, val_metrics=val_metrics)
    >>>
    >>> obs.export("observer_reports/run.json")
    >>> obs.close()
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        config: Optional[ObserverConfig] = None,
        run_name: Optional[str] = None,
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.config = config or ObserverConfig()
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_id = f"{project_id}_{self.run_name}"

        # ── Session metadata ──
        self.session: Dict[str, Any] = {
            "api_key_prefix": api_key[:8] + "..." if len(api_key) > 8 else "***",
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
        self.epoch_data: List[Dict[str, Any]] = []
        self.console_logs: List[Dict] = []
        self.error_logs: List[Dict] = []

        # ── Ephemeral (per-epoch) state ──
        self._current_epoch: Optional[int] = None
        self._epoch_start_time: float = 0.0
        self._epoch_batch_losses: List[float] = []
        self._epoch_batch_times: List[float] = []
        self._epoch_tokens_processed: int = 0
        self._epoch_samples_processed: int = 0
        self._profiler_snapshot: Optional[Dict] = None

        self._model: Optional[nn.Module] = None

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
            h = _LogCaptureHandler(self.console_logs, logging.INFO)
            h.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger().addHandler(h)
            self._capture_handlers.append(h)
        if self.config.track_error_logs:
            h = _LogCaptureHandler(self.error_logs, logging.WARNING)
            h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logging.getLogger().addHandler(h)
            self._capture_handlers.append(h)

        self._log.info(
            f"Initialized | project={project_id} | run={self.run_name} | "
            f"device={self.session['device']}"
        )

    # ==================================================================
    # Hyperparameters
    # ==================================================================

    def log_hyperparameters(self, params: dict):
        """Record training hyperparameters (call before training starts)."""
        self.hyperparameters.update(params)
        self._log.info(f"Hyperparameters logged: {list(params.keys())}")

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

        self._log.info(
            f"Model registered | {total_params:,} params "
            f"({total_params / 1e6:.2f}M) | {len(layer_map)} param layers"
        )

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
    # Epoch lifecycle
    # ==================================================================

    def _start_epoch(self, epoch: int):
        """Internal: reset per-epoch state when a new epoch begins."""
        self._current_epoch = epoch
        self._epoch_start_time = time.time()
        self._epoch_batch_losses.clear()
        self._epoch_batch_times.clear()
        self._epoch_tokens_processed = 0
        self._epoch_samples_processed = 0
        self._profiler_snapshot = None

        if torch.cuda.is_available() and self.config.track_memory:
            torch.cuda.reset_peak_memory_stats()

        self._log.info(f"--- Epoch {epoch} started ---")

    def should_profile(self, step: int) -> bool:
        """Return True if profiling should run on this step."""
        return (
            self.config.track_profiler
            and self.config.profile_at_step is not None
            and step == self.config.profile_at_step
        )

    def step(
        self,
        epoch: int,
        step: int,
        loss,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ):
        """
        Record a single training step. Automatically initialises a new
        epoch the first time a new epoch number is seen.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        step : int
            Batch / step index within the epoch.
        loss : torch.Tensor | float
            Loss value for this step.
        batch_size : int, optional
            Number of samples in this batch.
        seq_length : int, optional
            Sequence length per sample (for token throughput).
        """
        # Auto-start a new epoch when the epoch number changes
        if self._current_epoch is None or self._current_epoch != epoch:
            self._start_epoch(epoch)

        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        self._epoch_batch_losses.append(loss_val)
        self._epoch_batch_times.append(time.time())
        if batch_size:
            self._epoch_samples_processed += batch_size
        if batch_size and seq_length:
            self._epoch_tokens_processed += batch_size * seq_length

    def end_epoch(
        self,
        epoch: int,
        val_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Finalise the epoch: collect all channel data and return the epoch record.

        Parameters
        ----------
        epoch : int
            Epoch number (should match what was passed to start_epoch).
        val_metrics : dict, optional
            Validation metrics, e.g. {"val_loss": 1.23, "val_acc": 0.87}.

        Returns
        -------
        dict
            The complete epoch data record (also appended to self.epoch_data).
        """
        duration = time.time() - self._epoch_start_time

        rec: Dict[str, Any] = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration, 4),
        }

        # ── Loss ──
        if self.config.track_loss and self._epoch_batch_losses:
            losses = self._epoch_batch_losses
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
                "samples_processed": self._epoch_samples_processed,
                "tokens_processed": self._epoch_tokens_processed,
                "samples_per_second": round(self._epoch_samples_processed / duration, 2),
                "tokens_per_second": round(self._epoch_tokens_processed / duration, 2),
                "batches_per_second": round(len(self._epoch_batch_losses) / duration, 2),
                "seconds_per_batch": round(
                    duration / max(len(self._epoch_batch_losses), 1), 4
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

        self.epoch_data.append(rec)

        loss_str = rec.get("loss", {}).get("train_mean", "N/A")
        self._log.info(
            f"--- Epoch {epoch} done | {duration:.2f}s | loss={loss_str} ---"
        )
        return rec

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

        Call this *instead of* a normal training step (typically on the
        first batch of each epoch). Returns (logits, loss) like model(x, y).
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
                op["cuda_time_us"] = evt.cuda_time_total
                op["cuda_time_ms"] = round(evt.cuda_time_total / 1000, 3)
            if evt.cpu_memory_usage:
                op["cpu_mem_bytes"] = evt.cpu_memory_usage
            if hasattr(evt, "cuda_memory_usage") and evt.cuda_memory_usage:
                op["cuda_mem_bytes"] = evt.cuda_memory_usage
            if hasattr(evt, "input_shapes") and evt.input_shapes:
                op["input_shapes"] = [str(s) for s in evt.input_shapes]
            top_ops.append(op)

        total_cpu = sum(e.cpu_time_total for e in avgs)
        total_cuda = (
            sum(e.cuda_time_total for e in avgs) if torch.cuda.is_available() else 0
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
            cats[matched]["cuda_us"] += evt.cuda_time_total if torch.cuda.is_available() else 0
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
                    entry["cuda_time_us"] = evt.cuda_time_total
                    entry["cuda_time_ms"] = round(
                        evt.cuda_time_total / 1000, 3
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
    # Export
    # ==================================================================

    def get_epoch_report(self, epoch: int) -> Optional[Dict]:
        """Retrieve the data for a specific epoch."""
        for rec in self.epoch_data:
            if rec["epoch"] == epoch:
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
            "epochs": self.epoch_data,
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
        if not self.epoch_data:
            return {"status": "no_data"}

        summary: Dict[str, Any] = {
            "total_epochs": len(self.epoch_data),
            "total_duration_s": round(
                sum(e["duration_seconds"] for e in self.epoch_data), 2
            ),
        }

        # Loss trend
        means = [
            e["loss"]["train_mean"]
            for e in self.epoch_data
            if "loss" in e and "train_mean" in e.get("loss", {})
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
            e["throughput"]["tokens_per_second"]
            for e in self.epoch_data
            if "throughput" in e
        ]
        if tps:
            summary["avg_tokens_per_sec"] = round(sum(tps) / len(tps), 2)

        # Profiler highlights (from last profiled epoch)
        for e in reversed(self.epoch_data):
            if "profiler" in e:
                p = e["profiler"]
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

        return summary

    # ==================================================================
    # Cleanup
    # ==================================================================

    def close(self):
        """Finalize session metadata and remove log handlers."""
        for h in self._capture_handlers:
            logging.getLogger().removeHandler(h)
        self._capture_handlers.clear()
        self.session["ended_at"] = datetime.now().isoformat()
        self._log.info("Observer closed.")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (
            f"Observer(project={self.project_id!r}, run={self.run_name!r}, "
            f"epochs={len(self.epoch_data)})"
        )
