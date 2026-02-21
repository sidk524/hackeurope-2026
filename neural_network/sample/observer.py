"""
Observer - PyTorch Training Pipeline Monitor

A comprehensive training observer that integrates PyTorch Profiler with
custom hooks to capture architecture-level and pipeline-level diagnostics
epoch by epoch.

Collects:
  - PyTorch Profiler op-level breakdown (CPU/CUDA time, memory, op categories)
  - Gradient health per layer (norms, vanishing/exploding detection, NaN/Inf)
  - Activation statistics per layer (mean, std, dead neuron %, saturation)
  - Weight distribution evolution over epochs
  - Memory footprint (process RSS, CUDA allocated/reserved/peak)
  - Throughput (samples/sec, tokens/sec, batches/sec)
  - Loss curves with moving averages
  - Attention entropy and sparsity (transformer-specific)
  - Console and error log capture
  - System resource usage (CPU %, RAM)
  - Forward vs backward pass timing ratio
  - Layer-by-layer parameter and compute cost mapping
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
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ObserverConfig:
    """Controls which telemetry channels the Observer records.

    Toggle individual channels on/off depending on what you need to debug.
    Disabling expensive channels (profiler, activations, attention_entropy)
    reduces overhead for production-like runs.
    """

    # Core tracking
    track_profiler: bool = True
    track_gradients: bool = True
    track_activations: bool = True
    track_memory: bool = True
    track_throughput: bool = True
    track_loss: bool = True
    track_weights: bool = True

    # Transformer-specific
    track_attention_entropy: bool = False  # hooks into attention layers

    # Logging
    track_console_logs: bool = True
    track_error_logs: bool = True
    track_hyperparameters: bool = True

    # System
    track_system_resources: bool = True

    # Profiler tuning
    profiler_record_shapes: bool = True
    profiler_profile_memory: bool = True
    profiler_with_stack: bool = False
    profiler_top_n_ops: int = 20

    # Gradient health thresholds
    gradient_exploding_threshold: float = 10.0
    gradient_vanishing_threshold: float = 1e-7

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
    >>> config = ObserverConfig(track_profiler=True, track_gradients=True)
    >>> obs = Observer(api_key="key-123", project_id="proj-abc", config=config)
    >>> obs.log_hyperparameters({"lr": 3e-4, "batch_size": 64, ...})
    >>> obs.register_model(model)
    >>>
    >>> for epoch in range(num_epochs):
    ...     obs.start_epoch(epoch)
    ...     for step, (x, y) in enumerate(loader):
    ...         # On first batch, run a profiled step
    ...         if step == 0 and config.track_profiler:
    ...             logits, loss = obs.profile_step(model, x, y)
    ...         else:
    ...             logits, loss = model(x, y)
    ...         loss.backward()
    ...         optimizer.step(); optimizer.zero_grad()
    ...         obs.log_batch(step, loss, batch_size=x.size(0), seq_length=x.size(1))
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
        self._activation_stats: Dict[str, Dict] = {}
        self._attention_entropy_stats: Dict[str, Dict] = {}

        # ── Hooks ──
        self._activation_hooks: list = []
        self._attention_hooks: list = []
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

        This captures the full architecture map (layer types, param counts)
        and attaches forward hooks for activation / attention monitoring.
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

        # ── Attach hooks ──
        if self.config.track_activations:
            self._attach_activation_hooks(model)
        if self.config.track_attention_entropy:
            self._attach_attention_hooks(model)

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
    # Activation hooks
    # ------------------------------------------------------------------

    def _attach_activation_hooks(self, model: nn.Module):
        self._detach_hooks(self._activation_hooks)
        targets = (nn.Linear, nn.LayerNorm, nn.Embedding, nn.Dropout)
        for name, module in model.named_modules():
            if isinstance(module, targets):
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self._activation_hooks.append(hook)

    def _make_activation_hook(self, layer_name: str):
        def hook(_module, _input, output):
            if not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                flat = output.float()
                self._activation_stats[layer_name] = {
                    "mean": round(flat.mean().item(), 6),
                    "std": round(flat.std().item(), 6),
                    "min": round(flat.min().item(), 6),
                    "max": round(flat.max().item(), 6),
                    "abs_mean": round(flat.abs().mean().item(), 6),
                    "dead_fraction": round((flat == 0).float().mean().item(), 6),
                    "saturated_fraction": round(
                        ((flat.abs() > 0.99 * flat.abs().max()).float().mean().item()), 6
                    ),
                    "shape": list(output.shape),
                }
        return hook

    # ------------------------------------------------------------------
    # Attention entropy hooks (transformer-specific)
    # ------------------------------------------------------------------

    def _attach_attention_hooks(self, model: nn.Module):
        """Hook into attention heads to capture weight entropy & sparsity."""
        self._detach_hooks(self._attention_hooks)
        for name, module in model.named_modules():
            # Heuristic: look for modules named *sa*, *attention*, or with a
            # `tril` buffer (causal mask), indicating an attention head.
            if hasattr(module, "tril") or "attention" in name.lower() or "sa" in name.lower():
                if hasattr(module, "forward"):
                    hook = module.register_forward_hook(
                        self._make_attention_hook(name)
                    )
                    self._attention_hooks.append(hook)

    def _make_attention_hook(self, layer_name: str):
        """
        Capture attention weight entropy & sparsity.

        We monkey-patch the softmax output inside the hook by wrapping
        the module's forward. Instead, we approximate using the output:
        since a single Head returns wei @ v, we can't directly get wei.
        So this hook captures the *output* distribution as a proxy.
        For true attention weights, enable the profiler trace.
        """
        def hook(_module, _input, output):
            if not isinstance(output, torch.Tensor) or output.dim() < 2:
                return
            with torch.no_grad():
                # Treat output as attention-weighted representation
                # Compute distributional stats as proxy for attention pattern
                flat = output.float()
                # Approximate entropy of the output distribution along seq dim
                # (higher entropy = more uniform attention, lower = more peaked)
                abs_vals = flat.abs()
                probs = abs_vals / (abs_vals.sum(dim=-1, keepdim=True) + 1e-10)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
                max_entropy = math.log(flat.shape[-1])

                self._attention_entropy_stats[layer_name] = {
                    "output_entropy": round(entropy, 6),
                    "max_possible_entropy": round(max_entropy, 6),
                    "entropy_ratio": round(entropy / max(max_entropy, 1e-10), 4),
                    "output_sparsity": round(
                        (flat.abs() < 1e-4).float().mean().item(), 6
                    ),
                }
        return hook

    @staticmethod
    def _detach_hooks(hook_list: list):
        for h in hook_list:
            h.remove()
        hook_list.clear()

    # ==================================================================
    # Epoch lifecycle
    # ==================================================================

    def start_epoch(self, epoch: int):
        """Call at the beginning of each epoch."""
        self._current_epoch = epoch
        self._epoch_start_time = time.time()
        self._epoch_batch_losses.clear()
        self._epoch_batch_times.clear()
        self._epoch_tokens_processed = 0
        self._epoch_samples_processed = 0
        self._activation_stats.clear()
        self._attention_entropy_stats.clear()
        self._profiler_snapshot = None

        if torch.cuda.is_available() and self.config.track_memory:
            torch.cuda.reset_peak_memory_stats()

        self._log.info(f"--- Epoch {epoch} started ---")

    def log_batch(
        self,
        batch_idx: int,
        loss,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ):
        """Record metrics for a single training step within an epoch."""
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

        # ── Gradients ──
        if self.config.track_gradients and self._model is not None:
            rec["gradients"] = self._snapshot_gradients()

        # ── Weights ──
        if self.config.track_weights and self._model is not None:
            rec["weights"] = self._snapshot_weights()

        # ── Activations ──
        if self.config.track_activations and self._activation_stats:
            rec["activations"] = dict(self._activation_stats)

        # ── Attention entropy ──
        if self.config.track_attention_entropy and self._attention_entropy_stats:
            rec["attention_entropy"] = dict(self._attention_entropy_stats)

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

        Call this *instead of* a normal training step (typically on the
        first batch of each epoch). It returns (logits, loss) just like
        model(x, y) so it can be a drop-in replacement.

        The profiler results are stored and will be included in the
        next end_epoch() call.
        """
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=self.config.profiler_record_shapes,
            profile_memory=self.config.profiler_profile_memory,
            with_stack=self.config.profiler_with_stack,
        ) as prof:
            with record_function("model_forward"):
                logits, loss = model(x, y)
            with record_function("model_backward"):
                if loss is not None:
                    loss.backward()

        self._profiler_snapshot = self._parse_profiler(prof)
        return logits, loss

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

        return {
            "total_cpu_time_ms": round(total_cpu / 1000, 3),
            "total_cuda_time_ms": round(total_cuda / 1000, 3),
            "forward_time_ms": round(fwd_time / 1000, 3),
            "backward_time_ms": round(bwd_time / 1000, 3),
            "fwd_bwd_ratio": round(fwd_time / max(bwd_time, 1), 4),
            "num_unique_ops": len(avgs),
            "top_operations": top_ops,
            "operation_categories": categories,
        }

    # ==================================================================
    # Snapshot helpers
    # ==================================================================

    def _snapshot_gradients(self) -> Dict[str, Any]:
        per_layer: Dict[str, Dict] = {}
        total_norm_sq = 0.0
        issues: List[Dict] = []

        for name, param in self._model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.data.float()
            norm = g.norm(2).item()
            total_norm_sq += norm ** 2

            per_layer[name] = {
                "norm": round(norm, 6),
                "mean": round(g.mean().item(), 8),
                "std": round(g.std().item(), 8),
                "abs_mean": round(g.abs().mean().item(), 8),
                "min": round(g.min().item(), 8),
                "max": round(g.max().item(), 8),
            }

            if norm > self.config.gradient_exploding_threshold:
                issues.append({"layer": name, "issue": "exploding", "norm": norm})
            elif norm < self.config.gradient_vanishing_threshold:
                issues.append({"layer": name, "issue": "vanishing", "norm": norm})
            if torch.isnan(g).any():
                issues.append({"layer": name, "issue": "NaN"})
            if torch.isinf(g).any():
                issues.append({"layer": name, "issue": "Inf"})

        return {
            "total_norm": round(math.sqrt(total_norm_sq), 6),
            "num_layers": len(per_layer),
            "issues": issues,
            "health": "healthy" if not issues else "warning",
            "per_layer": per_layer,
        }

    def _snapshot_weights(self) -> Dict[str, Dict]:
        stats: Dict[str, Dict] = {}
        for name, param in self._model.named_parameters():
            w = param.data.float()
            stats[name] = {
                "mean": round(w.mean().item(), 8),
                "std": round(w.std().item(), 8),
                "norm": round(w.norm(2).item(), 6),
                "min": round(w.min().item(), 8),
                "max": round(w.max().item(), 8),
                "numel": param.numel(),
            }
        return stats

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

        # Gradient health across epochs
        grad_issue_counts = [
            len(e.get("gradients", {}).get("issues", []))
            for e in self.epoch_data
        ]
        summary["gradient_health"] = {
            "epochs_with_issues": sum(1 for c in grad_issue_counts if c > 0),
            "total_issues": sum(grad_issue_counts),
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
                summary["profiler_highlight"] = {
                    "fwd_bwd_ratio": p.get("fwd_bwd_ratio"),
                    "top_op": p["top_operations"][0]["name"] if p["top_operations"] else None,
                    "top_op_pct": round(
                        100
                        * p["top_operations"][0]["cpu_time_us"]
                        / max(p["total_cpu_time_ms"] * 1000, 1),
                        2,
                    )
                    if p["top_operations"]
                    else None,
                }
                break

        return summary

    # ==================================================================
    # Cleanup
    # ==================================================================

    def close(self):
        """Detach all hooks and finalize session metadata."""
        self._detach_hooks(self._activation_hooks)
        self._detach_hooks(self._attention_hooks)
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
