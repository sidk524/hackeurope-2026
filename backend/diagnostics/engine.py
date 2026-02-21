"""
ML Diagnostics Engine — "Sentry for ML Pipelines"

Analyses TrainStep payloads and SessionLogs for a session and returns
a list of IssueData objects plus a health_score and detected arch_type.

Architecture plug-in pattern:
  - Generic epoch/profiler checks run for every architecture.
  - Per-architecture checkers (CNN, RNN, Transformer) are registered in
    CHECKER_REGISTRY and auto-selected by `detect_arch_type()`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from models import IssueSeverity, IssueCategory


# ─────────────────────────────────────────────────────────────────────────────
# Data class carrying one flagged issue (not yet persisted)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IssueData:
    severity: IssueSeverity
    category: IssueCategory
    title: str
    description: str
    epoch_index: int | None = None
    layer_id: str | None = None
    metric_key: str | None = None
    metric_value: Any = None
    suggestion: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_WEIGHT = {
    IssueSeverity.critical: 20,
    IssueSeverity.warning: 7,
    IssueSeverity.info: 2,
}


def compute_health_score(issues: list[IssueData]) -> int:
    score = 100
    for issue in issues:
        score -= _SEVERITY_WEIGHT.get(issue.severity, 0)
    return max(0, score)


def detect_arch_type(layer_graph: dict | None) -> str:
    """Infer architecture family from layer categories in the graph."""
    if not layer_graph:
        return "generic"
    nodes = layer_graph.get("nodes", [])
    categories = {n.get("category", "") for n in nodes}
    if "attention" in categories:
        return "transformer"
    if "recurrent" in categories:
        return "rnn"
    if "convolution" in categories:
        return "cnn"
    return "generic"


def _layer_id_from_per_layer(entry: dict) -> str:
    """Robustly extract a layer name from a per_layer profiler entry."""
    return (
        entry.get("name")
        or entry.get("layer")
        or entry.get("layer_name")
        or entry.get("id")
        or "unknown"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generic epoch-level checks
# ─────────────────────────────────────────────────────────────────────────────

def check_loss_divergence(epochs: list[dict]) -> list[IssueData]:
    """Detect 2+ consecutive epochs where train_mean is rising after epoch 0."""
    issues: list[IssueData] = []
    if len(epochs) < 3:
        return issues
    for i in range(2, len(epochs)):
        prev_prev = epochs[i - 2]["loss"]["train_mean"]
        prev = epochs[i - 1]["loss"]["train_mean"]
        curr = epochs[i]["loss"]["train_mean"]
        if curr > prev > prev_prev:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.loss,
                title="Loss divergence detected",
                description=(
                    f"Training loss increased for 2 consecutive epochs "
                    f"(epochs {i - 1}→{i}: {prev:.4f}→{curr:.4f})"
                ),
                epoch_index=i,
                metric_key="train_mean",
                metric_value={"epoch": i, "prev": prev, "curr": curr},
                suggestion=(
                    "Check your learning rate — try a scheduler (e.g. CosineAnnealingLR). "
                    "Verify input data for NaN values or corrupt batches."
                ),
            ))
            break  # report first occurrence only
    return issues


def check_loss_explosion(epochs: list[dict]) -> list[IssueData]:
    """Detect a >100% jump in train_mean between consecutive epochs."""
    issues: list[IssueData] = []
    for i in range(1, len(epochs)):
        prev = epochs[i - 1]["loss"]["train_mean"]
        curr = epochs[i]["loss"]["train_mean"]
        if prev > 0 and curr / prev > 2.0:
            issues.append(IssueData(
                severity=IssueSeverity.critical,
                category=IssueCategory.loss,
                title="Loss explosion",
                description=(
                    f"Training loss more than doubled in epoch {i} "
                    f"({prev:.4f} → {curr:.4f}, ×{curr / prev:.1f})"
                ),
                epoch_index=i,
                metric_key="train_mean",
                metric_value={"prev": prev, "curr": curr, "ratio": round(curr / prev, 2)},
                suggestion=(
                    "Drastically reduce the learning rate. "
                    "Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0). "
                    "Scan for NaN values in inputs."
                ),
            ))
    return issues


def check_loss_plateau(epochs: list[dict]) -> list[IssueData]:
    """Detect < 1% improvement over the last 3 epochs (only after epoch 3)."""
    if len(epochs) < 4:
        return []
    last_3 = epochs[-3:]
    losses = [e["loss"]["train_mean"] for e in last_3]
    first_loss = losses[0]
    if first_loss == 0:
        return []
    improvement = (first_loss - losses[-1]) / first_loss
    if improvement < 0.01:
        return [IssueData(
            severity=IssueSeverity.warning,
            category=IssueCategory.loss,
            title="Loss plateau",
            description=(
                f"Training loss improved only {improvement * 100:.2f}% over the last 3 epochs "
                f"({losses[0]:.4f} → {losses[-1]:.4f})"
            ),
            epoch_index=len(epochs) - 1,
            metric_key="train_mean",
            metric_value={"last_3_losses": losses, "improvement_pct": round(improvement * 100, 2)},
            suggestion=(
                "Use a learning-rate scheduler (e.g. ReduceLROnPlateau). "
                "Consider early stopping or increasing model capacity."
            ),
        )]
    return []


def check_overfitting(epochs: list[dict]) -> list[IssueData]:
    """
    Detect overfitting: val_loss increases while train_mean decreases
    for 2+ consecutive epoch pairs.
    """
    if len(epochs) < 3:
        return []
    overfit_epochs: list[int] = []
    for i in range(1, len(epochs)):
        e_prev = epochs[i - 1]
        e_curr = epochs[i]
        val_prev = (e_prev["loss"].get("val") or {}).get("val_loss")
        val_curr = (e_curr["loss"].get("val") or {}).get("val_loss")
        train_prev = e_prev["loss"]["train_mean"]
        train_curr = e_curr["loss"]["train_mean"]
        if val_prev is not None and val_curr is not None:
            if val_curr > val_prev and train_curr < train_prev:
                overfit_epochs.append(i)
    if len(overfit_epochs) >= 2:
        return [IssueData(
            severity=IssueSeverity.warning,
            category=IssueCategory.loss,
            title="Overfitting detected",
            description=(
                f"Validation loss rising while training loss falls "
                f"at epochs {overfit_epochs}"
            ),
            epoch_index=overfit_epochs[-1],
            metric_key="val_loss",
            metric_value={"overfit_at_epochs": overfit_epochs},
            suggestion=(
                "Add regularisation: dropout (nn.Dropout), weight decay (AdamW). "
                "Try data augmentation or reduce model capacity."
            ),
        )]
    return []


def check_high_loss_variance(epochs: list[dict]) -> list[IssueData]:
    """Flag epochs where train_std / train_mean > 0.5 (noisy gradients)."""
    issues: list[IssueData] = []
    for e in epochs:
        mean = e["loss"]["train_mean"]
        std = e["loss"].get("train_std", 0)
        if mean > 0 and std / mean > 0.5:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.loss,
                title="High intra-epoch loss variance",
                description=(
                    f"Epoch {e['epoch']}: loss std/mean = {std / mean:.2f} "
                    f"(std={std:.4f}, mean={mean:.4f}) — noisy gradient signal"
                ),
                epoch_index=e["epoch"],
                metric_key="train_std",
                metric_value={
                    "train_mean": mean,
                    "train_std": std,
                    "ratio": round(std / mean, 3),
                },
                suggestion=(
                    "Increase batch size to reduce gradient noise. "
                    "Lower the learning rate or add learning rate warmup."
                ),
            ))
    return issues


def check_gradient_instability(epochs: list[dict]) -> list[IssueData]:
    """Flag epochs where max batch loss / mean > 20 (gradient spike proxy)."""
    issues: list[IssueData] = []
    for e in epochs:
        mean = e["loss"]["train_mean"]
        max_loss = e["loss"].get("train_max", 0)
        if mean > 0 and max_loss / mean > 20:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.loss,
                title="Gradient instability signal",
                description=(
                    f"Epoch {e['epoch']}: worst batch loss is "
                    f"{max_loss / mean:.0f}× the mean ({max_loss:.4f} vs {mean:.4f})"
                ),
                epoch_index=e["epoch"],
                metric_key="train_max",
                metric_value={
                    "train_max": max_loss,
                    "train_mean": mean,
                    "ratio": round(max_loss / mean, 1),
                },
                suggestion=(
                    "Add gradient clipping: "
                    "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)"
                ),
            ))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Throughput / system checks
# ─────────────────────────────────────────────────────────────────────────────

def check_throughput_degradation(epochs: list[dict]) -> list[IssueData]:
    """Flag epochs where throughput drops >20% below the peak epoch."""
    if not epochs:
        return []
    speeds = [
        (i, e.get("throughput", {}).get("samples_per_second", 0))
        for i, e in enumerate(epochs)
    ]
    peak = max(s for _, s in speeds) if speeds else 0
    if peak == 0:
        return []
    issues: list[IssueData] = []
    for idx, s in speeds:
        if s > 0 and s < peak * 0.8:
            e = epochs[idx]
            drop_pct = (1 - s / peak) * 100
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.throughput,
                title="Throughput degradation",
                description=(
                    f"Epoch {e['epoch']}: {s:.1f} samples/s "
                    f"({drop_pct:.1f}% below peak of {peak:.1f} samples/s)"
                ),
                epoch_index=e["epoch"],
                metric_key="samples_per_second",
                metric_value={
                    "current": s,
                    "peak": peak,
                    "drop_pct": round(drop_pct, 1),
                },
                suggestion=(
                    "Check data-loader worker count (num_workers), I/O bottlenecks, "
                    "or OS-level CPU throttling during this epoch."
                ),
            ))
    return issues


def check_memory_growth(epochs: list[dict]) -> list[IssueData]:
    """Flag >25% RSS memory growth between epoch 0 and the last epoch."""
    if len(epochs) < 2:
        return []
    first_rss = epochs[0].get("memory", {}).get("process_rss_mb", 0) or 0
    last_rss = epochs[-1].get("memory", {}).get("process_rss_mb", 0) or 0
    if first_rss <= 0:
        return []
    growth = (last_rss - first_rss) / first_rss
    if growth > 0.25:
        return [IssueData(
            severity=IssueSeverity.warning,
            category=IssueCategory.memory,
            title="Persistent memory growth",
            description=(
                f"RSS memory grew {growth * 100:.1f}% over training "
                f"({first_rss:.1f} MB → {last_rss:.1f} MB)"
            ),
            metric_key="process_rss_mb",
            metric_value={
                "first_epoch_mb": first_rss,
                "last_epoch_mb": last_rss,
                "growth_pct": round(growth * 100, 1),
            },
            suggestion=(
                "Detach losses before accumulating: use loss.item() not loss. "
                "Check for Python-side tensor references held across epochs."
            ),
        )]
    return []


def check_slow_epoch(epochs: list[dict]) -> list[IssueData]:
    """Flag any epoch that took >1.5× the median epoch duration."""
    if len(epochs) < 3:
        return []
    durations = [e.get("duration_seconds", 0) for e in epochs]
    median = statistics.median(durations)
    issues: list[IssueData] = []
    for e in epochs:
        d = e.get("duration_seconds", 0)
        if median > 0 and d > median * 1.5:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.system,
                title="Abnormally slow epoch",
                description=(
                    f"Epoch {e['epoch']} took {d:.1f}s "
                    f"({d / median:.1f}× the median of {median:.1f}s)"
                ),
                epoch_index=e["epoch"],
                metric_key="duration_seconds",
                metric_value={
                    "duration": d,
                    "median": median,
                    "ratio": round(d / median, 2),
                },
                suggestion=(
                    "Check for CPU throttling, data-loader stalls, "
                    "or OS background processes during this epoch."
                ),
            ))
    return issues


def check_high_cpu(epochs: list[dict]) -> list[IssueData]:
    """Flag epochs where system CPU utilisation exceeds 90%."""
    issues: list[IssueData] = []
    for e in epochs:
        cpu = (e.get("system") or {}).get("cpu_percent", 0) or 0
        if cpu > 90:
            issues.append(IssueData(
                severity=IssueSeverity.info,
                category=IssueCategory.system,
                title="High CPU utilisation",
                description=(
                    f"Epoch {e['epoch']}: CPU at {cpu:.1f}% — "
                    "training is compute-bottlenecked on CPU"
                ),
                epoch_index=e["epoch"],
                metric_key="cpu_percent",
                metric_value={"cpu_percent": cpu},
                suggestion=(
                    "Move training to GPU. "
                    "Enable pin_memory=True in DataLoader and increase num_workers."
                ),
            ))
    return issues


def check_error_logs(logs: list[Any]) -> list[IssueData]:
    """Flag sessions that have error-level log entries."""
    error_logs = [lg for lg in logs if getattr(lg, "kind", None) == "error"]
    if not error_logs:
        return []
    sample = error_logs[0]
    return [IssueData(
        severity=IssueSeverity.critical,
        category=IssueCategory.logs,
        title=f"{len(error_logs)} error log(s) detected",
        description=f"First error at {sample.ts}: {sample.msg}",
        metric_key="error_count",
        metric_value={"count": len(error_logs), "first_error": sample.msg},
        suggestion=(
            "Inspect the error_logs array in the session report and fix "
            "before treating any metrics as reliable."
        ),
    )]


# ─────────────────────────────────────────────────────────────────────────────
# Profiler layer-level checks (generic)
# ─────────────────────────────────────────────────────────────────────────────

def check_profiler_hotspot(epochs: list[dict]) -> list[IssueData]:
    """
    Two checks:
    1. A single layer >40% of total CPU profiler time in any epoch.
    2. The same layer consistently leads (>25%) across 3+ epochs.
    """
    issues: list[IssueData] = []
    # map layer_id → list of pct_total values across epochs
    layer_epoch_pcts: dict[str, list[float]] = {}
    hotspot_ids: set[str] = set()

    for e in epochs:
        per_layer = (e.get("profiler") or {}).get("per_layer", [])
        for entry in per_layer:
            lid = _layer_id_from_per_layer(entry)
            pct = entry.get("pct_total", 0) or 0
            layer_epoch_pcts.setdefault(lid, []).append(pct)
            if pct > 40:
                hotspot_ids.add(lid)
                issues.append(IssueData(
                    severity=IssueSeverity.warning,
                    category=IssueCategory.profiler,
                    title=f"Compute hotspot: {lid}",
                    description=(
                        f"Epoch {e['epoch']}: layer '{lid}' consumed "
                        f"{pct:.1f}% of total CPU profiler time"
                    ),
                    epoch_index=e["epoch"],
                    layer_id=lid,
                    metric_key="pct_total",
                    metric_value={"pct_total": pct},
                    suggestion=(
                        f"Layer '{lid}' is a compute bottleneck. "
                        "Consider reducing its size, using more efficient ops, "
                        "or restructuring surrounding layers."
                    ),
                ))

    # Consistently hot across epochs
    for lid, pcts in layer_epoch_pcts.items():
        high_count = sum(1 for p in pcts if p > 25)
        if high_count >= 3 and lid not in hotspot_ids:
            avg_pct = sum(pcts) / len(pcts)
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.profiler,
                title=f"Consistently hot layer: {lid}",
                description=(
                    f"Layer '{lid}' averaged {avg_pct:.1f}% of compute "
                    f"across {high_count} epochs"
                ),
                layer_id=lid,
                metric_key="avg_pct_total",
                metric_value={
                    "avg_pct": round(avg_pct, 2),
                    "high_epoch_count": high_count,
                },
                suggestion=(
                    f"'{lid}' is a persistent bottleneck. "
                    "Profile this layer individually and consider architectural changes."
                ),
            ))
    return issues


def check_backward_dominance(epochs: list[dict]) -> list[IssueData]:
    """Flag epochs where backward pass uses >45% of total CPU time."""
    issues: list[IssueData] = []
    for e in epochs:
        cats = (e.get("profiler") or {}).get("operation_categories", {})
        bwd_pct = (cats.get("backward_pass") or {}).get("pct_cpu", 0) or 0
        if bwd_pct > 45:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.profiler,
                title="Backward pass dominates compute",
                description=(
                    f"Epoch {e['epoch']}: backward pass uses "
                    f"{bwd_pct:.1f}% of CPU time"
                ),
                epoch_index=e["epoch"],
                metric_key="backward_pass_pct",
                metric_value={"pct_cpu": bwd_pct},
                suggestion=(
                    "Consider gradient checkpointing "
                    "(torch.utils.checkpoint) to reduce backward complexity."
                ),
            ))
    return issues


def check_fwd_bwd_ratio(epochs: list[dict]) -> list[IssueData]:
    """Flag epochs where forward/backward ratio < 0.15 (unusually heavy backward)."""
    issues: list[IssueData] = []
    for e in epochs:
        ratio = (e.get("profiler") or {}).get("fwd_bwd_ratio")
        if ratio is not None and ratio < 0.15:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.profiler,
                title="Low forward/backward ratio",
                description=(
                    f"Epoch {e['epoch']}: fwd/bwd ratio = {ratio:.4f} — "
                    "backward pass is disproportionately heavy"
                ),
                epoch_index=e["epoch"],
                metric_key="fwd_bwd_ratio",
                metric_value={"ratio": ratio},
                suggestion=(
                    "Possible vanishing gradient or over-regularisation. "
                    "Review activations, batch norm, and weight initialisation."
                ),
            ))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# GreenAI / sustainability checks (profiler-level)
# ─────────────────────────────────────────────────────────────────────────────

def check_diminishing_returns(epochs: list[dict]) -> list[IssueData]:
    """Flag epochs where marginal loss improvement < 5% of cumulative improvement."""
    issues: list[IssueData] = []
    if len(epochs) < 3:
        return issues

    # Try precomputed sustainability data first
    for i, e in enumerate(epochs):
        sus = (e.get("sustainability") or {}).get("marginal_loss") or {}
        moc = sus.get("marginal_over_cumulative")
        if moc is not None and moc < 0.05 and i >= 2:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.sustainability,
                title="Diminishing returns on training",
                description=(
                    f"Epoch {e.get('epoch', i)}: marginal loss improvement is only "
                    f"{moc * 100:.1f}% of cumulative improvement "
                    f"({sus.get('absolute_improvement', 0):.4f} vs "
                    f"{sus.get('cumulative_improvement', 0):.4f})"
                ),
                epoch_index=e.get("epoch", i),
                metric_key="marginal_over_cumulative",
                metric_value={
                    "marginal_over_cumulative": moc,
                    "absolute_improvement": sus.get("absolute_improvement"),
                    "cumulative_improvement": sus.get("cumulative_improvement"),
                },
                suggestion=(
                    "Consider early stopping at this epoch. The compute spent on "
                    "further training yields minimal loss reduction. Use a callback "
                    "like EarlyStopping(patience=3) or ReduceLROnPlateau."
                ),
            ))
            break

    # Fallback: raw loss computation
    if not issues:
        losses = [e.get("loss", {}).get("train_mean") for e in epochs]
        losses = [l for l in losses if l is not None]
        if len(losses) >= 3:
            first_loss = losses[0]
            for i in range(2, len(losses)):
                cumulative = first_loss - losses[i]
                marginal = losses[i - 1] - losses[i]
                if cumulative > 0 and marginal / cumulative < 0.05:
                    issues.append(IssueData(
                        severity=IssueSeverity.warning,
                        category=IssueCategory.sustainability,
                        title="Diminishing returns on training",
                        description=(
                            f"Epoch {epochs[i].get('epoch', i)}: marginal loss improvement "
                            f"({marginal:.4f}) is only {100 * marginal / cumulative:.1f}% of "
                            f"cumulative improvement ({cumulative:.4f})"
                        ),
                        epoch_index=epochs[i].get("epoch", i),
                        metric_key="marginal_over_cumulative",
                        metric_value={
                            "marginal": round(marginal, 6),
                            "cumulative": round(cumulative, 6),
                            "ratio": round(marginal / cumulative, 4),
                        },
                        suggestion=(
                            "Consider early stopping at this epoch. The compute spent on "
                            "further training yields minimal loss reduction."
                        ),
                    ))
                    break
    return issues


def check_over_parameterized_layer(epochs: list[dict], arch: dict | None) -> list[IssueData]:
    """Flag layers whose parameter share greatly exceeds their compute share (>10x)."""
    issues: list[IssueData] = []
    if not arch:
        return issues
    arch_layers = arch.get("layers") or {}
    if not arch_layers:
        return issues

    layer_compute: dict[str, list[float]] = {}
    for e in epochs:
        for entry in (e.get("profiler") or {}).get("per_layer", []):
            lid = _layer_id_from_per_layer(entry)
            pct = entry.get("pct_total", 0) or 0
            layer_compute.setdefault(lid, []).append(pct)

    for layer_name, arch_info in arch_layers.items():
        pct_params = arch_info.get("pct_of_total", 0)
        compute_vals = layer_compute.get(layer_name, [])
        if not compute_vals or pct_params < 1:
            continue
        avg_compute = sum(compute_vals) / len(compute_vals)
        if avg_compute < 0.1:
            continue
        ratio = pct_params / avg_compute
        if ratio > 10:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.sustainability,
                title=f"Over-parameterized layer: {layer_name}",
                description=(
                    f"Layer '{layer_name}' holds {pct_params:.1f}% of parameters but "
                    f"only contributes {avg_compute:.1f}% of compute "
                    f"(param-to-compute ratio: {ratio:.1f}x). "
                    f"These parameters may be underutilized."
                ),
                layer_id=layer_name,
                metric_key="param_to_compute_ratio",
                metric_value={
                    "pct_parameters": pct_params,
                    "avg_pct_compute": round(avg_compute, 2),
                    "ratio": round(ratio, 2),
                },
                suggestion=(
                    f"Layer '{layer_name}' has disproportionate parameters vs compute. "
                    "Consider reducing its size (fewer neurons/filters) or adding pooling "
                    "before it to reduce input dimensionality."
                ),
            ))
    return issues


def check_compute_inefficient_layer(epochs: list[dict], arch: dict | None) -> list[IssueData]:
    """Flag layers consuming disproportionate compute relative to parameter count."""
    issues: list[IssueData] = []
    if not arch:
        return issues
    arch_layers = arch.get("layers") or {}
    if not arch_layers:
        return issues

    layer_compute: dict[str, list[float]] = {}
    for e in epochs:
        for entry in (e.get("profiler") or {}).get("per_layer", []):
            lid = _layer_id_from_per_layer(entry)
            pct = entry.get("pct_total", 0) or 0
            layer_compute.setdefault(lid, []).append(pct)

    for layer_name, arch_info in arch_layers.items():
        pct_params = arch_info.get("pct_of_total", 0)
        compute_vals = layer_compute.get(layer_name, [])
        if not compute_vals:
            continue
        avg_compute = sum(compute_vals) / len(compute_vals)

        # Near-zero param layers with significant compute
        if pct_params < 0.1 and avg_compute > 15:
            issues.append(IssueData(
                severity=IssueSeverity.info,
                category=IssueCategory.sustainability,
                title=f"Compute-heavy lightweight layer: {layer_name}",
                description=(
                    f"Layer '{layer_name}' has only {pct_params:.2f}% of parameters "
                    f"but consumes {avg_compute:.1f}% of compute."
                ),
                layer_id=layer_name,
                metric_key="compute_to_param_ratio",
                metric_value={
                    "pct_parameters": pct_params,
                    "avg_pct_compute": round(avg_compute, 2),
                },
                suggestion=(
                    f"Review '{layer_name}' for optimization. Consider stride > 1 or "
                    "depthwise separable convolutions to reduce compute."
                ),
            ))
            continue

        if pct_params < 0.01:
            continue
        ratio = avg_compute / pct_params
        if ratio > 10 and avg_compute > 5:
            issues.append(IssueData(
                severity=IssueSeverity.info,
                category=IssueCategory.sustainability,
                title=f"Compute-inefficient layer: {layer_name}",
                description=(
                    f"Layer '{layer_name}' uses {avg_compute:.1f}% of compute but has "
                    f"only {pct_params:.2f}% of parameters "
                    f"(compute-to-param ratio: {ratio:.1f}x)"
                ),
                layer_id=layer_name,
                metric_key="compute_to_param_ratio",
                metric_value={
                    "pct_parameters": pct_params,
                    "avg_pct_compute": round(avg_compute, 2),
                    "ratio": round(ratio, 2),
                },
                suggestion=(
                    f"Layer '{layer_name}' consumes more compute than its parameter "
                    "share suggests. Consider more efficient alternatives "
                    "(e.g. depthwise separable convolutions, kernel decomposition)."
                ),
            ))
    return issues


def check_device_underutilization(epochs: list[dict]) -> list[IssueData]:
    """Detect when GPU/accelerator is not being used effectively."""
    issues: list[IssueData] = []
    if not epochs:
        return issues

    total_cuda = 0.0
    total_cpu = 0.0
    profiled_count = 0
    for e in epochs:
        prof = e.get("profiler") or {}
        if prof:
            profiled_count += 1
            total_cuda += prof.get("total_cuda_time_ms", 0) or 0
            total_cpu += prof.get("total_cpu_time_ms", 0) or 0

    if profiled_count > 0 and total_cuda == 0 and total_cpu > 0:
        issues.append(IssueData(
            severity=IssueSeverity.info,
            category=IssueCategory.sustainability,
            title="Training running entirely on CPU",
            description=(
                f"All {profiled_count} profiled epochs show 0ms CUDA time and "
                f"{total_cpu:.1f}ms total CPU time. GPU-accelerated training is "
                f"typically 10-100x faster and more energy-efficient per sample."
            ),
            metric_key="total_cuda_time_ms",
            metric_value={
                "total_cuda_ms": total_cuda,
                "total_cpu_ms": round(total_cpu, 2),
                "profiled_epochs": profiled_count,
            },
            suggestion=(
                "Move training to GPU with model.to('cuda') and data .to('cuda'). "
                "GPU training is significantly more energy-efficient per sample."
            ),
        ))

    if profiled_count > 0 and total_cuda > 0 and total_cpu > 0:
        gpu_ratio = total_cuda / total_cpu
        if gpu_ratio < 0.1:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.sustainability,
                title="Low GPU utilization",
                description=(
                    f"CUDA time is only {gpu_ratio * 100:.1f}% of CPU time across "
                    f"{profiled_count} profiled epochs. The GPU may be underutilized."
                ),
                metric_key="gpu_cpu_ratio",
                metric_value={
                    "total_cuda_ms": round(total_cuda, 2),
                    "total_cpu_ms": round(total_cpu, 2),
                    "gpu_cpu_ratio": round(gpu_ratio, 4),
                },
                suggestion=(
                    "Increase batch size to better utilize GPU parallelism. "
                    "Use pin_memory=True and increase num_workers in DataLoader."
                ),
            ))
    return issues


def check_early_stop_opportunity(epochs: list[dict]) -> list[IssueData]:
    """Suggest early stopping when loss improvement stalls and compute is being wasted."""
    issues: list[IssueData] = []
    if len(epochs) < 3:
        return issues

    losses = [e.get("loss", {}).get("train_mean") for e in epochs]
    losses = [l for l in losses if l is not None]
    if len(losses) < 3:
        return issues

    first_loss = losses[0]
    optimal_stop = len(losses) - 1

    for i in range(2, len(losses)):
        cumulative = first_loss - losses[i]
        marginal = losses[i - 1] - losses[i]
        if cumulative > 0 and marginal / cumulative < 0.05:
            optimal_stop = i - 1
            break

    wasted_epochs = len(losses) - 1 - optimal_stop
    if wasted_epochs >= 1:
        total_duration = sum(e.get("duration_seconds", 0) for e in epochs)
        wasted_duration = sum(
            e.get("duration_seconds", 0) for e in epochs[optimal_stop + 1:]
        )
        wasted_pct = round(100 * wasted_duration / max(total_duration, 1e-9), 2)
        wasted_samples = sum(
            (e.get("throughput") or {}).get("samples_processed", 0)
            for e in epochs[optimal_stop + 1:]
        )
        issues.append(IssueData(
            severity=IssueSeverity.warning,
            category=IssueCategory.sustainability,
            title=f"Early stopping opportunity at epoch {optimal_stop}",
            description=(
                f"Training could have stopped after epoch {optimal_stop} "
                f"(loss: {losses[optimal_stop]:.4f}). The remaining {wasted_epochs} "
                f"epoch(s) wasted {wasted_pct:.1f}% of total compute "
                f"({wasted_duration:.1f}s, {wasted_samples:,} samples)."
            ),
            epoch_index=optimal_stop,
            metric_key="wasted_compute_pct",
            metric_value={
                "optimal_stop_epoch": optimal_stop,
                "wasted_epochs": wasted_epochs,
                "wasted_compute_pct": wasted_pct,
                "wasted_duration_s": round(wasted_duration, 2),
                "wasted_samples": wasted_samples,
                "loss_at_stop": round(losses[optimal_stop], 6),
                "final_loss": round(losses[-1], 6),
            },
            suggestion=(
                f"Implement early stopping with patience=2-3 to save ~{wasted_pct:.0f}% "
                f"of compute. This would save {wasted_duration:.1f}s per run."
            ),
        ))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# GreenAI / layer health checks (tensor-level)
# ─────────────────────────────────────────────────────────────────────────────

def _get_layer_health(epoch: dict) -> dict:
    """Extract layer_health.layers dict from an epoch, or empty dict."""
    return (epoch.get("layer_health") or {}).get("layers", {})


def check_dead_neurons(epochs: list[dict]) -> list[IssueData]:
    """Flag layers with near-zero weights or fully dead (frozen output + sparse weights)."""
    issues: list[IssueData] = []
    reported: set[str] = set()

    for i, e in enumerate(epochs):
        layers = _get_layer_health(e)
        for name, data in layers.items():
            if name in reported:
                continue
            if data.get("is_dead"):
                issues.append(IssueData(
                    severity=IssueSeverity.critical,
                    category=IssueCategory.sustainability,
                    title=f"Dead layer: {name}",
                    description=(
                        f"Layer '{name}' at epoch {e.get('epoch', '?')}: output never changes "
                        f"and >90% of weights are near-zero. This layer is effectively dead."
                    ),
                    epoch_index=e.get("epoch", i),
                    layer_id=name,
                    metric_key="is_dead",
                    metric_value={
                        "weight_sparsity": data.get("weight_sparsity"),
                        "activation_var_of_means": data.get("activation_var_of_means"),
                    },
                    suggestion=(
                        f"Layer '{name}' is not contributing to learning. "
                        "Prune it, reduce its width, or reinitialize its weights."
                    ),
                ))
                reported.add(name)
            elif data.get("has_near_zero_weights") and data.get("weight_sparsity", 0) > 0.5:
                issues.append(IssueData(
                    severity=IssueSeverity.warning,
                    category=IssueCategory.sustainability,
                    title=f"Near-zero weights: {name}",
                    description=(
                        f"Layer '{name}' at epoch {e.get('epoch', '?')}: "
                        f"{data.get('weight_sparsity', 0) * 100:.1f}% of weights are near-zero."
                    ),
                    epoch_index=e.get("epoch", i),
                    layer_id=name,
                    metric_key="weight_sparsity",
                    metric_value={"weight_sparsity": data.get("weight_sparsity")},
                    suggestion=(
                        f"Layer '{name}' has mostly dead weights. "
                        "Consider pruning or reducing layer width."
                    ),
                ))
                reported.add(name)
    return issues


def check_vanishing_gradients(epochs: list[dict]) -> list[IssueData]:
    """Flag layers with consistently vanishing gradient flow."""
    issues: list[IssueData] = []
    # Track per-layer: how many epochs show vanishing gradients
    vanish_counts: dict[str, int] = {}
    vanish_values: dict[str, list[float]] = {}

    for e in epochs:
        layers = _get_layer_health(e)
        for name, data in layers.items():
            g_norm = data.get("gradient_norm_mean")
            if g_norm is not None:
                vanish_values.setdefault(name, []).append(g_norm)
                if data.get("has_vanishing_gradients"):
                    vanish_counts[name] = vanish_counts.get(name, 0) + 1

    for name, count in vanish_counts.items():
        if count >= 2:
            avg_norm = sum(vanish_values.get(name, [0])) / max(len(vanish_values.get(name, [1])), 1)
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.sustainability,
                title=f"Vanishing gradients: {name}",
                description=(
                    f"Layer '{name}' has near-zero gradient flow in {count} epoch(s). "
                    f"Average gradient norm: {avg_norm:.2e}."
                ),
                layer_id=name,
                metric_key="gradient_norm_mean",
                metric_value={
                    "vanishing_epoch_count": count,
                    "avg_gradient_norm": avg_norm,
                },
                suggestion=(
                    f"Layer '{name}' receives almost no gradient signal. "
                    "Use skip connections, better initialization (Kaiming/Xavier), "
                    "or switch activations (ReLU -> GELU)."
                ),
            ))

    # Check gradient ratio between first and last parameter layers
    if len(epochs) > 0:
        last_epoch = epochs[-1]
        layers = _get_layer_health(last_epoch)
        layer_names = list(layers.keys())
        if len(layer_names) >= 2:
            first_norm = (layers[layer_names[0]].get("gradient_norm_mean") or 0)
            last_norm = (layers[layer_names[-1]].get("gradient_norm_mean") or 0)
            if last_norm > 0 and first_norm > 0:
                ratio = last_norm / first_norm
                if ratio > 100:
                    issues.append(IssueData(
                        severity=IssueSeverity.warning,
                        category=IssueCategory.sustainability,
                        title="Gradient vanishing through network depth",
                        description=(
                            f"Gradient norm ratio between last ({layer_names[-1]}) and "
                            f"first ({layer_names[0]}) layer is {ratio:.0f}x. "
                            f"Gradients are diminishing as they flow backward."
                        ),
                        metric_key="gradient_depth_ratio",
                        metric_value={
                            "first_layer": layer_names[0],
                            "first_norm": round(first_norm, 8),
                            "last_layer": layer_names[-1],
                            "last_norm": round(last_norm, 8),
                            "ratio": round(ratio, 2),
                        },
                        suggestion=(
                            "Add skip/residual connections to improve gradient flow. "
                            "Consider using LayerNorm or gradient clipping."
                        ),
                    ))
    return issues


def check_frozen_output(epochs: list[dict]) -> list[IssueData]:
    """Flag layers whose output never changes across batches."""
    issues: list[IssueData] = []
    frozen_counts: dict[str, int] = {}

    for e in epochs:
        layers = _get_layer_health(e)
        for name, data in layers.items():
            if data.get("has_frozen_output"):
                frozen_counts[name] = frozen_counts.get(name, 0) + 1

    for name, count in frozen_counts.items():
        if count >= 2:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.sustainability,
                title=f"Frozen output: {name}",
                description=(
                    f"Layer '{name}' produces identical output regardless of input "
                    f"batch in {count} epoch(s). This layer is not learning."
                ),
                layer_id=name,
                metric_key="activation_var_of_means",
                metric_value={"frozen_epoch_count": count},
                suggestion=(
                    f"Layer '{name}' output never changes. Check if it is accidentally "
                    "frozen (requires_grad=False) or receiving zero gradients."
                ),
            ))
    return issues


def check_activation_collapse(epochs: list[dict]) -> list[IssueData]:
    """Flag layers with near-zero activation variance (representation collapse)."""
    issues: list[IssueData] = []
    collapse_counts: dict[str, int] = {}

    for e in epochs:
        layers = _get_layer_health(e)
        for name, data in layers.items():
            if data.get("has_low_activation_variance"):
                collapse_counts[name] = collapse_counts.get(name, 0) + 1

    for name, count in collapse_counts.items():
        if count >= 2:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.sustainability,
                title=f"Activation collapse: {name}",
                description=(
                    f"Layer '{name}' has near-zero activation variance in {count} "
                    f"epoch(s). All inputs produce nearly the same activation."
                ),
                layer_id=name,
                metric_key="activation_std",
                metric_value={"collapse_epoch_count": count},
                suggestion=(
                    f"Add BatchNorm or LayerNorm before '{name}'. "
                    "Check for saturating activations (sigmoid/tanh with large inputs)."
                ),
            ))
    return issues


def check_redundant_layers(epochs: list[dict]) -> list[IssueData]:
    """Flag consecutive layer pairs with highly correlated activations."""
    issues: list[IssueData] = []
    # Track pairs seen as high-correlation across epochs
    pair_counts: dict[tuple[str, str], list[float]] = {}

    for e in epochs:
        corrs = (e.get("layer_health") or {}).get("activation_correlations", [])
        for entry in corrs:
            pair = (entry["layer_a"], entry["layer_b"])
            corr = entry.get("correlation", 0)
            if abs(corr) > 0.95:
                pair_counts.setdefault(pair, []).append(corr)

    for (a, b), corr_vals in pair_counts.items():
        if len(corr_vals) >= 2:
            avg_corr = sum(corr_vals) / len(corr_vals)
            issues.append(IssueData(
                severity=IssueSeverity.info,
                category=IssueCategory.sustainability,
                title=f"Redundant layers: {a} <-> {b}",
                description=(
                    f"Layers '{a}' and '{b}' produce nearly identical outputs "
                    f"(avg correlation: {avg_corr:.3f}) across {len(corr_vals)} epochs."
                ),
                layer_id=a,
                metric_key="activation_correlation",
                metric_value={
                    "layer_a": a,
                    "layer_b": b,
                    "avg_correlation": round(avg_corr, 4),
                    "epoch_count": len(corr_vals),
                },
                suggestion=(
                    f"Layers '{a}' and '{b}' may be redundant. Consider removing one "
                    "or adding non-linearity between them."
                ),
            ))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# GreenAI / carbon footprint checks (CodeCarbon data)
# ─────────────────────────────────────────────────────────────────────────────

def check_high_carbon_intensity(epochs: list[dict]) -> list[IssueData]:
    """Report training carbon footprint and flag high-intensity epochs."""
    issues: list[IssueData] = []
    carbon_data: list[tuple[int, dict]] = [
        (i, e["carbon_emissions"])
        for i, e in enumerate(epochs)
        if e.get("carbon_emissions") is not None
    ]
    if not carbon_data:
        return issues

    total_co2 = sum(c["epoch_co2_kg"] for _, c in carbon_data)
    total_energy = sum(c["epoch_energy_kwh"] for _, c in carbon_data)
    total_samples = sum(
        epochs[i].get("throughput", {}).get("samples_processed", 0)
        for i, _ in carbon_data
    )
    avg_power = (
        sum(c["power_draw_watts"] for _, c in carbon_data) / len(carbon_data)
    )
    # Aggregate per-component power & energy from EmissionsData fields
    total_cpu_energy = sum(c.get("cpu_energy_kwh", 0.0) or 0.0 for _, c in carbon_data)
    total_gpu_energy = sum(c.get("gpu_energy_kwh", 0.0) or 0.0 for _, c in carbon_data)
    total_ram_energy = sum(c.get("ram_energy_kwh", 0.0) or 0.0 for _, c in carbon_data)
    total_water = sum(c.get("water_consumed_l", 0.0) or 0.0 for _, c in carbon_data)
    avg_cpu_power = sum(c.get("cpu_power_w", 0.0) or 0.0 for _, c in carbon_data) / len(carbon_data)
    avg_gpu_power = sum(c.get("gpu_power_w", 0.0) or 0.0 for _, c in carbon_data) / len(carbon_data)
    avg_ram_power = sum(c.get("ram_power_w", 0.0) or 0.0 for _, c in carbon_data) / len(carbon_data)
    # Use country/region/hardware from the last epoch that has data
    last_carbon = carbon_data[-1][1]
    country_name = last_carbon.get("country_name")
    region = last_carbon.get("region")
    cpu_model = last_carbon.get("cpu_model")
    gpu_model = last_carbon.get("gpu_model")

    # Info-level: report total carbon footprint
    if total_co2 > 0:
        power_parts = [f"CPU {avg_cpu_power:.1f}W", f"RAM {avg_ram_power:.1f}W"]
        if avg_gpu_power > 0:
            power_parts.insert(1, f"GPU {avg_gpu_power:.1f}W")
        location_str = ""
        if country_name:
            location_str = f" ({country_name}{', ' + region if region else ''})"
        issues.append(IssueData(
            severity=IssueSeverity.info,
            category=IssueCategory.sustainability,
            title=f"Training carbon footprint: {total_co2 * 1000:.2f}g CO2",
            description=(
                f"Total emissions: {total_co2 * 1000:.2f}g CO2 over "
                f"{len(carbon_data)} epoch(s){location_str}. "
                f"Energy consumed: {total_energy * 1000:.2f} Wh "
                f"(CPU {total_cpu_energy * 1000:.2f} / "
                f"GPU {total_gpu_energy * 1000:.2f} / "
                f"RAM {total_ram_energy * 1000:.2f} Wh). "
                f"Avg power: {avg_power:.1f}W ({', '.join(power_parts)})."
                + (f" Water: {total_water * 1000:.1f} ml." if total_water > 0 else "")
            ),
            metric_key="total_co2_kg",
            metric_value={
                "total_co2_kg": round(total_co2, 10),
                "total_energy_kwh": round(total_energy, 10),
                "cpu_energy_kwh": round(total_cpu_energy, 10),
                "gpu_energy_kwh": round(total_gpu_energy, 10),
                "ram_energy_kwh": round(total_ram_energy, 10),
                "avg_power_watts": round(avg_power, 2),
                "avg_cpu_power_w": round(avg_cpu_power, 2),
                "avg_gpu_power_w": round(avg_gpu_power, 2),
                "avg_ram_power_w": round(avg_ram_power, 2),
                "total_water_l": round(total_water, 6),
                "total_samples": total_samples,
                "co2_per_1k_samples_kg": round(
                    total_co2 / max(total_samples / 1000, 1e-9), 12
                ) if total_samples else None,
                "country_name": country_name,
                "region": region,
                "cpu_model": cpu_model,
                "gpu_model": gpu_model,
            },
            suggestion=(
                "Consider early stopping, model pruning, or training during "
                "low-carbon grid hours to reduce emissions."
            ),
        ))

    # Warning-level: flag any epoch whose carbon intensity (CO2 per % loss
    # improvement) is 10x+ the session average
    losses = [e.get("loss", {}).get("train_mean") for e in epochs]
    if len(losses) >= 2 and losses[0] is not None:
        epoch_intensities: list[tuple[int, float]] = []
        for idx, carbon in carbon_data:
            if idx == 0 or losses[idx] is None or losses[idx - 1] is None:
                continue
            improvement_pct = (losses[idx - 1] - losses[idx]) / max(abs(losses[0]), 1e-9) * 100
            if improvement_pct > 0:
                intensity = carbon["epoch_co2_kg"] / improvement_pct
                epoch_intensities.append((idx, intensity))

        if len(epoch_intensities) >= 2:
            avg_intensity = (
                sum(v for _, v in epoch_intensities) / len(epoch_intensities)
            )
            for idx, intensity in epoch_intensities:
                if intensity > avg_intensity * 10:
                    issues.append(IssueData(
                        severity=IssueSeverity.warning,
                        category=IssueCategory.sustainability,
                        title=f"High carbon intensity at epoch {idx}",
                        description=(
                            f"Epoch {idx} used {intensity / avg_intensity:.1f}x "
                            f"the average CO2 per % loss improvement. "
                            f"Consider stopping training earlier."
                        ),
                        epoch_index=idx,
                        metric_key="carbon_intensity_ratio",
                        metric_value={
                            "ratio_vs_avg": round(intensity / avg_intensity, 2),
                            "epoch_co2_kg": carbon_data[idx][1]["epoch_co2_kg"]
                            if idx < len(carbon_data) else None,
                            "cpu_power_w": carbon_data[idx][1].get("cpu_power_w"),
                            "gpu_power_w": carbon_data[idx][1].get("gpu_power_w"),
                        },
                        suggestion=(
                            "This epoch's carbon cost far exceeds its training "
                            "benefit. Implement early stopping or learning rate "
                            "scheduling to avoid wasteful epochs."
                        ),
                    ))

    return issues


def check_wasted_carbon(epochs: list[dict]) -> list[IssueData]:
    """Quantify CO2 wasted on epochs past the optimal stopping point."""
    issues: list[IssueData] = []
    if len(epochs) < 3:
        return issues

    # Check carbon data exists
    has_carbon = any(e.get("carbon_emissions") for e in epochs)
    if not has_carbon:
        return issues

    # Reuse optimal stop detection
    losses = [e.get("loss", {}).get("train_mean") for e in epochs]
    losses = [l for l in losses if l is not None]
    if len(losses) < 3:
        return issues

    first_loss = losses[0]
    optimal_stop = len(losses) - 1

    for i in range(2, len(losses)):
        cumulative = first_loss - losses[i]
        marginal = losses[i - 1] - losses[i]
        if cumulative > 0 and marginal / cumulative < 0.05:
            optimal_stop = i - 1
            break

    wasted_epochs = len(losses) - 1 - optimal_stop
    if wasted_epochs < 1:
        return issues

    wasted_co2 = sum(
        e.get("carbon_emissions", {}).get("epoch_co2_kg", 0)
        for e in epochs[optimal_stop + 1:]
    )
    wasted_energy = sum(
        e.get("carbon_emissions", {}).get("epoch_energy_kwh", 0)
        for e in epochs[optimal_stop + 1:]
    )

    wasted_water = sum(
        e.get("carbon_emissions", {}).get("water_consumed_l", 0)
        for e in epochs[optimal_stop + 1:]
    )

    if wasted_co2 > 0:
        water_str = f" and {wasted_water * 1000:.1f} ml of water" if wasted_water > 0 else ""
        issues.append(IssueData(
            severity=IssueSeverity.warning,
            category=IssueCategory.sustainability,
            title=f"Wasted {wasted_co2 * 1000:.2f}g CO2 on unnecessary epochs",
            description=(
                f"Training could have stopped after epoch {optimal_stop}. "
                f"The remaining {wasted_epochs} epoch(s) wasted "
                f"{wasted_co2 * 1000:.2f}g CO2, {wasted_energy * 1000:.2f} Wh"
                f"{water_str} with diminishing returns."
            ),
            epoch_index=optimal_stop,
            metric_key="wasted_co2_kg",
            metric_value={
                "wasted_co2_kg": round(wasted_co2, 10),
                "wasted_energy_kwh": round(wasted_energy, 10),
                "wasted_water_l": round(wasted_water, 6),
                "wasted_epochs": wasted_epochs,
                "optimal_stop_epoch": optimal_stop,
            },
            suggestion=(
                f"Implement early stopping with patience=2-3 to save "
                f"~{wasted_co2 * 1000:.2f}g CO2 per training run."
            ),
        ))

    return issues


def check_gpu_power_efficiency(epochs: list[dict]) -> list[IssueData]:
    """
    Flag epochs where GPU is drawing significant power but utilization is low
    (wasted idle GPU power), and CPU-dominant training on a GPU-equipped machine.
    """
    issues: list[IssueData] = []
    carbon_epochs: list[tuple[int, dict]] = [
        (i, e["carbon_emissions"])
        for i, e in enumerate(epochs)
        if e.get("carbon_emissions") is not None
    ]
    if not carbon_epochs:
        return issues

    # Check for idle GPU draw: gpu_power > 10W but gpu_utilization < 20%
    idle_gpu_epochs: list[int] = []
    for idx, c in carbon_epochs:
        gpu_power = c.get("gpu_power_w", 0.0) or 0.0
        gpu_util = c.get("gpu_utilization_pct", 0.0) or 0.0
        if gpu_power > 10.0 and 0 < gpu_util < 20.0:
            idle_gpu_epochs.append(idx)

    if idle_gpu_epochs:
        sample = carbon_epochs[idle_gpu_epochs[0]][1]
        issues.append(IssueData(
            severity=IssueSeverity.warning,
            category=IssueCategory.sustainability,
            title=f"Low GPU utilization with active power draw ({len(idle_gpu_epochs)} epoch(s))",
            description=(
                f"GPU is drawing power ({sample.get('gpu_power_w', 0):.1f}W) but utilization "
                f"is below 20% in {len(idle_gpu_epochs)} epoch(s). "
                f"This wastes energy without compute benefit."
            ),
            epoch_index=idle_gpu_epochs[0],
            metric_key="gpu_utilization_pct",
            metric_value={
                "affected_epochs": idle_gpu_epochs,
                "gpu_power_w": sample.get("gpu_power_w"),
                "gpu_utilization_pct": sample.get("gpu_utilization_pct"),
                "gpu_model": sample.get("gpu_model"),
            },
            suggestion=(
                "Increase batch size to saturate the GPU, use DataLoader with "
                "pin_memory=True and num_workers>0, or profile data pipeline bottlenecks "
                "that are leaving the GPU starved."
            ),
        ))

    # Check for CPU-dominant training: cpu_energy far exceeds gpu_energy when gpu present
    high_cpu_epochs: list[int] = []
    for idx, c in carbon_epochs:
        cpu_e = c.get("cpu_energy_kwh", 0.0) or 0.0
        gpu_e = c.get("gpu_energy_kwh", 0.0) or 0.0
        gpu_power = c.get("gpu_power_w", 0.0) or 0.0
        # GPU present (drawing power) but CPU consumed >5x the energy of GPU
        if gpu_power > 1.0 and gpu_e > 0 and cpu_e / gpu_e > 5.0:
            high_cpu_epochs.append(idx)

    if high_cpu_epochs:
        sample = carbon_epochs[high_cpu_epochs[0]][1]
        cpu_e = sample.get("cpu_energy_kwh", 0) or 0
        gpu_e = sample.get("gpu_energy_kwh", 0) or 0
        ratio = cpu_e / max(gpu_e, 1e-12)
        issues.append(IssueData(
            severity=IssueSeverity.info,
            category=IssueCategory.sustainability,
            title=f"CPU consuming {ratio:.1f}x more energy than GPU ({len(high_cpu_epochs)} epoch(s))",
            description=(
                f"CPU energy ({cpu_e * 1000:.2f} Wh) far exceeds GPU energy "
                f"({gpu_e * 1000:.2f} Wh) in epoch {high_cpu_epochs[0]}. "
                f"Data preprocessing or CPU-side ops may be the bottleneck."
            ),
            epoch_index=high_cpu_epochs[0],
            metric_key="cpu_to_gpu_energy_ratio",
            metric_value={
                "affected_epochs": high_cpu_epochs,
                "cpu_energy_kwh": cpu_e,
                "gpu_energy_kwh": gpu_e,
                "ratio": round(ratio, 2),
                "cpu_model": sample.get("cpu_model"),
                "gpu_model": sample.get("gpu_model"),
            },
            suggestion=(
                "Move data preprocessing to GPU transforms, increase DataLoader "
                "num_workers, or use mixed-precision (torch.autocast) to shift "
                "compute from CPU to GPU."
            ),
        ))

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Architecture checkers (plug-in pattern)
# ─────────────────────────────────────────────────────────────────────────────

class CnnChecker:
    """CNN-specific architecture checks. Operates on the static layer graph."""

    def run(self, arch: dict) -> list[IssueData]:
        issues: list[IssueData] = []
        layer_graph: dict = arch.get("layer_graph") or {}
        layers: dict = arch.get("layers") or {}
        total_params: int = arch.get("total_parameters", 0)
        nodes: list[dict] = layer_graph.get("nodes", [])
        seq_path: list[str] = layer_graph.get("sequential_path", [])

        issues += self._check_fc_dominates(layers, total_params)
        issues += self._check_conv_bottleneck(layers, total_params)
        issues += self._check_missing_pooling(nodes, seq_path)
        issues += self._check_large_kernel(nodes)
        issues += self._check_early_channel_explosion(nodes)
        return issues

    # ── individual sub-checks ──────────────────────────────────────────────

    def _check_fc_dominates(self, layers: dict, total_params: int) -> list[IssueData]:
        """A Linear layer holding >92% of total params in a CNN pre-dates inadequate pooling."""
        issues: list[IssueData] = []
        if total_params == 0:
            return issues
        for name, info in layers.items():
            if info.get("type") == "Linear":
                pct = info.get("pct_of_total", 0)
                if pct > 92:
                    issues.append(IssueData(
                        severity=IssueSeverity.warning,
                        category=IssueCategory.architecture,
                        title=f"FC layer '{name}' dominates parameters ({pct:.1f}%)",
                        description=(
                            f"Linear layer '{name}' holds {pct:.1f}% of all parameters "
                            f"({info.get('parameters', 0):,} / {total_params:,}). "
                            "In a CNN this almost always means the conv output is too "
                            "large before the FC layer."
                        ),
                        layer_id=name,
                        metric_key="pct_of_total",
                        metric_value={
                            "pct": pct,
                            "params": info.get("parameters", 0),
                            "total_params": total_params,
                        },
                        suggestion=(
                            "Add nn.AdaptiveAvgPool2d(1) before the FC to collapse "
                            "spatial dimensions, or reduce the number of conv output channels."
                        ),
                    ))
        return issues

    def _check_conv_bottleneck(self, layers: dict, total_params: int) -> list[IssueData]:
        """A single Conv2d layer holding >60% of total params is disproportionate."""
        issues: list[IssueData] = []
        if total_params == 0:
            return issues
        for name, info in layers.items():
            if info.get("type") == "Conv2d":
                pct = info.get("pct_of_total", 0)
                if pct > 60:
                    issues.append(IssueData(
                        severity=IssueSeverity.warning,
                        category=IssueCategory.architecture,
                        title=f"Conv layer '{name}' is a parameter bottleneck ({pct:.1f}%)",
                        description=(
                            f"Conv2d '{name}' holds {pct:.1f}% of total parameters — "
                            "disproportionate for a convolutional layer."
                        ),
                        layer_id=name,
                        metric_key="pct_of_total",
                        metric_value={"pct": pct, "params": info.get("parameters", 0)},
                        suggestion=(
                            "Split into two smaller conv layers or reduce filter count. "
                            "Use bottleneck blocks (1×1 → 3×3 → 1×1) for efficiency."
                        ),
                    ))
        return issues

    def _check_missing_pooling(self, nodes: list[dict], seq_path: list[str]) -> list[IssueData]:
        """
        Flag when 3+ consecutive Conv2d layers appear with no pooling between them.
        2 consecutive convs (e.g. VGG-style blocks) are idiomatic — only warn at 3+.
        """
        issues: list[IssueData] = []
        node_map = {n["id"]: n for n in nodes}
        POOL_TYPES = {"MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d"}

        conv_streak: list[str] = []
        for layer_id in seq_path:
            node = node_map.get(layer_id, {})
            cat = node.get("category", "")
            ntype = node.get("type", "")

            if cat == "convolution":
                conv_streak.append(layer_id)
            elif ntype in POOL_TYPES:
                conv_streak = []
            else:
                if len(conv_streak) >= 3:
                    issues.append(IssueData(
                        severity=IssueSeverity.info,
                        category=IssueCategory.architecture,
                        title="3+ consecutive conv layers without pooling",
                        description=(
                            f"Layers {conv_streak} are consecutive Conv2d layers "
                            "with no pooling between them. Spatial dimensions stay "
                            "large, inflating the FC input size."
                        ),
                        layer_id=conv_streak[-1],
                        suggestion=(
                            "Add nn.MaxPool2d or nn.AdaptiveAvgPool2d after "
                            "convolutional blocks to reduce spatial size before FC layers."
                        ),
                    ))
                conv_streak = []
        # check at end of path
        if len(conv_streak) >= 3:
            issues.append(IssueData(
                severity=IssueSeverity.info,
                category=IssueCategory.architecture,
                title="3+ consecutive conv layers without pooling",
                description=(
                    f"Layers {conv_streak} end the network without intermediate pooling."
                ),
                layer_id=conv_streak[-1],
                suggestion=(
                    "Add nn.AdaptiveAvgPool2d before FC layers to collapse "
                    "spatial dimensions."
                ),
            ))
        return issues

    def _check_large_kernel(self, nodes: list[dict]) -> list[IssueData]:
        """Warn on conv kernels ≥7×7 — typically only useful for large inputs."""
        issues: list[IssueData] = []
        for node in nodes:
            if node.get("category") != "convolution":
                continue
            kernel = node.get("kernel_size") or []
            if isinstance(kernel, list) and kernel:
                k = max(kernel)
                if k >= 7:
                    issues.append(IssueData(
                        severity=IssueSeverity.info,
                        category=IssueCategory.architecture,
                        title=f"Large kernel in '{node['id']}' ({k}×{k})",
                        description=(
                            f"Conv layer '{node['id']}' uses a {k}×{k} kernel. "
                            "On small spatial inputs (≤64×64) this can be excessive "
                            "and may hurt generalisation."
                        ),
                        layer_id=node["id"],
                        metric_key="kernel_size",
                        metric_value={"kernel_size": kernel},
                        suggestion=(
                            "Prefer 3×3 kernels for small inputs. "
                            "Large kernels (≥7×7) are most beneficial for high-res "
                            "inputs (224×224+) as in ImageNet-style pipelines."
                        ),
                    ))
        return issues

    def _check_early_channel_explosion(self, nodes: list[dict]) -> list[IssueData]:
        """
        Warn if the first conv layer jumps from 1 (grayscale) to >32 channels —
        likely overparameterised for small inputs like MNIST.
        """
        issues: list[IssueData] = []
        first_conv = next(
            (n for n in nodes if n.get("category") == "convolution"), None
        )
        if first_conv:
            in_ch = first_conv.get("in_channels", 0) or 0
            out_ch = first_conv.get("out_channels", 0) or 0
            if in_ch == 1 and out_ch > 32:
                issues.append(IssueData(
                    severity=IssueSeverity.info,
                    category=IssueCategory.architecture,
                    title=(
                        f"Early channel explosion in '{first_conv['id']}' "
                        f"(1→{out_ch})"
                    ),
                    description=(
                        f"First conv layer expands from 1 (grayscale) to {out_ch} "
                        f"channels in '{first_conv['id']}'. For small inputs like MNIST "
                        "this may over-parameterise early feature extraction."
                    ),
                    layer_id=first_conv["id"],
                    metric_key="out_channels",
                    metric_value={"in_channels": in_ch, "out_channels": out_ch},
                    suggestion=(
                        "Start with a smaller expansion (e.g. 1→8 or 1→16) for "
                        "grayscale small images. Scale up if accuracy is insufficient."
                    ),
                ))
        return issues


# Stub checkers for future work
class _NotImplementedChecker:
    def run(self, arch: dict) -> list[IssueData]:  # noqa: ARG002
        return []


CHECKER_REGISTRY: dict[str, Any] = {
    "cnn": CnnChecker(),
    "rnn": _NotImplementedChecker(),
    "transformer": _NotImplementedChecker(),
    "generic": _NotImplementedChecker(),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(
    epochs: list[dict],
    logs: list[Any],
    arch: dict | None,
) -> tuple[list[IssueData], int, str]:
    """
    Run the full suite of ML heuristic checks.

    Parameters
    ----------
    epochs : list of EpochRecord-shaped dicts (from TrainStep.payload)
    logs   : list of SessionLog ORM objects
    arch   : ModelArchitecture-shaped dict (from Model.architecture) or None

    Returns
    -------
    (issues, health_score, arch_type)
    """
    issues: list[IssueData] = []

    # ── Generic epoch checks ──────────────────────────────────────────────
    issues += check_loss_divergence(epochs)
    issues += check_loss_explosion(epochs)
    issues += check_loss_plateau(epochs)
    issues += check_overfitting(epochs)
    issues += check_high_loss_variance(epochs)
    issues += check_gradient_instability(epochs)
    issues += check_throughput_degradation(epochs)
    issues += check_memory_growth(epochs)
    issues += check_slow_epoch(epochs)
    issues += check_high_cpu(epochs)
    issues += check_error_logs(logs)

    # ── Profiler checks ───────────────────────────────────────────────────
    issues += check_profiler_hotspot(epochs)
    issues += check_backward_dominance(epochs)
    issues += check_fwd_bwd_ratio(epochs)

    # ── GreenAI / Sustainability checks (profiler-level) ─────────────────
    issues += check_diminishing_returns(epochs)
    issues += check_over_parameterized_layer(epochs, arch)
    issues += check_compute_inefficient_layer(epochs, arch)
    issues += check_device_underutilization(epochs)
    issues += check_early_stop_opportunity(epochs)

    # ── GreenAI / Layer health checks (tensor-level) ─────────────────────
    issues += check_dead_neurons(epochs)
    issues += check_vanishing_gradients(epochs)
    issues += check_frozen_output(epochs)
    issues += check_activation_collapse(epochs)
    issues += check_redundant_layers(epochs)

    # -- GreenAI / Carbon footprint checks (CodeCarbon) --
    issues += check_high_carbon_intensity(epochs)
    issues += check_wasted_carbon(epochs)
    issues += check_gpu_power_efficiency(epochs)

    # ── Architecture-specific checks ──────────────────────────────────────
    arch_type = "generic"
    if arch:
        layer_graph = arch.get("layer_graph")
        arch_type = detect_arch_type(layer_graph)
        checker = CHECKER_REGISTRY.get(arch_type)
        if checker:
            issues += checker.run(arch)

    health_score = compute_health_score(issues)
    return issues, health_score, arch_type
