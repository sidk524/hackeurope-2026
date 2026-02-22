from datetime import datetime
from typing import Any
from pydantic import BaseModel

from models import IssueSeverity, IssueCategory


class IssueOut(BaseModel):
    id: int | None = None
    severity: IssueSeverity
    category: IssueCategory
    title: str
    description: str
    epoch_index: int | None = None
    layer_id: str | None = None
    metric_key: str | None = None
    metric_value: Any | None = None
    suggestion: str = ""

    class Config:
        from_attributes = True


class LayerHighlight(BaseModel):
    """All issues pointing to a specific layer, scored for ranking."""
    layer_id: str
    layer_type: str | None = None
    severity_score: int          # critical=20, warning=7, info=2
    issues: list[IssueOut]


class EpochDiagnosticSummary(BaseModel):
    epoch_index: int
    issues: list[IssueOut]


class DiagnosticRunSummary(BaseModel):
    """Lightweight view returned by the list endpoint."""
    id: int
    session_id: int
    created_at: datetime
    health_score: int
    issue_count: int
    arch_type: str
    summary_json: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class SustainabilityInsight(BaseModel):
    """GreenAI sustainability metrics for a diagnostic run."""
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
    redundant_layer_pairs: list[dict[str, Any]] = []
    sustainability_issue_count: int = 0

    # Carbon footprint (CodeCarbon)
    total_co2_kg: float | None = None
    total_energy_kwh: float | None = None
    co2_per_epoch_avg_kg: float | None = None
    co2_per_1k_samples_kg: float | None = None
    avg_power_draw_watts: float | None = None
    wasted_co2_kg: float | None = None


class DiagnosticRunOut(BaseModel):
    """Full detail for a single diagnostic run."""
    id: int
    session_id: int
    created_at: datetime
    health_score: int
    issue_count: int
    arch_type: str
    summary_json: dict[str, Any] | None = None

    issues: list[IssueOut]
    epoch_trends: list[EpochDiagnosticSummary]   # per-epoch grouping
    session_level_issues: list[IssueOut]          # issues with no epoch_index
    layer_highlights: list[LayerHighlight]        # sorted by severity_score desc
    sustainability: SustainabilityInsight | None = None

    class Config:
        from_attributes = True


class HealthOut(BaseModel):
    session_id: int
    health_score: int
    severity_counts: dict[str, int]   # {"critical": n, "warning": n, "info": n}
    issues: list[IssueOut]
    layers: list[LayerHighlight]


class SessionTrendItem(BaseModel):
    session_id: int
    run_id: str
    run_name: str
    started_at: str
    health_score: int | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    val_acc: float | None = None
    issue_count: int | None = None
    diagnostic_run_count: int = 0  # number of diagnostic runs for this session


class ProjectTrendOut(BaseModel):
    project_id: int
    sessions: list[SessionTrendItem]
    improving: bool | None = None   # None if < 2 sessions with diagnostics


class FixPromptOut(BaseModel):
    """Generated system prompt for solving a diagnostic issue."""
    issue_id: int
    prompt: str
    cached: bool = False
