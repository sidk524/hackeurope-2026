"""
Diagnostics Router — /diagnostics

Provides ML Observability API endpoints for analysing training sessions,
persisting diagnostic runs, and surfacing layer-level insights.
"""

from collections import defaultdict
from fastapi import APIRouter, HTTPException
from sqlalchemy import func
from sqlmodel import select

from database import SessionDep
from models import (
    DiagnosticIssue,
    DiagnosticRun,
    IssueCategory,
    IssueSeverity,
    Model,
    Project,
    SessionLog,
    TrainSession,
    TrainStep,
)
from diagnostics.engine import IssueData, run_diagnostics
from event_bus import EventType, SSEEvent, publish_from_sync
from diagnostics.schemas import (
    DiagnosticRunOut,
    DiagnosticRunSummary,
    EpochDiagnosticSummary,
    FixPromptOut,
    HealthOut,
    IssueOut,
    LayerHighlight,
    ProjectTrendOut,
    SessionTrendItem,
    SustainabilityInsight,
)

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])

# ─────────────────────────────────────────────────────────────────────────────
# Fix-prompt in-memory cache  (issue_id → generated prompt string)
# ─────────────────────────────────────────────────────────────────────────────
_fix_prompt_cache: dict[int, str] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_WEIGHT = {
    IssueSeverity.critical: 20,
    IssueSeverity.warning: 7,
    IssueSeverity.info: 2,
}


def _step_to_epoch(step: TrainStep) -> dict:
    """Build an epoch dict from a TrainStep's individual columns."""
    epoch: dict = {
        "epoch": step.step_index,
        "duration_seconds": step.duration_seconds,
    }
    if step.loss:
        epoch["loss"] = step.loss
    if step.throughput:
        epoch["throughput"] = step.throughput
    if step.profiler:
        epoch["profiler"] = step.profiler
    if step.memory:
        epoch["memory"] = step.memory
    if step.system:
        epoch["system"] = step.system
    if step.layer_health:
        epoch["layer_health"] = step.layer_health
    if step.sustainability:
        epoch["sustainability"] = step.sustainability
    if step.carbon_emissions:
        epoch["carbon_emissions"] = step.carbon_emissions
    if step.log_counts:
        epoch["log_counts"] = step.log_counts
    return epoch


def _build_layer_highlights(
    issues: list[IssueOut], arch: dict | None
) -> list[LayerHighlight]:
    """Group issues by layer_id and compute severity scores."""
    layer_issues: dict[str, list[IssueOut]] = defaultdict(list)
    for issue in issues:
        if issue.layer_id:
            layer_issues[issue.layer_id].append(issue)

    # Build layer_type lookup from arch
    layer_type_map: dict[str, str] = {}
    if arch:
        nodes = (arch.get("layer_graph") or {}).get("nodes", [])
        for node in nodes:
            lid = node.get("id")
            ltype = node.get("type")
            if lid and ltype:
                layer_type_map[lid] = ltype

    highlights: list[LayerHighlight] = []
    for layer_id, layer_issues_list in layer_issues.items():
        severity_score = sum(
            _SEVERITY_WEIGHT.get(i.severity, 0) for i in layer_issues_list
        )
        highlights.append(LayerHighlight(
            layer_id=layer_id,
            layer_type=layer_type_map.get(layer_id),
            severity_score=severity_score,
            issues=layer_issues_list,
        ))

    # Sort descending by severity score
    highlights.sort(key=lambda h: h.severity_score, reverse=True)
    return highlights


def _build_epoch_trends(issues: list[IssueOut]) -> list[EpochDiagnosticSummary]:
    """Group issues by epoch_index (excluding None)."""
    epoch_issues: dict[int, list[IssueOut]] = defaultdict(list)
    for issue in issues:
        if issue.epoch_index is not None:
            epoch_issues[issue.epoch_index].append(issue)

    return [
        EpochDiagnosticSummary(epoch_index=epoch_idx, issues=epoch_list)
        for epoch_idx, epoch_list in sorted(epoch_issues.items())
    ]


def _build_session_level_issues(issues: list[IssueOut]) -> list[IssueOut]:
    """Return issues with no epoch_index (session-wide)."""
    return [i for i in issues if i.epoch_index is None]


def _severity_counts(issues: list[IssueOut]) -> dict[str, int]:
    counts = {"critical": 0, "warning": 0, "info": 0}
    for issue in issues:
        sev = issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity)
        if sev in counts:
            counts[sev] += 1
    return counts


def _build_sustainability_insight(
    issues: list[IssueOut], epochs: list[dict]
) -> SustainabilityInsight:
    """Construct SustainabilityInsight from issues and epoch data."""
    sus_issues = [i for i in issues if i.category == IssueCategory.sustainability]

    insight = SustainabilityInsight(
        sustainability_issue_count=len(sus_issues),
        total_training_duration_seconds=round(
            sum(e.get("duration_seconds", 0) for e in epochs), 2
        ) if epochs else None,
        total_samples_processed=sum(
            (e.get("throughput") or {}).get("samples_processed", 0) for e in epochs
        ) if epochs else None,
    )

    # Extract fields from specific issue types
    for issue in sus_issues:
        mv = issue.metric_value if isinstance(issue.metric_value, dict) else {}
        if issue.metric_key == "wasted_compute_pct":
            insight.optimal_stop_epoch = mv.get("optimal_stop_epoch")
            insight.wasted_epochs = mv.get("wasted_epochs")
            insight.wasted_compute_pct = mv.get("wasted_compute_pct")
            insight.wasted_duration_seconds = mv.get("wasted_duration_s")
        elif issue.metric_key == "is_dead":
            if issue.layer_id and issue.layer_id not in insight.dead_layers:
                insight.dead_layers.append(issue.layer_id)
        elif issue.metric_key == "gradient_norm_mean" and issue.title.startswith("Vanishing"):
            if issue.layer_id and issue.layer_id not in insight.vanishing_gradient_layers:
                insight.vanishing_gradient_layers.append(issue.layer_id)
        elif issue.metric_key == "activation_var_of_means":
            if issue.layer_id and issue.layer_id not in insight.frozen_output_layers:
                insight.frozen_output_layers.append(issue.layer_id)
        elif issue.metric_key == "activation_correlation":
            insight.redundant_layer_pairs.append({
                "layer_a": mv.get("layer_a"),
                "layer_b": mv.get("layer_b"),
                "correlation": mv.get("avg_correlation"),
            })

    # Get parameter_efficiency_score from last epoch sustainability data
    if epochs:
        for e in reversed(epochs):
            sus_data = (e.get("sustainability") or {})
            score = None
            # Check summary-level sustainability if available
            le = sus_data.get("layer_efficiency")
            if le:
                import math
                ratios = [item["compute_to_param_ratio"] for item in le]
                devs = [abs(math.log2(max(r, 0.001))) for r in ratios]
                avg_dev = sum(devs) / len(devs) if devs else 0
                score = max(0, round(100 - avg_dev * 20, 1))
            if score is not None:
                insight.parameter_efficiency_score = score
                break

    # Carbon footprint from epoch data
    carbon_epochs = [
        e["carbon_emissions"] for e in epochs if e.get("carbon_emissions")
    ]
    if carbon_epochs:
        total_co2 = sum(c["epoch_co2_kg"] for c in carbon_epochs)
        total_energy = sum(c["epoch_energy_kwh"] for c in carbon_epochs)
        total_samples = insight.total_samples_processed or 0
        insight.total_co2_kg = round(total_co2, 8)
        insight.total_energy_kwh = round(total_energy, 8)
        insight.co2_per_epoch_avg_kg = round(total_co2 / len(carbon_epochs), 10)
        insight.co2_per_1k_samples_kg = (
            round(total_co2 / max(total_samples / 1000, 1e-9), 10)
            if total_samples
            else None
        )
        insight.avg_power_draw_watts = round(
            sum(c["power_draw_watts"] for c in carbon_epochs) / len(carbon_epochs),
            2,
        )

    # Extract wasted CO2 from engine check
    for issue in sus_issues:
        mv = issue.metric_value if isinstance(issue.metric_value, dict) else {}
        if issue.metric_key == "wasted_co2_kg":
            insight.wasted_co2_kg = mv.get("wasted_co2_kg")

    return insight


def _issue_data_to_model(issue: IssueData, run_id: int) -> DiagnosticIssue:
    """Convert engine's IssueData to a DiagnosticIssue ORM object."""
    return DiagnosticIssue(
        run_id=run_id,
        severity=issue.severity,
        category=issue.category,
        title=issue.title,
        description=issue.description,
        epoch_index=issue.epoch_index,
        layer_id=issue.layer_id,
        metric_key=issue.metric_key,
        metric_value=issue.metric_value if isinstance(issue.metric_value, dict) else {"value": issue.metric_value},
        suggestion=issue.suggestion,
    )


def _db_issue_to_out(issue: DiagnosticIssue) -> IssueOut:
    return IssueOut(
        id=issue.id,
        severity=issue.severity,
        category=issue.category,
        title=issue.title,
        description=issue.description,
        epoch_index=issue.epoch_index,
        layer_id=issue.layer_id,
        metric_key=issue.metric_key,
        metric_value=issue.metric_value,
        suggestion=issue.suggestion,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/run", response_model=DiagnosticRunOut)
def run_session_diagnostics(session_id: int, db: SessionDep):
    """
    Trigger a full diagnostic run for a training session.

    Pulls all TrainSteps and SessionLogs, runs the heuristics engine,
    persists results to DiagnosticRun and DiagnosticIssue tables,
    and returns the full analysis including layer_highlights and epoch_trends.
    """
    # Fetch session
    session = db.get(TrainSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Fetch train steps (epochs) ordered by step_index
    steps = db.exec(
        select(TrainStep)
        .where(TrainStep.session_id == session_id)
        .order_by(TrainStep.step_index)
    ).all()

    # Build epochs list from step columns
    epochs: list[dict] = [_step_to_epoch(s) for s in steps]

    # Fetch session logs
    logs = db.exec(
        select(SessionLog).where(SessionLog.session_id == session_id)
    ).all()

    # Fetch model architecture (first model linked to session)
    model: Model | None = db.exec(
        select(Model).where(Model.session_id == session_id)
    ).first()
    arch: dict | None = model.architecture if model else None
    hp: dict | None = model.hyperparameters if model else None

    # Run diagnostics engine
    issue_data_list, health_score, arch_type = run_diagnostics(epochs, logs, arch, hp)

    # Build summary json
    summary_json = {
        "severity_breakdown": {
            "critical": sum(1 for i in issue_data_list if i.severity == IssueSeverity.critical),
            "warning": sum(1 for i in issue_data_list if i.severity == IssueSeverity.warning),
            "info": sum(1 for i in issue_data_list if i.severity == IssueSeverity.info),
        },
        "category_breakdown": {},
    }
    for issue in issue_data_list:
        cat = issue.category.value
        summary_json["category_breakdown"][cat] = summary_json["category_breakdown"].get(cat, 0) + 1

    # Persist DiagnosticRun
    run = DiagnosticRun(
        session_id=session_id,
        health_score=health_score,
        issue_count=len(issue_data_list),
        arch_type=arch_type,
        summary_json=summary_json,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Persist DiagnosticIssues
    db_issues: list[DiagnosticIssue] = []
    run_id = run.id  # type: ignore[assignment]  # refreshed, so id is set
    for issue_data in issue_data_list:
        db_issue = _issue_data_to_model(issue_data, run_id)
        db.add(db_issue)
        db_issues.append(db_issue)
    db.commit()

    publish_from_sync(SSEEvent(
        event_type=EventType.diagnostic_completed,
        project_id=session.project_id,
        session_id=session_id,
        data={"health_score": health_score, "issue_count": len(issue_data_list)},
    ))

    # Refresh to get IDs
    for db_issue in db_issues:
        db.refresh(db_issue)

    # Build response
    issues_out = [_db_issue_to_out(i) for i in db_issues]
    sustainability = _build_sustainability_insight(issues_out, epochs)
    return DiagnosticRunOut(
        id=run_id,
        session_id=run.session_id,
        created_at=run.created_at,
        health_score=run.health_score,
        issue_count=run.issue_count,
        arch_type=run.arch_type,
        summary_json=run.summary_json,
        issues=issues_out,
        epoch_trends=_build_epoch_trends(issues_out),
        session_level_issues=_build_session_level_issues(issues_out),
        layer_highlights=_build_layer_highlights(issues_out, arch),
        sustainability=sustainability,
    )


@router.get("/sessions/{session_id}", response_model=list[DiagnosticRunSummary])
def list_session_diagnostic_runs(session_id: int, db: SessionDep):
    """List all past diagnostic runs for a session (summary only)."""
    session = db.get(TrainSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    runs = db.exec(
        select(DiagnosticRun)
        .where(DiagnosticRun.session_id == session_id)
        .order_by(DiagnosticRun.created_at.desc())
    ).all()

    return [
        DiagnosticRunSummary(
            id=r.id,
            session_id=r.session_id,
            created_at=r.created_at,
            health_score=r.health_score,
            issue_count=r.issue_count,
            arch_type=r.arch_type,
            summary_json=r.summary_json,
        )
        for r in runs
    ]


@router.get("/runs/{run_id}", response_model=DiagnosticRunOut)
def get_diagnostic_run(run_id: int, db: SessionDep):
    """Get full detail for a specific diagnostic run including all issues."""
    run = db.get(DiagnosticRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Diagnostic run not found")

    # Fetch issues
    db_issues = db.exec(
        select(DiagnosticIssue).where(DiagnosticIssue.run_id == run_id)
    ).all()

    # Fetch architecture for layer_highlights
    model: Model | None = db.exec(
        select(Model).where(Model.session_id == run.session_id)
    ).first()
    arch: dict | None = model.architecture if model else None

    issues_out = [_db_issue_to_out(i) for i in db_issues]

    # Reconstruct epochs for sustainability insight
    steps = db.exec(
        select(TrainStep)
        .where(TrainStep.session_id == run.session_id)
        .order_by(TrainStep.step_index)
    ).all()
    epochs: list[dict] = [_step_to_epoch(s) for s in steps]

    sustainability = _build_sustainability_insight(issues_out, epochs)
    return DiagnosticRunOut(
        id=run.id,
        session_id=run.session_id,
        created_at=run.created_at,
        health_score=run.health_score,
        issue_count=run.issue_count,
        arch_type=run.arch_type,
        summary_json=run.summary_json,
        issues=issues_out,
        epoch_trends=_build_epoch_trends(issues_out),
        session_level_issues=_build_session_level_issues(issues_out),
        layer_highlights=_build_layer_highlights(issues_out, arch),
        sustainability=sustainability,
    )


@router.get("/sessions/{session_id}/health", response_model=HealthOut)
def get_session_health(session_id: int, db: SessionDep):
    """
    Get health summary for a session.

    Returns the latest diagnostic run's health score if available,
    otherwise computes on-the-fly without persisting.
    """
    session = db.get(TrainSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Try to get latest run
    latest_run = db.exec(
        select(DiagnosticRun)
        .where(DiagnosticRun.session_id == session_id)
        .order_by(DiagnosticRun.created_at.desc())
    ).first()

    if latest_run:
        # Use persisted data
        db_issues = db.exec(
            select(DiagnosticIssue).where(DiagnosticIssue.run_id == latest_run.id)
        ).all()
        issues_out = [_db_issue_to_out(i) for i in db_issues]

        model: Model | None = db.exec(
            select(Model).where(Model.session_id == session_id)
        ).first()
        arch = model.architecture if model else None

        return HealthOut(
            session_id=session_id,
            health_score=latest_run.health_score,
            severity_counts=_severity_counts(issues_out),
            issues=sorted(
                issues_out,
                key=lambda x: _SEVERITY_WEIGHT.get(x.severity, 0),
                reverse=True,
            ),
            layers=_build_layer_highlights(issues_out, arch),
        )

    # Compute on-the-fly
    steps = db.exec(
        select(TrainStep)
        .where(TrainStep.session_id == session_id)
        .order_by(TrainStep.step_index)
    ).all()
    epochs = [_step_to_epoch(s) for s in steps]

    logs = db.exec(
        select(SessionLog).where(SessionLog.session_id == session_id)
    ).all()

    model = db.exec(
        select(Model).where(Model.session_id == session_id)
    ).first()
    arch = model.architecture if model else None
    hp = model.hyperparameters if model else None

    issue_data_list, health_score, _ = run_diagnostics(epochs, logs, arch, hp)

    # Convert to IssueOut for consistency
    issues_out = [
        IssueOut(
            severity=i.severity,
            category=i.category,
            title=i.title,
            description=i.description,
            epoch_index=i.epoch_index,
            layer_id=i.layer_id,
            metric_key=i.metric_key,
            metric_value=i.metric_value if isinstance(i.metric_value, dict) else {"value": i.metric_value},
            suggestion=i.suggestion,
        )
        for i in issue_data_list
    ]

    return HealthOut(
        session_id=session_id,
        health_score=health_score,
        severity_counts=_severity_counts(issues_out),
        issues=sorted(
            issues_out,
            key=lambda x: _SEVERITY_WEIGHT.get(x.severity, 0),
            reverse=True,
        ),
        layers=_build_layer_highlights(issues_out, arch),
    )


@router.get("/projects/{project_id}/trend", response_model=ProjectTrendOut)
def get_project_trend(project_id: int, db: SessionDep):
    """
    Get improvement trend across all sessions in a project.

    For each session, returns the latest diagnostic run metrics
    and final training/validation loss. `improving` is true if
    the final loss of the most recent session is lower than the first.
    """
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    sessions = db.exec(
        select(TrainSession)
        .where(TrainSession.project_id == project_id)
        .order_by(TrainSession.started_at)
    ).all()

    trend_items: list[SessionTrendItem] = []
    for sess in sessions:
        # Get latest diagnostic run for this session
        latest_run = db.exec(
            select(DiagnosticRun)
            .where(DiagnosticRun.session_id == sess.id)
            .order_by(DiagnosticRun.created_at.desc())
        ).first()
        diagnostic_run_count = db.exec(
            select(func.count(DiagnosticRun.id)).where(DiagnosticRun.session_id == sess.id)
        ).one() or 0

        # Get final epoch metrics from summary or last step
        final_train_loss: float | None = None
        final_val_loss: float | None = None
        val_acc: float | None = None

        if sess.summary:
            lt = sess.summary.get("loss_trend") or {}
            final_train_loss = lt.get("last")

        # Try to get detailed metrics from last step
        last_step = db.exec(
            select(TrainStep)
            .where(TrainStep.session_id == sess.id)
            .order_by(TrainStep.step_index.desc())
        ).first()
        if last_step and last_step.loss:
            loss_data = last_step.loss
            if final_train_loss is None:
                final_train_loss = loss_data.get("train_mean")
            val_info = loss_data.get("val") or {}
            final_val_loss = val_info.get("val_loss")
            val_acc = val_info.get("val_acc")

        trend_items.append(SessionTrendItem(
            session_id=sess.id,
            run_id=sess.run_id,
            run_name=sess.run_name,
            started_at=sess.started_at,
            health_score=latest_run.health_score if latest_run else None,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            val_acc=val_acc,
            issue_count=latest_run.issue_count if latest_run else None,
            diagnostic_run_count=diagnostic_run_count,
        ))

    # Determine if improving (compare first and last sessions with losses)
    improving: bool | None = None
    sessions_with_loss = [t for t in trend_items if t.final_train_loss is not None]
    if len(sessions_with_loss) >= 2:
        first_loss = sessions_with_loss[0].final_train_loss
        last_loss = sessions_with_loss[-1].final_train_loss
        if first_loss is not None and last_loss is not None:
            improving = last_loss < first_loss

    return ProjectTrendOut(
        project_id=project_id,
        sessions=trend_items,
        improving=improving,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fix Prompt generation  (POST to generate / GET to retrieve cached)
# ─────────────────────────────────────────────────────────────────────────────

_FIX_PROMPT_TEMPLATE = """You are an expert ML engineer debugging a training run.

A diagnostic analysis has detected the following issue:

## Issue
- **Title:** {title}
- **Severity:** {severity}
- **Category:** {category}
- **Description:** {description}
{epoch_line}{layer_line}{metric_line}
## Suggested Fix
{suggestion}

## Your Task
Implement the suggested fix above. Provide:
1. The exact code changes needed (with before/after snippets).
2. Where in the training script or model definition to apply them.
3. Expected impact on the training metrics after the fix.
4. Any caveats or alternative approaches worth considering.

Be concise and actionable. Use code blocks for all code."""


def _build_fix_prompt(issue: DiagnosticIssue) -> str:
    """Build a system prompt from a DiagnosticIssue row."""
    epoch_line = f"- **Epoch:** {issue.epoch_index}\n" if issue.epoch_index is not None else ""
    layer_line = f"- **Layer:** `{issue.layer_id}`\n" if issue.layer_id else ""
    metric_line = ""
    if issue.metric_key:
        val = issue.metric_value if issue.metric_value is not None else "N/A"
        metric_line = f"- **Metric:** {issue.metric_key} = {val}\n"

    return _FIX_PROMPT_TEMPLATE.format(
        title=issue.title,
        severity=issue.severity.value if hasattr(issue.severity, 'value') else issue.severity,
        category=issue.category.value if hasattr(issue.category, 'value') else issue.category,
        description=issue.description,
        epoch_line=epoch_line,
        layer_line=layer_line,
        metric_line=metric_line,
        suggestion=issue.suggestion or "No specific suggestion available.",
    )


@router.post("/issues/{issue_id}/prompt", response_model=FixPromptOut)
def generate_fix_prompt(issue_id: int, db: SessionDep):
    """
    Generate (and cache) an actionable system prompt from a diagnostic issue's
    suggested fix. Returns the prompt text ready to paste into an LLM.
    """
    # Return from cache if available
    if issue_id in _fix_prompt_cache:
        return FixPromptOut(
            issue_id=issue_id,
            prompt=_fix_prompt_cache[issue_id],
            cached=True,
        )

    issue = db.get(DiagnosticIssue, issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="Issue not found")

    prompt = _build_fix_prompt(issue)
    _fix_prompt_cache[issue_id] = prompt

    return FixPromptOut(issue_id=issue_id, prompt=prompt, cached=False)


@router.get("/issues/{issue_id}/prompt", response_model=FixPromptOut)
def get_fix_prompt(issue_id: int, db: SessionDep):
    """
    Retrieve a previously cached fix prompt. If not cached yet, generates it.
    """
    if issue_id in _fix_prompt_cache:
        return FixPromptOut(
            issue_id=issue_id,
            prompt=_fix_prompt_cache[issue_id],
            cached=True,
        )

    issue = db.get(DiagnosticIssue, issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="Issue not found")

    prompt = _build_fix_prompt(issue)
    _fix_prompt_cache[issue_id] = prompt

    return FixPromptOut(issue_id=issue_id, prompt=prompt, cached=False)
