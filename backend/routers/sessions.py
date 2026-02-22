import logging
import time
from typing import Literal
from pydantic import BaseModel
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlmodel import select, Session
from database import SessionDep, engine
from models import (
    DiagnosticIssue,
    DiagnosticRun,
    IssueSeverity,
    LogKind,
    Model,
    SessionLog,
    TrainSession,
    SessionStatus,
    TrainStep,
)
from diagnostics.engine import run_diagnostics
from event_bus import EventType, SSEEvent, event_bus, publish_from_sync

_log = logging.getLogger("sessions.diagnostics")


router = APIRouter(
    prefix="/sessions",
    tags=["sessions"],
    responses={404: {"description": "Not found"}},
)


class ModelCreateRequest(BaseModel):
    architecture: dict
    hyperparameters: dict


class TrainSessionCreate(BaseModel):
    run_id: str
    run_name: str
    started_at: str  # ISO format datetime string
    device: str
    cuda_available: bool
    pytorch_version: str
    config: dict
    status: SessionStatus = SessionStatus.running

class TrainStepCreate(BaseModel):
    step_index: int
    timestamp: str
    duration_seconds: float
    loss: dict
    throughput: dict
    profiler: dict
    memory: dict
    system: dict
    layer_health: dict | None = None
    sustainability: dict | None = None
    carbon_emissions: dict | None = None
    log_counts: dict | None = None

class TrainSessionUpdate(BaseModel):
    ended_at: datetime | None = None
    summary: dict | None = None
    status: SessionStatus | None = None


class SessionLogCreate(BaseModel):
    ts: str
    level: str
    msg: str
    module: str = ""
    lineno: int = 0
    kind: Literal["console", "error"] = "console"


class SessionLogBatchCreate(BaseModel):
    logs: list[SessionLogCreate]


class SessionActionRequest(BaseModel):
    action: Literal["stop", "resume"]

    
@router.post("/project/{project_id}", response_model=TrainSession)
def create_train_session(project_id: int, train_session: TrainSessionCreate, session: SessionDep):
    train_session = TrainSession(project_id=project_id, **train_session.model_dump())
    session.add(train_session)
    session.commit()
    session.refresh(train_session)
    publish_from_sync(SSEEvent(
        event_type=EventType.session_created,
        project_id=project_id,
        session_id=train_session.id,
        data={"run_name": train_session.run_name, "status": train_session.status},
    ))
    return train_session

    
@router.get("/project/{project_id}", response_model=list[TrainSession])
def get_train_sessions(project_id: int, session: SessionDep):
    train_sessions = session.exec(
        select(TrainSession)
        .where(TrainSession.project_id == project_id)
        .order_by(TrainSession.id.desc())
    ).all()
    return train_sessions


@router.get("/{session_id}", response_model=TrainSession)
def get_train_session(session_id: int, session: SessionDep):
    session = session.get(TrainSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.patch("/{session_id}", response_model=TrainSession)
def update_train_session(session_id: int, train_session_update: TrainSessionUpdate, session: SessionDep):
    train_session_db = session.get(TrainSession, session_id)
    if not train_session_db:
        raise HTTPException(status_code=404, detail="Session not found")
    train_session_db.sqlmodel_update(train_session_update.model_dump(exclude_unset=True))
    session.commit()
    session.refresh(train_session_db)
    publish_from_sync(SSEEvent(
        event_type=EventType.session_updated,
        project_id=train_session_db.project_id,
        session_id=session_id,
        data={"status": train_session_db.status},
    ))
    return train_session_db


@router.get("/{session_id}/status", response_model=SessionStatus)
def get_train_session_status(session_id: int, session: SessionDep):
    train_session = session.get(TrainSession, session_id)
    if not train_session:
        raise HTTPException(status_code=404, detail="Session not found")
    return train_session.status


@router.post("/{session_id}/action", response_model=TrainSession)
def session_action(session_id: int, body: SessionActionRequest, session: SessionDep):
    train_session = session.get(TrainSession, session_id)
    if not train_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if train_session.status != SessionStatus.pending:
        raise HTTPException(
            status_code=409,
            detail="Action is only available when session is pending",
        )
    if body.action == "stop":
        train_session.status = SessionStatus.stopped
        train_session.ended_at = datetime.utcnow().isoformat()
    else:
        train_session.status = SessionStatus.running
    session.add(train_session)
    session.commit()
    session.refresh(train_session)
    publish_from_sync(SSEEvent(
        event_type=EventType.session_status_changed,
        project_id=train_session.project_id,
        session_id=session_id,
        data={"status": train_session.status, "action": body.action},
    ))
    return train_session


@router.post("/{session_id}/model", response_model=Model)
def register_model(session_id: int, model_create_request: ModelCreateRequest, session: SessionDep):
    model_db = Model(session_id=session_id, **model_create_request.model_dump())
    session.add(model_db)
    session.commit()
    session.refresh(model_db)
    train_session = session.get(TrainSession, session_id)
    publish_from_sync(SSEEvent(
        event_type=EventType.model_registered,
        project_id=train_session.project_id if train_session else None,
        session_id=session_id,
    ))
    return model_db

    
@router.get("/{session_id}/model", response_model=Model)
def get_model(session_id: int, session: SessionDep):
    train_session = session.get(TrainSession, session_id)
    if not train_session:
        raise HTTPException(status_code=404, detail="Session not found")
    model = session.exec(select(Model).where(Model.session_id == train_session.id)).first()
    if not model:
        raise HTTPException(status_code=404, detail="No model registered for this session")
    return model

    
def _set_session_running_after_delay(session_id: int) -> None:
    time.sleep(10)
    with Session(engine) as db:
        train_session = db.get(TrainSession, session_id)
        if train_session:
            train_session.status = SessionStatus.running
            db.add(train_session)
            db.commit()


def _run_step_diagnostics(session_id: int) -> None:
    """Background task: run diagnostics on all steps for a session."""
    try:
        with Session(engine) as db:
            # Fetch all steps
            steps = db.exec(
                select(TrainStep)
                .where(TrainStep.session_id == session_id)
                .order_by(TrainStep.step_index)
            ).all()

            epochs: list[dict] = []
            for step in steps:
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
                epochs.append(epoch)

            # Fetch logs and architecture
            logs = db.exec(
                select(SessionLog).where(SessionLog.session_id == session_id)
            ).all()
            model = db.exec(
                select(Model).where(Model.session_id == session_id)
            ).first()
            arch = model.architecture if model else None

            # Run engine
            issue_data_list, health_score, arch_type = run_diagnostics(
                epochs, logs, arch
            )

            # Build summary
            summary_json = {
                "severity_breakdown": {
                    "critical": sum(
                        1 for i in issue_data_list
                        if i.severity == IssueSeverity.critical
                    ),
                    "warning": sum(
                        1 for i in issue_data_list
                        if i.severity == IssueSeverity.warning
                    ),
                    "info": sum(
                        1 for i in issue_data_list
                        if i.severity == IssueSeverity.info
                    ),
                },
                "category_breakdown": {},
            }
            for issue in issue_data_list:
                cat = issue.category.value
                summary_json["category_breakdown"][cat] = (
                    summary_json["category_breakdown"].get(cat, 0) + 1
                )

            # Persist
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

            for issue_data in issue_data_list:
                db.add(DiagnosticIssue(
                    run_id=run.id,
                    severity=issue_data.severity,
                    category=issue_data.category,
                    title=issue_data.title,
                    description=issue_data.description,
                    epoch_index=issue_data.epoch_index,
                    layer_id=issue_data.layer_id,
                    metric_key=issue_data.metric_key,
                    metric_value=(
                        issue_data.metric_value
                        if isinstance(issue_data.metric_value, dict)
                        else {"value": issue_data.metric_value}
                    ),
                    suggestion=issue_data.suggestion,
                ))
            db.commit()

            # Set session status based on results
            train_session = db.get(TrainSession, session_id)
            if train_session:
                has_critical = any(
                    i.severity == IssueSeverity.critical
                    for i in issue_data_list
                )
                # Count warnings per layer
                warnings_per_layer: dict[str | None, int] = {}
                for i in issue_data_list:
                    if i.severity == IssueSeverity.warning and i.layer_id:
                        warnings_per_layer[i.layer_id] = (
                            warnings_per_layer.get(i.layer_id, 0) + 1
                        )
                has_warning_cluster = any(
                    count >= 3 for count in warnings_per_layer.values()
                )
                train_session.status = (
                    SessionStatus.pending
                    if has_critical or has_warning_cluster
                    else SessionStatus.running
                )
                db.add(train_session)
                db.commit()
                publish_from_sync(SSEEvent(
                    event_type=EventType.diagnostic_completed,
                    project_id=train_session.project_id,
                    session_id=session_id,
                    data={"health_score": health_score, "issue_count": len(issue_data_list)},
                ))
                publish_from_sync(SSEEvent(
                    event_type=EventType.session_status_changed,
                    project_id=train_session.project_id,
                    session_id=session_id,
                    data={"status": train_session.status, "action": "diagnostics"},
                ))

            _log.info(
                f"Diagnostics for session {session_id}: "
                f"health={health_score}, issues={len(issue_data_list)}"
            )
    except Exception as e:
        _log.error(f"Diagnostics failed for session {session_id}: {e}")
        # On failure, set back to pending
        try:
            with Session(engine) as db:
                train_session = db.get(TrainSession, session_id)
                if train_session:
                    train_session.status = SessionStatus.pending
                    db.add(train_session)
                    db.commit()
        except Exception:
            pass


@router.post("/{session_id}/step", response_model=TrainStep)
def register_step(
    session_id: int,
    step_create_request: TrainStepCreate,
    session: SessionDep,
    background: BackgroundTasks,
):
    train_session = session.get(TrainSession, session_id)
    if not train_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if train_session.status != SessionStatus.running:
        raise HTTPException(status_code=400, detail="Session is not running")
    train_session.status = SessionStatus.analyzing  # Diagnostics will run in background
    session.add(train_session)
    step_db = TrainStep(session_id=session_id, **step_create_request.model_dump())
    session.add(step_db)
    session.commit()
    session.refresh(step_db)
    publish_from_sync(SSEEvent(
        event_type=EventType.step_registered,
        project_id=train_session.project_id,
        session_id=session_id,
        data={"step_index": step_db.step_index},
    ))
    background.add_task(_run_step_diagnostics, session_id)
    return step_db

@router.get("/{session_id}/step", response_model=list[TrainStep])
def get_steps(session_id: int, session: SessionDep):
    train_session = session.get(TrainSession, session_id)
    if not train_session:
        raise HTTPException(status_code=404, detail="Session not found")
    steps = session.exec(select(TrainStep).where(TrainStep.session_id == session_id).order_by(TrainStep.step_index.asc())).all()
    return steps


@router.post("/{session_id}/log", response_model=SessionLog, status_code=201)
def create_session_log(session_id: int, body: SessionLogCreate, session: SessionDep):
    train_session = session.get(TrainSession, session_id)
    if not train_session:
        raise HTTPException(status_code=404, detail="Session not found")
    kind = LogKind.console if body.kind == "console" else LogKind.error
    log_db = SessionLog(
        session_id=session_id,
        ts=body.ts,
        level=body.level,
        msg=body.msg,
        module=body.module,
        lineno=body.lineno,
        kind=kind,
    )
    session.add(log_db)
    session.commit()
    session.refresh(log_db)
    publish_from_sync(SSEEvent(
        event_type=EventType.log_created,
        project_id=train_session.project_id,
        session_id=session_id,
        data={"level": body.level, "kind": body.kind},
    ))
    return log_db


@router.post("/{session_id}/logs", response_model=list[SessionLog], status_code=201)
def create_session_logs_batch(session_id: int, body: SessionLogBatchCreate, session: SessionDep):
    """Accept a batch of log entries to reduce request spam from the observer."""
    train_session = session.get(TrainSession, session_id)
    if not train_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not body.logs:
        return []
    created = []
    for log_in in body.logs:
        kind = LogKind.console if log_in.kind == "console" else LogKind.error
        log_db = SessionLog(
            session_id=session_id,
            ts=log_in.ts,
            level=log_in.level,
            msg=log_in.msg,
            module=log_in.module,
            lineno=log_in.lineno,
            kind=kind,
        )
        session.add(log_db)
        created.append(log_db)
    session.commit()
    for log_db in created:
        session.refresh(log_db)
    # Single SSE event for the batch so the UI can refetch or update once
    publish_from_sync(SSEEvent(
        event_type=EventType.log_created,
        project_id=train_session.project_id,
        session_id=session_id,
        data={"batch": True, "count": len(created)},
    ))
    return created


@router.get("/{session_id}/logs", response_model=list[SessionLog])
def get_session_logs(session_id: int, session: SessionDep):
    train_session = session.get(TrainSession, session_id)
    if not train_session:
        raise HTTPException(status_code=404, detail="Session not found")
    logs = session.exec(
        select(SessionLog)
        .where(SessionLog.session_id == session_id)
        .order_by(SessionLog.id.asc())
    ).all()
    return logs