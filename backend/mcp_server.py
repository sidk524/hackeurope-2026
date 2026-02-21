"""
MCP Server for ML Diagnostics — "Sentry for ML Pipelines"

Exposes tools for querying training sessions, diagnostic runs,
and triggering diagnostics analysis via the Model Context Protocol.

Internally calls the existing FastAPI router functions directly,
avoiding code duplication.

Run:
    python mcp_server.py                    # stdio (default, for Claude Desktop)
    python mcp_server.py --transport sse    # SSE on port 8100
"""

import os
import sys

from fastmcp import FastMCP
from fastapi import HTTPException
from sqlmodel import Session, create_engine, select, func, col

# Ensure backend package imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    DiagnosticRun,
    Model,
    SessionLog,
    TrainSession,
    TrainStep,
)
from routers.diagnostics import (
    run_session_diagnostics as _api_run_diagnostics,
    list_session_diagnostic_runs as _api_list_runs,
    get_diagnostic_run as _api_get_run,
    get_session_health as _api_get_health,
    get_project_trend as _api_get_trend,
)
from routers.sessions import (
    get_train_sessions as _api_get_sessions,
    get_train_session as _api_get_session,
    get_model as _api_get_model,
    get_steps as _api_get_steps,
)

# ── Database ──────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
engine = create_engine(DATABASE_URL, connect_args={"timeout": 30})


def _serialize(obj) -> dict | list:
    """Convert Pydantic/SQLModel objects to plain dicts."""
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _call(fn, *args):
    """Call an API function, catching HTTPException as error dicts."""
    try:
        result = fn(*args)
        return _serialize(result)
    except HTTPException as e:
        return {"error": e.detail}


# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="ML Diagnostics",
    instructions=(
        "This server provides tools for analyzing ML training sessions. "
        "Use list_sessions to discover sessions, get_session_health for a "
        "quick overview, get_training_steps for raw epoch telemetry, and "
        "get_diagnostic_run_detail for full issue analysis. Use "
        "run_session_diagnostics to trigger a fresh analysis."
    ),
)


@mcp.tool(annotations={"readOnlyHint": True})
def list_sessions(project_id: int | None = None) -> dict:
    """List training sessions with status, health score, and step count.
    Optionally filter by project_id."""
    with Session(engine) as db:
        query = select(TrainSession)
        if project_id is not None:
            query = query.where(TrainSession.project_id == project_id)
        query = query.order_by(col(TrainSession.started_at).desc())
        sessions = db.exec(query).all()

        results = []
        for sess in sessions:
            step_count = db.exec(
                select(func.count(TrainStep.id))
                .where(TrainStep.session_id == sess.id)
            ).one()

            latest_run = db.exec(
                select(DiagnosticRun)
                .where(DiagnosticRun.session_id == sess.id)
                .order_by(col(DiagnosticRun.created_at).desc())
            ).first()

            results.append({
                "id": sess.id,
                "project_id": sess.project_id,
                "run_id": sess.run_id,
                "run_name": sess.run_name,
                "started_at": sess.started_at,
                "ended_at": sess.ended_at,
                "status": sess.status.value,
                "device": sess.device,
                "step_count": step_count,
                "latest_health_score": latest_run.health_score if latest_run else None,
            })

        return {"sessions": results}


@mcp.tool(annotations={"readOnlyHint": True})
def get_session_detail(session_id: int) -> dict:
    """Get full detail for a training session including config, summary,
    and training statistics."""
    with Session(engine) as db:
        result = _call(_api_get_session, session_id, db)
        if isinstance(result, dict) and "error" in result:
            return result

        # Enrich with step stats
        steps = db.exec(
            select(TrainStep)
            .where(TrainStep.session_id == session_id)
            .order_by(TrainStep.step_index)
        ).all()

        final_train_loss = None
        final_val_loss = None

        sess = db.get(TrainSession, session_id)
        if sess and sess.summary:
            lt = sess.summary.get("loss_trend") or {}
            final_train_loss = lt.get("last")

        if steps:
            last_step = steps[-1]
            if last_step.loss:
                if final_train_loss is None:
                    final_train_loss = last_step.loss.get("train_mean")
                val_info = last_step.loss.get("val") or {}
                final_val_loss = val_info.get("val_loss")

        indices = [s.step_index for s in steps]
        return {
            "session": result,
            "step_count": len(steps),
            "epoch_range": [min(indices), max(indices)] if indices else None,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "total_duration_seconds": round(sum(s.duration_seconds for s in steps), 2),
        }


@mcp.tool(annotations={"readOnlyHint": True})
def get_training_steps(
    session_id: int,
    start_epoch: int | None = None,
    end_epoch: int | None = None,
) -> dict:
    """Get training steps/epochs with full telemetry: loss, throughput,
    memory, profiler, layer_health, sustainability, carbon emissions.
    Optionally filter by epoch range (inclusive)."""
    with Session(engine) as db:
        sess = db.get(TrainSession, session_id)
        if not sess:
            return {"error": f"Session {session_id} not found"}

        # Get all steps via API
        all_steps = _call(_api_get_steps, session_id, db)
        if isinstance(all_steps, dict) and "error" in all_steps:
            return all_steps

        # Filter by epoch range if requested
        steps = all_steps
        if start_epoch is not None or end_epoch is not None:
            steps = [
                s for s in all_steps
                if (start_epoch is None or s["step_index"] >= start_epoch)
                and (end_epoch is None or s["step_index"] <= end_epoch)
            ]

        return {
            "session_id": session_id,
            "total_steps": len(all_steps),
            "returned_steps": len(steps),
            "steps": steps,
        }


@mcp.tool(annotations={"readOnlyHint": True})
def list_diagnostic_runs(session_id: int) -> dict:
    """List all diagnostic runs for a session with summary info."""
    with Session(engine) as db:
        result = _call(_api_list_runs, session_id, db)
        if isinstance(result, dict) and "error" in result:
            return result
        return {"session_id": session_id, "runs": result}


@mcp.tool(annotations={"readOnlyHint": True})
def get_diagnostic_run_detail(run_id: int) -> dict:
    """Get full diagnostic run detail: all issues with severity, category,
    description, suggestions, epoch trends, layer highlights, and
    sustainability insight."""
    with Session(engine) as db:
        return _call(_api_get_run, run_id, db)


@mcp.tool(annotations={"readOnlyHint": True})
def get_session_health(session_id: int) -> dict:
    """Quick health summary: score, severity counts, top 5 issues,
    top 3 problematic layers. Uses latest diagnostic run if available,
    otherwise computes on the fly."""
    with Session(engine) as db:
        return _call(_api_get_health, session_id, db)


@mcp.tool
def run_session_diagnostics(session_id: int) -> dict:
    """Trigger a fresh diagnostic run for a session. Persists results to the
    database and returns the full analysis with all issues, layer highlights,
    epoch trends, and sustainability insight."""
    with Session(engine) as db:
        return _call(_api_run_diagnostics, session_id, db)


@mcp.tool(annotations={"readOnlyHint": True})
def get_project_trend(project_id: int) -> dict:
    """Get improvement trend across all sessions in a project: health scores,
    final losses, and whether the project is improving over time."""
    with Session(engine) as db:
        return _call(_api_get_trend, project_id, db)


@mcp.tool(annotations={"readOnlyHint": True})
def get_session_logs(
    session_id: int,
    kind: str | None = None,
    limit: int = 100,
) -> dict:
    """Get session logs. Filter by kind ('console' or 'error').
    Returns up to `limit` most recent logs."""
    with Session(engine) as db:
        sess = db.get(TrainSession, session_id)
        if not sess:
            return {"error": f"Session {session_id} not found"}

        query = (
            select(SessionLog)
            .where(SessionLog.session_id == session_id)
            .order_by(col(SessionLog.ts).desc())
            .limit(limit)
        )
        if kind:
            query = query.where(SessionLog.kind == kind)

        logs = db.exec(query).all()

        total = db.exec(
            select(func.count(SessionLog.id))
            .where(SessionLog.session_id == session_id)
        ).one()

        return {
            "session_id": session_id,
            "total_count": total,
            "returned_count": len(logs),
            "logs": [_serialize(log) for log in logs],
        }


@mcp.tool(annotations={"readOnlyHint": True})
def get_model_architecture(session_id: int) -> dict:
    """Get model architecture (module tree, layers, layer graph)
    and hyperparameters for a session."""
    with Session(engine) as db:
        result = _call(_api_get_model, session_id, db)
        if isinstance(result, dict) and "error" in result:
            return result
        return {"session_id": session_id, **result}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Diagnostics MCP Server")
    parser.add_argument(
        "--transport", default="stdio",
        choices=["stdio", "sse"],
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--port", type=int, default=8100,
        help="Port for SSE transport (default: 8100)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=args.transport, host="127.0.0.1", port=args.port)
