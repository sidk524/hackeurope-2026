"""
Agent tool definitions — OpenAI-format tool schemas that wrap the existing
MCP server functions.  Each tool executor calls the router functions directly
(same pattern as mcp_server.py) to avoid HTTP round-trips.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from fastapi import HTTPException
from sqlmodel import Session, create_engine, select, func, col

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
_engine = create_engine(DATABASE_URL, connect_args={"timeout": 30})


def _serialize(obj: Any) -> Any:
    """Convert Pydantic / SQLModel objects to plain dicts."""
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _call(fn: Callable, *args: Any) -> Any:
    try:
        result = fn(*args)
        return _serialize(result)
    except HTTPException as e:
        return {"error": e.detail}


# ── Tool executors ────────────────────────────────────────────────────────────
# Each function accepts **kwargs parsed from the LLM tool_call arguments and
# returns a JSON-serialisable dict.

def exec_get_session_detail(session_id: int, **_: Any) -> dict:
    """Full session detail + step stats."""
    with Session(_engine) as db:
        result = _call(_api_get_session, session_id, db)
        if isinstance(result, dict) and "error" in result:
            return result

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


def exec_get_training_steps(
    session_id: int,
    start_epoch: int | None = None,
    end_epoch: int | None = None,
    **_: Any,
) -> dict:
    """Per-epoch telemetry with optional range filter."""
    with Session(_engine) as db:
        sess = db.get(TrainSession, session_id)
        if not sess:
            return {"error": f"Session {session_id} not found"}

        all_steps = _call(_api_get_steps, session_id, db)
        if isinstance(all_steps, dict) and "error" in all_steps:
            return all_steps

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


def exec_get_session_health(session_id: int, **_: Any) -> dict:
    """Health score + severity counts + top issues."""
    with Session(_engine) as db:
        return _call(_api_get_health, session_id, db)


def exec_get_diagnostic_run_detail(run_id: int, **_: Any) -> dict:
    """Full diagnostic run: issues, layer highlights, sustainability."""
    with Session(_engine) as db:
        return _call(_api_get_run, run_id, db)


def exec_run_session_diagnostics(session_id: int, **_: Any) -> dict:
    """Trigger a fresh diagnostic run."""
    with Session(_engine) as db:
        return _call(_api_run_diagnostics, session_id, db)


def exec_get_model_architecture(session_id: int, **_: Any) -> dict:
    """Model architecture + hyperparameters."""
    with Session(_engine) as db:
        result = _call(_api_get_model, session_id, db)
        if isinstance(result, dict) and "error" in result:
            return result
        return {"session_id": session_id, **result}


def exec_get_session_logs(
    session_id: int,
    kind: str | None = None,
    limit: int = 50,
    **_: Any,
) -> dict:
    """Session console/error logs."""
    with Session(_engine) as db:
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


def exec_get_project_trend(project_id: int, **_: Any) -> dict:
    """Cross-session improvement trend."""
    with Session(_engine) as db:
        return _call(_api_get_trend, project_id, db)


def exec_get_sustainability_report(session_id: int, **_: Any) -> dict:
    """Aggregated Green-AI sustainability report: carbon footprint, efficiency,
    waste analysis, and problematic layers."""
    with Session(_engine) as db:
        # First get the latest diagnostic run for sustainability insight
        latest_run = db.exec(
            select(DiagnosticRun)
            .where(DiagnosticRun.session_id == session_id)
            .order_by(col(DiagnosticRun.created_at).desc())
        ).first()

        sustainability: dict[str, Any] = {}
        if latest_run:
            full_run = _call(_api_get_run, latest_run.id, db)
            if isinstance(full_run, dict) and full_run.get("sustainability"):
                sustainability = full_run["sustainability"]

        # Get carbon data from training steps
        steps = db.exec(
            select(TrainStep)
            .where(TrainStep.session_id == session_id)
            .order_by(TrainStep.step_index)
        ).all()

        carbon_timeline: list[dict] = []
        total_energy_kwh = 0.0
        total_co2_kg = 0.0
        total_duration = 0.0
        for step in steps:
            total_duration += step.duration_seconds or 0
            if step.carbon_emissions:
                epoch_co2 = step.carbon_emissions.get("epoch_co2_kg", 0) or 0
                epoch_energy = step.carbon_emissions.get("epoch_energy_kwh", 0) or 0
                total_co2_kg += epoch_co2
                total_energy_kwh += epoch_energy
                carbon_timeline.append({
                    "epoch": step.step_index,
                    "co2_kg": epoch_co2,
                    "energy_kwh": epoch_energy,
                    "power_draw_watts": step.carbon_emissions.get("power_draw_watts"),
                })

        # Layer efficiency from sustainability data
        layer_efficiency: list[dict] = []
        for step in steps:
            if step.sustainability and "layer_efficiency" in step.sustainability:
                for le in step.sustainability["layer_efficiency"]:
                    layer_efficiency.append({
                        "epoch": step.step_index,
                        **le,
                    })

        # Cost estimate (€50/ton CO2 — EU ETS ballpark)
        carbon_price_eur_per_ton = 50.0
        estimated_cost_eur = total_co2_kg * carbon_price_eur_per_ton / 1000.0

        return {
            "session_id": session_id,
            "sustainability_insight": sustainability,
            "carbon_timeline": carbon_timeline,
            "total_energy_kwh": round(total_energy_kwh, 6),
            "total_co2_kg": round(total_co2_kg, 6),
            "total_duration_seconds": round(total_duration, 2),
            "estimated_carbon_cost_eur": round(estimated_cost_eur, 4),
            "layer_efficiency_snapshots": layer_efficiency[:50],  # cap for LLM context
            "epoch_count": len(steps),
        }


def exec_compare_sessions(
    session_id_a: int,
    session_id_b: int,
    **_: Any,
) -> dict:
    """Compare two sessions side-by-side: health, loss, carbon."""
    with Session(_engine) as db:
        def _session_summary(sid: int) -> dict:
            sess = db.get(TrainSession, sid)
            if not sess:
                return {"error": f"Session {sid} not found"}

            health = _call(_api_get_health, sid, db)

            steps = db.exec(
                select(TrainStep)
                .where(TrainStep.session_id == sid)
                .order_by(TrainStep.step_index)
            ).all()

            final_loss: dict[str, Any] = {}
            total_co2 = 0.0
            total_energy = 0.0
            total_duration = sum(s.duration_seconds or 0 for s in steps)

            if steps:
                last = steps[-1]
                if last.loss:
                    final_loss = {
                        "train_mean": last.loss.get("train_mean"),
                        "val_loss": (last.loss.get("val") or {}).get("val_loss"),
                    }
                for s in steps:
                    if s.carbon_emissions:
                        total_co2 += s.carbon_emissions.get("epoch_co2_kg", 0) or 0
                        total_energy += s.carbon_emissions.get("epoch_energy_kwh", 0) or 0

            return {
                "session_id": sid,
                "run_name": sess.run_name,
                "status": sess.status.value if hasattr(sess.status, "value") else sess.status,
                "epoch_count": len(steps),
                "final_loss": final_loss,
                "total_duration_seconds": round(total_duration, 2),
                "total_co2_kg": round(total_co2, 6),
                "total_energy_kwh": round(total_energy, 6),
                "health": health if not isinstance(health, dict) or "error" not in health else None,
            }

        return {
            "session_a": _session_summary(session_id_a),
            "session_b": _session_summary(session_id_b),
        }


# ── Tool schemas (OpenAI function-calling format) ────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_session_detail",
            "description": (
                "Get full detail for a training session including config, "
                "summary, step count, final losses, and total duration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The training session ID.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_training_steps",
            "description": (
                "Get training steps/epochs with full telemetry: loss, throughput, "
                "memory, profiler, layer_health, sustainability, carbon emissions. "
                "Optionally filter by epoch range (inclusive)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The training session ID.",
                    },
                    "start_epoch": {
                        "type": "integer",
                        "description": "Start epoch (inclusive). Omit for all.",
                    },
                    "end_epoch": {
                        "type": "integer",
                        "description": "End epoch (inclusive). Omit for all.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_health",
            "description": (
                "Quick health summary: score (0-100), severity counts, "
                "top issues, and most problematic layers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The training session ID.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_diagnostic_run_detail",
            "description": (
                "Get full diagnostic run detail: all issues with severity, "
                "category, description, suggestions, epoch trends, layer "
                "highlights, and sustainability insight."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "integer",
                        "description": "The diagnostic run ID.",
                    },
                },
                "required": ["run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_session_diagnostics",
            "description": (
                "Trigger a fresh diagnostic analysis for a session. Persists "
                "results and returns full analysis with all issues, layer "
                "highlights, epoch trends, and sustainability insight. Use this "
                "when you need the very latest analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The training session ID.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_architecture",
            "description": (
                "Get the model architecture (module tree, all layers with "
                "parameter counts, layer graph) and hyperparameters for a "
                "training session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The training session ID.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_logs",
            "description": (
                "Get session console and error logs. Filter by kind "
                "('console' or 'error'). Returns most recent logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The training session ID.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["console", "error"],
                        "description": "Filter by log kind. Omit for all.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max logs to return (default 50).",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_trend",
            "description": (
                "Get improvement trend across all sessions in a project: "
                "health scores, final losses, and whether the project is "
                "improving over time. Good for cross-run comparison."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "integer",
                        "description": "The project ID.",
                    },
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sustainability_report",
            "description": (
                "Get a comprehensive Green-AI sustainability report: carbon "
                "footprint timeline, total energy/CO2, efficiency scores, "
                "waste analysis (wasted epochs, wasted CO2), problematic "
                "layers (dead neurons, vanishing gradients, frozen outputs, "
                "redundant layers), and estimated carbon cost in EUR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The training session ID.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_sessions",
            "description": (
                "Compare two training sessions side-by-side: health scores, "
                "final losses, carbon footprint, epoch count, and duration. "
                "Useful for understanding what changed between runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id_a": {
                        "type": "integer",
                        "description": "First session ID.",
                    },
                    "session_id_b": {
                        "type": "integer",
                        "description": "Second session ID.",
                    },
                },
                "required": ["session_id_a", "session_id_b"],
            },
        },
    },
]

# Map from tool name → executor function
TOOL_EXECUTORS: dict[str, Callable[..., dict]] = {
    "get_session_detail": exec_get_session_detail,
    "get_training_steps": exec_get_training_steps,
    "get_session_health": exec_get_session_health,
    "get_diagnostic_run_detail": exec_get_diagnostic_run_detail,
    "run_session_diagnostics": exec_run_session_diagnostics,
    "get_model_architecture": exec_get_model_architecture,
    "get_session_logs": exec_get_session_logs,
    "get_project_trend": exec_get_project_trend,
    "get_sustainability_report": exec_get_sustainability_report,
    "compare_sessions": exec_compare_sessions,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict:
    """Execute a tool by name with the given arguments. Returns tool output."""
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return executor(**arguments)
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}
