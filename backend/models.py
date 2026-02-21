from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship, SQLModel

class SessionStatus(str, Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    pending = "pending"
    analyzing = "analyzing"
    stopped = "stopped"


class LogKind(str, Enum):
    console = "console"
    error = "error"


class IssueSeverity(str, Enum):
    critical = "critical"
    warning = "warning"
    info = "info"


class IssueCategory(str, Enum):
    loss = "loss"
    throughput = "throughput"
    memory = "memory"
    profiler = "profiler"
    logs = "logs"
    system = "system"
    architecture = "architecture"
    sustainability = "sustainability"


class Project(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    sessions: list["TrainSession"] = Relationship(back_populates="project")


class TrainSession(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    project_id: int = Field(foreign_key="project.id")
    run_id: str
    run_name: str
    started_at: str
    ended_at: str | None = None
    device: str = ""
    cuda_available: bool = False
    pytorch_version: str = ""
    config: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    summary: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))

    status: SessionStatus = SessionStatus.running

    project: Project = Relationship(back_populates="sessions")
    steps: list["TrainStep"] = Relationship(back_populates="session")
    session_logs: list["SessionLog"] = Relationship(back_populates="session")
    models: list["Model"] = Relationship(back_populates="session")
    diagnostic_runs: list["DiagnosticRun"] = Relationship(back_populates="session")


class Model(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    session_id: int = Field(foreign_key="trainsession.id", unique=True)
    architecture: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON()))
    hyperparameters: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON()))

    session: TrainSession = Relationship(back_populates="models")

class TrainStep(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    session_id: int = Field(foreign_key="trainsession.id")
    step_index: int
    timestamp: str
    duration_seconds: float
    loss: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON()))
    throughput: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON()))
    profiler: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON()))
    memory: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON()))
    system: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON()))
    layer_health: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    sustainability: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    carbon_emissions: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    log_counts: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))

    session: TrainSession = Relationship(back_populates="steps")


class SessionLog(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    session_id: int = Field(foreign_key="trainsession.id")
    ts: str
    level: str
    msg: str
    module: str = ""
    lineno: int = 0
    kind: LogKind = LogKind.console

    session: TrainSession = Relationship(back_populates="session_logs")


class DiagnosticRun(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    session_id: int = Field(foreign_key="trainsession.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    health_score: int = 100
    issue_count: int = 0
    arch_type: str = "generic"
    summary_json: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))

    session: TrainSession = Relationship(back_populates="diagnostic_runs")
    issues: list["DiagnosticIssue"] = Relationship(back_populates="run")


class DiagnosticIssue(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    run_id: int = Field(foreign_key="diagnosticrun.id")
    severity: IssueSeverity
    category: IssueCategory
    title: str
    description: str
    epoch_index: int | None = Field(default=None, nullable=True)
    layer_id: str | None = Field(default=None, nullable=True)
    metric_key: str | None = Field(default=None, nullable=True)
    metric_value: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON()))
    suggestion: str = ""

    run: DiagnosticRun = Relationship(back_populates="issues")
