from datetime import datetime
from typing import Any, Literal

from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship, SQLModel


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

    status: Literal["running", "completed", "failed", "paused", "pending"] = "running"

    project: Project = Relationship(back_populates="sessions")
    epochs: list["Epoch"] = Relationship(back_populates="session")
    session_logs: list["SessionLog"] = Relationship(back_populates="session")
    model: "Model | None" = Relationship(back_populates="session")



class Model(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    session_id: int = Field(foreign_key="session.id", unique=True)
    architecture: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON()))
    hyperparameters: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON()))

    session: TrainSession = Relationship(back_populates="model")

class TrainStep(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    session_id: int = Field(foreign_key="session.id")
    epoch_index: int
    timestamp: str
    duration_seconds: float
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON()))

    session: TrainSession = Relationship(back_populates="epochs")


class SessionLog(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    session_id: int = Field(foreign_key="session.id")
    ts: str
    level: str
    msg: str
    module: str = ""
    lineno: int = 0
    kind: Literal["console", "error"] = "console"

    session: TrainSession = Relationship(back_populates="session_logs")
