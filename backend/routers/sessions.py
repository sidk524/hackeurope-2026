from typing import Literal
from pydantic import BaseModel
from datetime import datetime
from fastapi import APIRouter, HTTPException
from sqlmodel import select
from backend.database import SessionDep
from backend.models import TrainSession


router = APIRouter(
    prefix="/sessions",
    tags=["sessions"],
    responses={404: {"description": "Not found"}},
)


class TrainSessionCreate(BaseModel):
    run_id: str
    run_name: str
    started_at: datetime
    device: str
    cuda_available: bool
    pytorch_version: str
    config: dict
    status: Literal["running", "completed", "failed", "paused", "pending"] = "running"


class TrainSessionUpdate(BaseModel):
    ended_at: datetime | None = None
    summary: dict | None = None
    status: Literal["completed", "failed", "paused", "pending"] | None = None

    
@router.post("/project/{project_id}", response_model=TrainSession)
def create_train_session(project_id: int, train_session: TrainSessionCreate, session: SessionDep):
    train_session = TrainSession(project_id=project_id, **train_session.model_dump())
    train_session.add(train_session)
    train_session.commit()
    train_session.refresh(train_session)
    return train_session

    
@router.get("/project/{project_id}", response_model=list[TrainSession])
def get_train_sessions(project_id: int, session: SessionDep):
    train_sessions = session.exec(select(TrainSession).where(TrainSession.project_id == project_id)).all()
    return train_sessions


@router.get("/{session_id}", response_model=TrainSession)
def get_train_session(session_id: int, session: SessionDep):
    session = session.get(TrainSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/{session_id}", response_model=TrainSession)
def update_train_session(session_id: int, train_session: TrainSessionUpdate, session: SessionDep):
    session = session.get(TrainSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    train_session.update(train_session.model_dump(exclude_unset=True))
    session.commit()
    session.refresh(train_session)
    return train_session


@router.get("/{session_id}/status", response_model=Literal["running", "completed", "failed", "paused", "pending"])
def get_train_session_status(session_id: int, session: SessionDep):
    session = session.get(TrainSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.status

    