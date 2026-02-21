from typing import Literal
from pydantic import BaseModel
from datetime import datetime
from fastapi import APIRouter, HTTPException
from sqlmodel import select
from database import SessionDep
from models import Model, TrainSession, SessionStatus


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


class TrainSessionUpdate(BaseModel):
    ended_at: datetime | None = None
    summary: dict | None = None
    status: SessionStatus | None = None

    
@router.post("/project/{project_id}", response_model=TrainSession)
def create_train_session(project_id: int, train_session: TrainSessionCreate, session: SessionDep):
    train_session = TrainSession(project_id=project_id, **train_session.model_dump())
    session.add(train_session)
    session.commit()
    session.refresh(train_session)
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
def update_train_session(session_id: int, train_session_update: TrainSessionUpdate, session: SessionDep):
    train_session_db = session.get(TrainSession, session_id)
    if not train_session_db:
        raise HTTPException(status_code=404, detail="Session not found")
    train_session_db.sqlmodel_update(train_session_update.model_dump(exclude_unset=True))
    session.commit()
    session.refresh(train_session_db)
    return train_session_db


@router.get("/{session_id}/status", response_model=SessionStatus)
def get_train_session_status(session_id: int, session: SessionDep):
    train_session = session.get(TrainSession, session_id)
    if not train_session:
        raise HTTPException(status_code=404, detail="Session not found")
    return train_session.status

    
@router.post("/{session_id}/model", response_model=Model)
def register_model(session_id: int, model_create_request: ModelCreateRequest, session: SessionDep):
    model_db = Model(session_id=session_id, **model_create_request.model_dump())
    session.add(model_db)
    session.commit()
    session.refresh(model_db)
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