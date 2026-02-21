from fastapi import APIRouter, HTTPException
from sqlalchemy import func
from sqlmodel import select
from pydantic import BaseModel
from database import SessionDep
from models import Project, ProjectBase, SessionStatus, TrainSession

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    responses={404: {"description": "Not found"}},
)


class ProjectCreate(BaseModel):
    name: str

    
class ProjectWithStatus(ProjectBase):
    status: SessionStatus | None = None


@router.post("/", response_model=Project)   
def create_project(project: ProjectCreate, session: SessionDep):
    project = Project(name=project.name)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project

    
@router.get("/", response_model=list[ProjectWithStatus])
def get_projects(session: SessionDep):
    # Subquery: per project, the id of the most recent session (by id = creation order)
    latest_session_id_per_project = (
        select(
            TrainSession.project_id,
            func.max(TrainSession.id).label("latest_session_id"),
        )
        .group_by(TrainSession.project_id)
    ).subquery()
    # Select Project and that session's status; left join so projects with no sessions still appear
    query = (
        select(Project, TrainSession.status)
        .outerjoin(
            latest_session_id_per_project,
            Project.id == latest_session_id_per_project.c.project_id,
        )
        .outerjoin(
            TrainSession,
            TrainSession.id == latest_session_id_per_project.c.latest_session_id,
        )
    )
    rows = session.exec(query).all()
    return [
        ProjectWithStatus(
            id=proj.id,
            name=proj.name,
            created_at=proj.created_at,
            status=st,
        )
        for proj, st in rows
    ]

    
@router.get("/{project_id}", response_model=Project)
def get_project(project_id: int, session: SessionDep):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


