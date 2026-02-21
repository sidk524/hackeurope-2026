from fastapi import APIRouter, Depends
from sqlmodel import select
from pydantic import BaseModel
from database import SessionDep
from models import Project

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    responses={404: {"description": "Not found"}},
)


class ProjectCreate(BaseModel):
    name: str


@router.post("/", response_model=Project)   
def create_project(project: ProjectCreate, session: SessionDep):
    project = Project(name=project.name)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project

    
@router.get("/", response_model=list[Project])
def get_projects(session: SessionDep):
    projects = session.exec(select(Project)).all()
    return projects

    
@router.get("/{project_id}", response_model=Project)
def get_project(project_id: int, session: SessionDep):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


