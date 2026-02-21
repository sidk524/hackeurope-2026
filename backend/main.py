import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import ProgrammingError as SQLAlchemyProgrammingError

from routers import projects, sessions, diagnostics

fastapi = FastAPI()

allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")

fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastapi.include_router(projects.router)
fastapi.include_router(sessions.router)
fastapi.include_router(diagnostics.router)