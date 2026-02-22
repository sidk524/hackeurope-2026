import asyncio
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import ProgrammingError as SQLAlchemyProgrammingError

from event_bus import set_event_loop
from routers import projects, sessions, diagnostics, events, llm, agent

fastapi = FastAPI()

# With credentials, browser requires a specific origin (not "*"). Default to Next.js dev origin.
_default_origins = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173"
allowed_origins = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]

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
fastapi.include_router(events.router)
fastapi.include_router(llm.router)
fastapi.include_router(agent.router)


@fastapi.on_event("startup")
async def _capture_event_loop() -> None:
    set_event_loop(asyncio.get_running_loop())
