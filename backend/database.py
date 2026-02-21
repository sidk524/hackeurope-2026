from typing import Annotated
from fastapi import Depends
from sqlmodel import create_engine, text
import os

from sqlmodel import Session

# Database URL configuration
DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

if not DATABASE_URL:
    raise Exception("DATABASE_URL not provided!")

engine = create_engine(DATABASE_URL)


def get_session():
    with Session(engine) as session:
        session.exec(text("SET LOCAL SESSION AUTHORIZATION 'anonymous';"))
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
