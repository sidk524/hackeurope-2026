from typing import Annotated
from fastapi import Depends
from sqlmodel import create_engine, text
import os

from sqlmodel import Session

# Database URL configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

if not DATABASE_URL:
    raise Exception("DATABASE_URL not provided!")

engine = create_engine(DATABASE_URL)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
