from typing import Generator
from sqlmodel import SQLModel, create_engine, Session
from .core.config import settings
import os
from pathlib import Path
from sqlalchemy import event
from sqlalchemy.engine import Engine


db_path = Path(settings.DB_URL.replace("sqlite:///", ""))
if db_path.suffix == ".db":
    db_path.parent.mkdir(parents=True, exist_ok=True)

connect_args = {}
if settings.DB_URL.startswith("sqlite"):
    # Better concurrency for SQLite
    connect_args = {"check_same_thread": False}
engine = create_engine(settings.DB_URL, echo=False, connect_args=connect_args)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)

    # Apply SQLite PRAGMAs for performance if using SQLite
    if settings.DB_URL.startswith("sqlite"):
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
