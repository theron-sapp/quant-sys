from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd
from sqlalchemy import (
    create_engine, MetaData, Table, Column, String, Integer, Float, PrimaryKeyConstraint, text
)
from sqlalchemy.engine import Engine

metadata = MetaData()

symbols = Table(
    "symbols", metadata,
    Column("symbol", String, primary_key=True),
    Column("name", String),
    Column("sector", String),
    Column("is_active", Integer, default=1),
)

prices_daily = Table(
    "prices_daily", metadata,
    Column("symbol", String, nullable=False),
    Column("date", String, nullable=False),
    Column("open", Float), Column("high", Float), Column("low", Float),
    Column("close", Float), Column("adj_close", Float), Column("volume", Integer),
    PrimaryKeyConstraint("symbol", "date")
)

prices_weekly = Table(
    "prices_weekly", metadata,
    Column("symbol", String, nullable=False),
    Column("week", String, nullable=False),
    Column("open", Float), Column("high", Float), Column("low", Float),
    Column("close", Float), Column("adj_close", Float), Column("volume", Integer),
    PrimaryKeyConstraint("symbol", "week")
)

def get_engine(db_path: str) -> Engine:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(f"sqlite:///{db_path}", future=True)
    return eng

def init_db(engine: Engine) -> None:
    metadata.create_all(engine)

def upsert_dataframe(engine: Engine, table: Table, df: pd.DataFrame, pk_cols: Iterable[str]) -> None:
    """
    Use SQLite 'INSERT OR REPLACE' semantics for a whole DataFrame.
    """
    if df.empty:
        return
    # enforce column order
    cols = list(df.columns)
    with engine.begin() as conn:
        # create temp table to speed bulk insert (optional)
        # Direct 'OR REPLACE' for simplicity:
        placeholders = ",".join([":" + c for c in cols])
        collist = ",".join(cols)
        stmt = f"INSERT OR REPLACE INTO {table.name} ({collist}) VALUES ({placeholders})"
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.execute(text("PRAGMA synchronous=NORMAL;"))
        conn.execute(text(stmt), df.to_dict(orient="records"))
