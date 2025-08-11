from __future__ import annotations
from pathlib import Path
import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine

from ..core.storage import prices_daily as tbl_daily, prices_weekly as tbl_weekly, upsert_dataframe
from ..core.config import Settings
from ..core.calendar import weekly_label

def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample per-symbol daily bars to weekly (Fri close). Label by Monday date.
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol","date"])
    out = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.set_index("date")
        wk = pd.DataFrame({
            "open": g["open"].resample("W-FRI").first(),
            "high": g["high"].resample("W-FRI").max(),
            "low":  g["low"].resample("W-FRI").min(),
            "close":g["close"].resample("W-FRI").last(),
            "adj_close": g["adj_close"].resample("W-FRI").last(),
            "volume": g["volume"].resample("W-FRI").sum(),
        }).dropna(how="any")
        wk["symbol"] = sym
        wk["week"] = wk.index.map(lambda d: weekly_label(d)).astype(str)
        wk = wk.reset_index(drop=True)
        out.append(wk[["symbol","week","open","high","low","close","adj_close","volume"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def transform_daily_to_weekly(engine: Engine, settings: Settings) -> pd.DataFrame:
    with engine.begin() as conn:
        daily = pd.read_sql(select(tbl_daily), conn)

    weekly = _to_weekly(daily)
    upsert_dataframe(engine, tbl_weekly, weekly, pk_cols=["symbol","week"])

    # store curated parquet by week
    curated_dir = Path(settings.paths.curated_dir); curated_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(curated_dir / "prices_weekly.parquet", index=False)
    return weekly
