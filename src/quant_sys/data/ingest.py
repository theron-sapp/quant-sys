from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import re
import pandas as pd
import yfinance as yf
from rich.progress import track
from sqlalchemy.engine import Engine

from ..core.config import Settings
from ..core.storage import get_engine, init_db, symbols as tbl_symbols, prices_daily as tbl_daily, upsert_dataframe

WIKI_SNP_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def _fix_symbol(s: str) -> str:
    # yfinance uses '-' for dots: BRK.B -> BRK-B, BF.B -> BF-B
    return s.replace(".", "-").strip().upper()

def fetch_sp500_from_wikipedia() -> pd.DataFrame:
    dfs = pd.read_html(WIKI_SNP_URL)
    df = dfs[0].copy()
    df["Symbol"] = df["Symbol"].apply(_fix_symbol)
    df.rename(columns={"Security": "Name", "GICS Sector": "Sector"}, inplace=True)
    return df[["Symbol", "Name", "Sector"]]

def ensure_universe_csv(settings: Settings) -> pd.DataFrame:
    csv_path = Path(settings.paths.universe_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        live = fetch_sp500_from_wikipedia()
        if csv_path.exists():
            # merge/overwrite existing
            existing = pd.read_csv(csv_path)
            existing["symbol"] = existing["symbol"].apply(str.upper)
            merged = (live.rename(columns={"Symbol":"symbol","Name":"name","Sector":"sector"})
                        .drop_duplicates("symbol"))
        else:
            merged = live.rename(columns={"Symbol":"symbol","Name":"name","Sector":"sector"})
        # include extras
        extras = pd.DataFrame({"symbol": settings.data.symbols_extra,
                               "name": settings.data.symbols_extra,
                               "sector": "ETF"})
        merged = pd.concat([merged, extras], ignore_index=True).drop_duplicates("symbol")
        merged.to_csv(csv_path, index=False)
        return merged
    except Exception:
        # fallback to local CSV if offline
        if not csv_path.exists():
            raise RuntimeError("No universe_sp500.csv found and could not fetch from Wikipedia.")
        return pd.read_csv(csv_path)

def download_ohlcv(symbols: List[str], start: str) -> dict[str, pd.DataFrame]:
    # Use yfinance bulk download for speed, then split by ticker
    data = yf.download(
        tickers=" ".join(symbols),
        start=start,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    out: dict[str, pd.DataFrame] = {}
    multi = isinstance(data.columns, pd.MultiIndex)
    for sym in symbols:
        try:
            df = data[sym] if multi else data
        except KeyError:
            continue
        df = df.rename(columns=lambda c: c.lower().replace(" ", "_"))
        df = df.rename(columns={"adj_close":"adj_close"})  # ensure the name exists
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna(how="any")
        df["symbol"] = sym
        df["date"] = df.index.strftime("%Y-%m-%d")
        df = df[["symbol","date","open","high","low","close","adj_close","volume"]]
        out[sym] = df.reset_index(drop=True)
    return out

def ingest(settings: Settings, top_n: int | None = None, no_autofetch: bool = False) -> Tuple[Engine, pd.DataFrame]:
    eng = get_engine(settings.paths.db_path)
    init_db(eng)

    if no_autofetch:
        uni = pd.read_csv(settings.paths.universe_file)
    else:
        uni = ensure_universe_csv(settings)

    # liquidity trimming (proxy: use recent avg dollar volume via yfinance if needed later)
    if top_n is None:
        top_n = settings.data.universe_trim_top_liquidity
    # For MVP, just take first N (the CSV is already roughly large caps first on Wikipedia)
    uni = uni.head(top_n).copy()

    # write symbols table
    sym_df = uni.rename(columns={"symbol":"symbol","name":"name","sector":"sector"})
    with eng.begin() as conn:
        conn.execute(tbl_symbols.delete())
    upsert_dataframe(eng, tbl_symbols, sym_df, pk_cols=["symbol"])

    # pull prices
    sym_list = sym_df["symbol"].tolist()
    chunks = []
    for batch in track([sym_list[i:i+50] for i in range(0, len(sym_list), 50)], description="Downloading OHLCV"):
        d = download_ohlcv(batch, settings.data.start)
        for df in d.values():
            chunks.append(df)

    if chunks:
        all_df = pd.concat(chunks, ignore_index=True)
        # raw parquet by symbol (optional)
        raw_dir = Path(settings.paths.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
        for sym, g in all_df.groupby("symbol"):
            g.to_parquet(raw_dir / f"{sym}.parquet", index=False)
        # upsert into SQLite
        upsert_dataframe(eng, tbl_daily, all_df, pk_cols=["symbol","date"])

    return eng, sym_df
