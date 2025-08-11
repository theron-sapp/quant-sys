from __future__ import annotations
from pathlib import Path
from typing import List
import yaml
from pydantic import BaseModel, Field

class Paths(BaseModel):
    db_path: str = "db/quant.sqlite"
    universe_file: str = "config/universe_sp500.csv"
    raw_dir: str = "data/raw"
    curated_dir: str = "data/curated"

class EquityCosts(BaseModel):
    commission_per_share: float = 0.005
    slippage_bps: int = 8

class Costs(BaseModel):
    equities: EquityCosts = EquityCosts()

class DataCfg(BaseModel):
    start: str = "2005-01-01"
    source: str = "yfinance"
    symbols_extra: List[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM", "^VIX"])
    universe_trim_top_liquidity: int = 300

class Settings(BaseModel):
    capital_base: int = 10000
    gross_exposure_cap: int = 5000
    target_vol_annual: float = 0.11
    max_drawdown_stop: float = 0.25
    rebalance_day: str = "MONDAY_OPEN"
    paths: Paths = Paths()
    data: DataCfg = DataCfg()
    costs: Costs = Costs()

def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    p = Path(path)
    with p.open("r") as f:
        raw = yaml.safe_load(f)
    s = Settings(**raw)

    # ensure dirs exist
    for d in [s.paths.raw_dir, s.paths.curated_dir, Path(s.paths.db_path).parent]:
        Path(d).mkdir(parents=True, exist_ok=True)
    return s
