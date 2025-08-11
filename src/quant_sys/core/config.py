"""
Complete replacement for config.py with all necessary configuration classes.
Location: src/quant_sys/core/config.py

This is the COMPLETE file - replace your entire config.py with this.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
import yaml


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

class Paths(BaseModel):
    """Path configuration."""
    db_path: str = "db/quant.sqlite"
    universe_file: str = "config/universe_sp500.csv"
    raw_dir: str = "data/raw"
    curated_dir: str = "data/curated"


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

class DataCfg(BaseModel):
    """Data configuration."""
    start: str = "2005-01-01"
    source: str = "yfinance"
    symbols_extra: List[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM", "^VIX"])
    universe_trim_top_liquidity: int = 300


# ============================================================================
# COST CONFIGURATION
# ============================================================================

class EquityCosts(BaseModel):
    """Equity trading costs."""
    commission_per_share: float = 0.005
    slippage_bps: int = 8


class OptionCosts(BaseModel):
    """Option trading costs."""
    commission_per_contract: float = 1.0
    spread_penalty_pct: float = 0.10


class Costs(BaseModel):
    """Trading costs configuration."""
    equities: EquityCosts = Field(default_factory=EquityCosts)
    options: Optional[OptionCosts] = Field(default_factory=OptionCosts)


# ============================================================================
# SIGNAL CONFIGURATION
# ============================================================================

class TimeSeriesMomentum(BaseModel):
    """Time-series momentum signal configuration."""
    lookback_weeks: int = 52
    skip_recent_weeks: int = 4
    threshold: float = 0.0
    weight: float = 0.5


class CrossSectionalMomentum(BaseModel):
    """Cross-sectional momentum signal configuration."""
    lookback_weeks: int = 52
    bucket: int = 5
    long_bucket: int = 5
    short_bucket: int = 1
    gross_alloc: float = 0.5
    weight: float = 0.35


class MeanReversion(BaseModel):
    """Mean reversion signal configuration."""
    lookback_days: int = 5
    z_entry: float = 1.0
    weight: float = 0.15


class RiskFilters(BaseModel):
    """Risk filter configuration."""
    vix_percentile_lookback: int = 60
    vix_block_above_pct: int = 85


class Signals(BaseModel):
    """Signal generation configuration."""
    ts_mom: TimeSeriesMomentum = Field(default_factory=TimeSeriesMomentum)
    xs_mom: CrossSectionalMomentum = Field(default_factory=CrossSectionalMomentum)
    mean_rev: MeanReversion = Field(default_factory=MeanReversion)
    risk_filters: RiskFilters = Field(default_factory=RiskFilters)


# ============================================================================
# PORTFOLIO CONFIGURATION
# ============================================================================

class VolatilityScaling(BaseModel):
    """Volatility scaling configuration."""
    ewma_lambda: float = 0.94
    floor_gross: float = 0.30
    cap_gross: float = 1.00


class Portfolio(BaseModel):
    """Portfolio construction configuration."""
    max_names: int = 30
    max_weight_per_name: float = 0.08
    max_sector_weight: float = 0.25
    min_trade_notional: float = 100
    turnover_cap_weekly: float = 0.40
    volatility_scaling: VolatilityScaling = Field(default_factory=VolatilityScaling)


# ============================================================================
# OPTIONS OVERLAY
# ============================================================================

class HedgeOptions(BaseModel):
    """Options hedging configuration."""
    trigger_vix_pct: int = 85
    target_beta_hedge: float = 0.35
    product: str = "SPY"
    dte: List[int] = Field(default_factory=lambda: [30, 45])
    type: str = "put_spread"
    width_pct: float = 0.05
    budget_pct_of_gross: float = 0.02


class DirectionalOptions(BaseModel):
    """Directional options configuration."""
    use_on_top_conviction: int = 5
    dte: List[int] = Field(default_factory=lambda: [30, 45])
    structure: str = "debit_call_spread"
    target_deltas: List[float] = Field(default_factory=lambda: [0.35, 0.20])
    notional_pct_per_trade: float = 0.03


class OptionsOverlay(BaseModel):
    """Options overlay configuration."""
    enabled: bool = True
    hedge: HedgeOptions = Field(default_factory=HedgeOptions)
    directional: DirectionalOptions = Field(default_factory=DirectionalOptions)


# ============================================================================
# REGIME DETECTION
# ============================================================================

class VixThresholds(BaseModel):
    """VIX thresholds for regime detection."""
    growth_max: int = 20
    dividend_min: int = 30


class SignalWeights(BaseModel):
    """Weights for regime detection signals."""
    vix: float = 0.4
    momentum: float = 0.2
    macd: float = 0.2
    economic: float = 0.2


class Persistence(BaseModel):
    """Regime persistence requirements."""
    min_days: int = 5
    confidence_threshold: float = 0.75


class RegimeDetection(BaseModel):
    """Regime detection configuration."""
    vix_thresholds: VixThresholds = Field(default_factory=VixThresholds)
    signal_weights: SignalWeights = Field(default_factory=SignalWeights)
    persistence: Persistence = Field(default_factory=Persistence)


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

class PortfolioStops(BaseModel):
    """Portfolio stop loss configuration."""
    individual: float = 0.20
    portfolio: float = 0.12


class RiskManagement(BaseModel):
    """Risk management configuration."""
    max_drawdown: float = 0.15
    portfolio_stops: PortfolioStops = Field(default_factory=PortfolioStops)


# ============================================================================
# STRATEGY ALLOCATION
# ============================================================================

class GrowthRegimeAllocation(BaseModel):
    """Growth regime strategy allocation."""
    momentum_weight: float = 0.6
    value_weight: float = 0.2
    mean_reversion: float = 0.2


class DividendRegimeAllocation(BaseModel):
    """Dividend/defensive regime strategy allocation."""
    dividend_quality: float = 0.6
    value: float = 0.3
    momentum: float = 0.1


class StrategyAllocation(BaseModel):
    """Strategy allocation by regime."""
    growth_regime: GrowthRegimeAllocation = Field(default_factory=GrowthRegimeAllocation)
    dividend_regime: DividendRegimeAllocation = Field(default_factory=DividendRegimeAllocation)


# ============================================================================
# MAIN SETTINGS CLASS
# ============================================================================

class Settings(BaseModel):
    """Main settings configuration with all components."""
    # Core settings
    capital_base: float
    gross_exposure_cap: float
    target_vol_annual: float
    max_drawdown_stop: float
    rebalance_day: str
    
    # Component configurations
    paths: Paths = Field(default_factory=Paths)
    data: DataCfg = Field(default_factory=DataCfg)
    costs: Costs = Field(default_factory=Costs)
    signals: Signals = Field(default_factory=Signals)
    portfolio: Portfolio = Field(default_factory=Portfolio)
    options_overlay: OptionsOverlay = Field(default_factory=OptionsOverlay)
    regime_detection: RegimeDetection = Field(default_factory=RegimeDetection)
    risk_management: RiskManagement = Field(default_factory=RiskManagement)
    strategy_allocation: StrategyAllocation = Field(default_factory=StrategyAllocation)


# ============================================================================
# LOADER FUNCTION
# ============================================================================

def load_settings(yaml_path: str | Path) -> Settings:
    """
    Load settings from YAML file.
    
    Args:
        yaml_path: Path to settings YAML file
        
    Returns:
        Settings object with all configuration
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Settings file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return Settings(**data)