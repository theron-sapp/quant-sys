# Quant Trading System Development Context

## Project Overview

I'm building a quantitative trading system that:

- Manages $10,000 capital with max $5,000 gross exposure
- Uses hybrid dividend/growth strategies based on market regime detection
- Targets 10-12% annualized volatility with 15% max drawdown
- Performs weekly rebalancing with optional options overlay
- Runs locally with <$1,000 total cost
- Stack: Python, SQLite, yfinance, pandas, scikit-learn, Typer CLI

## Current Status

[PASTE CURRENT PROJECT_TRACKER.md HERE]

## Module Architecture & Mapping

[PASTE MODULE MAPPING GUIDE HERE - Module to milestone mapping guide.md]

## Current File Structure

```
quant-sys/
├── config/
│   ├── settings.yaml          # Configuration file
│   └── universe_sp500.csv     # Stock universe
├── db/
│   └── quant.sqlite           # SQLite database
├── data/
│   ├── raw/                   # Daily OHLCV parquet files
│   └── curated/               # Weekly bars parquet
├── src/quant_sys/
│   ├── core/                  # ✅ COMPLETE
│   │   ├── config.py
│   │   ├── calendar.py
│   │   ├── storage.py
│   │   └── costs.py
│   ├── data/                  # ✅ COMPLETE
│   │   ├── ingest.py
│   │   └── transform.py
│   ├── analysis/              # ⚠️ IN PROGRESS
│   │   └── regime_detector.py # ✅ DONE
│   └── cli.py                 # Main CLI
└── test_regime_detection.py   # Test script
```

## Key Design Decisions

1. **Regime Detection**: VIX-based with 40% weight, momentum 20%, MACD 20%, economic 20%
2. **Risk Management**: Max DD 15% (tightened from 25%), individual position stop at 20%
3. **Strategy Allocation**: Growth regime = 60% growth/40% dividend, Crisis = 20% growth/80% dividend
4. **Position Sizing**: Max 5% per position, max 25% per sector, 30 names max
5. **Regime Persistence**: 5-day minimum, 75% confidence threshold to prevent whipsaw

## Working CLI Commands

```bash
# Data pipeline
quant ingest --top-n 50    # Fetch OHLCV data
quant transform             # Create weekly bars
quant check-data           # Verify data quality

# Analysis
quant detect-regime        # Show current market regime
quant detect-regime --date 2020-03-23  # Historical regime

# Testing
python test_regime_detection.py  # Run regime detection tests
```

## Test Results to Verify

- COVID Crash (2020-03-23): Should show Crisis regime (VIX 61.59)
- Current market: Should show Growth regime (VIX ~15)
- Data quality: Should have 5,183 days, zero duplicates

## Next Task

[UPDATE THIS SECTION BASED ON PROJECT_TRACKER.md]

Currently working on: **M3 - Advanced Features & Technical Indicators**

Specifically need to implement:

1. `features/technical.py` - Core technical indicators
2. `features/high_quality_momentum.py` - Multi-timeframe momentum
3. `features/vol_models.py` - Volatility modeling
4. `analysis/technical_analyst.py` - Feature coordination

## Configuration (settings.yaml)

```yaml
capital_base: 10000
gross_exposure_cap: 5000
target_vol_annual: 0.11
max_drawdown_stop: 0.25
rebalance_day: "MONDAY_OPEN"

paths:
  db_path: "db/quant.sqlite"
  universe_file: "config/universe_sp500.csv"
  raw_dir: "data/raw"
  curated_dir: "data/curated"

data:
  start: "2005-01-01"
  source: "yfinance"
  symbols_extra: ["SPY", "QQQ", "IWM", "^VIX"]
  universe_trim_top_liquidity: 300

costs:
  equities:
    commission_per_share: 0.005
    slippage_bps: 8

regime_detection:
  vix_thresholds:
    growth_max: 20
    dividend_min: 30
  signal_weights:
    vix: 0.4
    momentum: 0.2
    macd: 0.2
    economic: 0.2
```

## Request

Please continue development from where we left off. The PROJECT_TRACKER.md shows what's complete (✅) and what needs to be done next. Focus on the current milestone and implement the next modules according to the roadmap.

## Additional Context (if needed)

[ADD ANY SPECIFIC ISSUES, ERRORS, OR DECISIONS FROM LAST SESSION]

---

_Note: This is a continuation of an ongoing project. Please review the PROJECT_TRACKER.md for full context and pick up where the last session ended._
