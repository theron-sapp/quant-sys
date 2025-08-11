```
quant-sys/
  pyproject.toml            # deps & entry points (Typer CLI)
  README.md
  config/
    settings.yaml           # runtime config (see below)
    universe_sp500.csv      # ticker universe
  db/
    quant.sqlite            # SQLite (SQLAlchemy)
  data/
    raw/                    # yfinance dumps
    curated/                # parquet: daily/weekly
  src/
    core/
      config.py             # pydantic settings loader
      calendar.py           # trading calendar (pandas_market_calendars)
      storage.py            # SQLite engine + ORM helpers
      costs.py              # slippage/commission models
    data/
      ingest.py             # OHLCV (yfinance), ^VIX, dividends/splits
      transform.py          # daily->weekly resample, adj & QC
    features/
      technical.py          # momentum, vol, ATR, z-scores
      vol_models.py         # EWMA, (optional) GARCH from arch
    signals/
      ts_momentum.py        # time-series (12-1, 26w slope)
      xs_momentum.py        # cross-sectional rank
      mean_reversion.py     # 1w/5d z-score reversion
      risk_filters.py       # VIX regime, crash filter
      ensemble.py           # combine alphas (PSR/IC weighting)
    options/
      chains.py             # yfinance chains cache → SQLite
      iv_greeks.py          # BS greeks, IV est (py_vollib/mibian)
      overlays.py           # call/put spread construction & sizing
    portfolio/
      sizing.py             # vol targeting, position caps
      constraints.py        # gross cap, sector caps, turnover
      rebalance.py          # target→orders (netting, min lot)
    backtest/
      engine.py             # vectorized weekly engine (vectorbt / bt)
      options_sim.py        # simple spread P&L, decay, fees
      walkforward.py        # purged CV + embargo, rolling OOS
      metrics.py            # Sharpe, Sortino, DD, ES, turnover
    reporting/
      plots.py              # equity curve, exposures, factor tears
      trade_sheet.py        # weekly PDF/HTML with tickets
    cli.py                  # Typer: ingest, research, backtest, run, report
  notebooks/                # ad hoc research
  tests/
```

test
