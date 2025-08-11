from __future__ import annotations
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime

def nyse_schedule(start: str, end: str) -> pd.DataFrame:
    nyse = mcal.get_calendar("XNYS")
    return nyse.schedule(start_date=start, end_date=end)

def last_trading_day(date: str | datetime) -> pd.Timestamp:
    dt = pd.Timestamp(date).tz_localize(None)
    sched = nyse_schedule(dt - pd.Timedelta(days=10), dt + pd.Timedelta(days=1))
    return mcal.date_range(sched, frequency="1D")[-1]

def weekly_label(date: str | pd.Timestamp) -> str:
    # label weeks by Monday date (for consistent indexing)
    ts = pd.Timestamp(date)
    monday = (ts - pd.Timedelta(days=int(ts.dayofweek))).normalize()
    return monday.strftime("%Y-%m-%d")
