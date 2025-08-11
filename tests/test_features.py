"""
Updated CLI with regime detection capabilities
Add this to your existing src/quant_sys/cli.py
"""

from __future__ import annotations
import typer
from rich import print
from rich.table import Table
from rich.console import Console
from typing import Optional
import pandas as pd

from quant_sys.core.config import load_settings
from quant_sys.core.storage import get_engine
from quant_sys.data.ingest import ingest as ingest_fn
from quant_sys.data.transform import transform_daily_to_weekly

# Import the new regime detector
from quant_sys.analysis.regime_detector import RegimeDetector, MarketRegime

app = typer.Typer(add_completion=False, help="quant-sys CLI")
console = Console()

@app.command()
def ingest(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
    top_n: Optional[int] = typer.Option(None, help="Trim universe to top N symbols"),
    no_autofetch: bool = typer.Option(False, help="Don't fetch S&P500 from Wikipedia; use CSV only"),
):
    """
    Pull OHLCV via yfinance for the universe and store in SQLite + raw parquet.
    """
    s = load_settings(config_path)
    eng, uni = ingest_fn(s, top_n=top_n, no_autofetch=no_autofetch)
    print(f"[bold green]Ingest complete[/]: {len(uni)} symbols → {s.paths.db_path}")

@app.command("transform")
def transform_cmd(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
):
    """
    Build weekly bars from daily prices and store in SQLite + curated parquet.
    """
    s = load_settings(config_path)
    from quant_sys.core.storage import get_engine
    eng = get_engine(s.paths.db_path)
    weekly = transform_daily_to_weekly(eng, s)
    print(f"[bold cyan]Weekly bars stored[/]: {len(weekly)} rows")

@app.command("detect-regime")
def detect_regime(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
    date: Optional[str] = typer.Option(None, help="As-of date (YYYY-MM-DD), default=today"),
    history: bool = typer.Option(False, help="Show regime history"),
):
    """
    Detect current market regime using VIX, momentum, and technical indicators.
    """
    s = load_settings(config_path)
    eng = get_engine(s.paths.db_path)
    
    # Initialize regime detector
    detector = RegimeDetector(eng)
    
    # Detect current regime
    try:
        regime_score = detector.detect_current_regime(as_of_date=date)
        
        # Create summary table
        table = Table(title="Market Regime Detection", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        # Add regime information
        table.add_row("Current Regime", f"[bold yellow]{regime_score.regime.value}[/]")
        table.add_row("Confidence", f"{regime_score.confidence:.1%}")
        table.add_row("Composite Score", f"{regime_score.composite_score:+.3f}")
        
        # Add signals
        table.add_section()
        table.add_row("VIX Level", f"{regime_score.signals.vix_level:.2f}")
        table.add_row("VIX Percentile", f"{regime_score.signals.vix_percentile:.1%}")
        table.add_row("VIX Signal", f"{regime_score.signals.vix_signal:+.3f}")
        
        table.add_section()
        table.add_row("SPY Momentum (21d)", f"{regime_score.signals.spy_momentum:+.2f}%")
        table.add_row("SPY RSI", f"{regime_score.signals.spy_rsi:.1f}")
        table.add_row("MACD Histogram", f"{regime_score.signals.macd_histogram:+.3f}")
        table.add_row("Momentum Signal", f"{regime_score.signals.momentum_signal:+.3f}")
        
        # Add allocation recommendation
        allocation = detector.get_regime_allocation(regime_score.regime)
        risk_scale = detector.get_risk_scaling(regime_score.regime)
        
        table.add_section()
        table.add_row("Dividend Allocation", f"{allocation['dividend']:.0%}")
        table.add_row("Growth Allocation", f"{allocation['growth']:.0%}")
        table.add_row("Risk Scaling", f"{risk_scale:.0%}")
        
        console.print(table)
        
        # Show regime interpretation
        console.print("\n[bold]Regime Interpretation:[/]")
        interpretations = {
            MarketRegime.STRONG_GROWTH: "✅ Strong growth environment - maximize growth allocation, full risk-on",
            MarketRegime.GROWTH: "📈 Growth favorable - tilt toward growth stocks with some defensive balance", 
            MarketRegime.NEUTRAL: "⚖️ Balanced market - equal weight dividend and growth strategies",
            MarketRegime.DIVIDEND: "🛡️ Defensive needed - favor dividend stocks and quality factors",
            MarketRegime.CRISIS: "🚨 Crisis mode - maximum defense, dividend focus, reduce gross exposure"
        }
        console.print(f"  {interpretations[regime_score.regime]}")
        
        # Show history if requested
        if history:
            console.print("\n[bold]Recent Regime History:[/]")
            history_df = detector.get_regime_summary()
            if not history_df.empty:
                print(history_df.tail(10))
                
    except Exception as e:
        console.print(f"[red]Error detecting regime: {e}[/]")
        console.print("[yellow]Make sure you have VIX and SPY data. Run:[/]")
        console.print("  quant ingest --top-n 10  # This will include SPY and ^VIX")

@app.command("check-data")
def check_data(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
):
    """
    Run data quality checks and show coverage report.
    """
    s = load_settings(config_path)
    eng = get_engine(s.paths.db_path)
    
    with eng.connect() as conn:
        # Check symbols
        symbols_count = pd.read_sql("SELECT COUNT(*) as count FROM symbols", conn).iloc[0]['count']
        
        # Check daily prices
        daily_stats = pd.read_sql("""
            SELECT 
                COUNT(DISTINCT symbol) as symbols,
                COUNT(*) as total_rows,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM prices_daily
        """, conn).iloc[0]
        
        # Check weekly prices  
        weekly_stats = pd.read_sql("""
            SELECT 
                COUNT(DISTINCT symbol) as symbols,
                COUNT(*) as total_rows,
                MIN(week) as min_week,
                MAX(week) as max_week
            FROM prices_weekly
        """, conn).iloc[0]
        
        # Check for duplicates
        daily_dupes = pd.read_sql("""
            SELECT symbol, date, COUNT(*) as count
            FROM prices_daily
            GROUP BY symbol, date
            HAVING COUNT(*) > 1
        """, conn)
        
        # Check for critical symbols
        critical_symbols = ['SPY', '^VIX', 'QQQ', 'IWM']
        critical_check = pd.read_sql(f"""
            SELECT symbol, COUNT(*) as days
            FROM prices_daily
            WHERE symbol IN ({','.join([f"'{s}'" for s in critical_symbols])})
            GROUP BY symbol
        """, conn)
    
    # Create report table
    table = Table(title="Data Quality Report", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Total Symbols", str(symbols_count))
    
    table.add_section()
    table.add_row("Daily Price Records", f"{daily_stats['total_rows']:,}")
    table.add_row("Daily Symbols Coverage", f"{daily_stats['symbols']}/{symbols_count}")
    table.add_row("Daily Date Range", f"{daily_stats['min_date']} to {daily_stats['max_date']}")
    
    table.add_section()
    table.add_row("Weekly Price Records", f"{weekly_stats['total_rows']:,}")
    table.add_row("Weekly Symbols Coverage", f"{weekly_stats['symbols']}/{symbols_count}")
    table.add_row("Weekly Date Range", f"{weekly_stats['min_week']} to {weekly_stats['max_week']}")
    
    table.add_section()
    table.add_row("Duplicate Records", "❌ Found" if len(daily_dupes) > 0 else "✅ None")
    
    console.print(table)
    
    # Check critical symbols
    console.print("\n[bold]Critical Symbols Check:[/]")
    for _, row in critical_check.iterrows():
        status = "✅" if row['days'] > 100 else "❌"
        console.print(f"  {status} {row['symbol']}: {row['days']} days")
    
    # Show any missing critical symbols
    found_symbols = set(critical_check['symbol'].tolist())
    missing = set(critical_symbols) - found_symbols
    if missing:
        console.print(f"  [red]❌ Missing: {', '.join(missing)}[/]")
        console.print("  [yellow]Run 'quant ingest' to fetch missing data[/]")

if __name__ == "__main__":
    app()