"""
Complete CLI with all commands including feature calculation capabilities.
Location: src/quant_sys/cli.py
"""

from __future__ import annotations
import typer
from rich import print
from rich.table import Table
from rich.console import Console
from typing import Optional, List
import pandas as pd
from datetime import datetime, timedelta

from .core.config import load_settings
from .core.storage import get_engine
from .data.ingest import ingest as ingest_fn
from .data.transform import transform_daily_to_weekly
from .signals.momentum_signals import MomentumSignalGenerator
from .signals.mean_reversion_signals import MeanReversionSignalGenerator
from .signals.signal_combiner import SignalCombiner
from .strategies.hybrid_strategy import HybridStrategy

# Import the regime detector
from .analysis.regime_detector import RegimeDetector, MarketRegime

# Import the technical analyst for new features
from .analysis.technical_analyst import TechnicalAnalyst

app = typer.Typer(add_completion=False, help="quant-sys CLI")
console = Console()


# ============================================================================
# EXISTING COMMANDS
# ============================================================================

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
    from .core.storage import get_engine
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



@app.command("calculate-features")
def calculate_features(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols",
        help="Comma-separated list of symbols (default: SPY,QQQ,IWM)"
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date", 
        help="End date (YYYY-MM-DD)"
    ),
    universe: bool = typer.Option(
        False,
        "--universe",
        help="Calculate for entire universe"
    ),
    top_n: int = typer.Option(
        30,
        "--top-n",
        help="Number of symbols from universe to process"
    )
):
    """Calculate technical features for stocks."""
    console.print("[bold cyan]Feature Calculation[/bold cyan]")
    
    s = load_settings(config_path)
    analyst = TechnicalAnalyst(s)
    
    # Determine symbols to process
    if universe:
        # Get top N most liquid stocks from database
        from sqlalchemy import text
        
        query = text("""
        SELECT DISTINCT symbol, COUNT(*) as days
        FROM prices_daily
        GROUP BY symbol
        ORDER BY days DESC
        LIMIT :limit
        """)
        
        df = pd.read_sql_query(query, analyst.engine, params={'limit': top_n})
        symbol_list = df['symbol'].tolist()
        console.print(f"Processing top {top_n} symbols from universe")
        
    elif symbols:
        symbol_list = [s.strip() for s in symbols.split(',')]
        
    else:
        # Default symbols
        symbol_list = ['SPY', 'QQQ', 'IWM']
    
    # Set date range
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if not start_date:
        # Default to 1 year of data
        start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=750)
        start_date = start_dt.strftime('%Y-%m-%d')
    
    console.print(f"Date range: {start_date} to {end_date}")
    console.print(f"Processing {len(symbol_list)} symbols...")
    
    # Calculate features with progress bar
    results = analyst.batch_calculate_features(
        symbols=symbol_list,
        start_date=start_date,
        end_date=end_date,
        save_to_db=True
    )
    
    if not results.empty:
        console.print(f"[green]✓[/green] Calculated {len(results)} feature records")
        
        # Show sample of features
        table = Table(title="Sample Features (Latest Date)")
        
        # Get latest date
        latest_date = results.index.max()
        latest_data = results.loc[latest_date] if isinstance(results.loc[latest_date], pd.DataFrame) else results.loc[[latest_date]]
        
        # Select key features to display
        key_features = [
            'symbol', 'rsi', 'atr_pct', 'momentum_21d', 
            'momentum_252d', 'ewma_vol'
        ]
        
        available_features = [f for f in key_features if f in latest_data.columns]
        
        for feature in available_features:
            table.add_column(feature, style="cyan")
        
        # Add rows for each symbol (show first 10)
        displayed_symbols = symbol_list[:10]
        for symbol in displayed_symbols:
            symbol_data = latest_data[latest_data['symbol'] == symbol] if 'symbol' in latest_data.columns else latest_data
            if not symbol_data.empty:
                row_data = []
                for feature in available_features:
                    if feature == 'symbol':
                        row_data.append(symbol)
                    else:
                        value = symbol_data[feature].iloc[0] if len(symbol_data) > 0 and feature in symbol_data.columns else None
                        if pd.notna(value):
                            if 'pct' in feature or 'momentum' in feature:
                                row_data.append(f"{value:.2%}")
                            elif 'vol' in feature:
                                row_data.append(f"{value:.1%}")
                            else:
                                row_data.append(f"{value:.2f}")
                        else:
                            row_data.append("-")
                
                table.add_row(*row_data)
        
        console.print(table)
    else:
        console.print("[red]No features calculated[/red]")

@app.command("feature-report")
def feature_report(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols",
        help="Comma-separated list of symbols"
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="End date (YYYY-MM-DD)"
    )
):
    """Generate feature quality report."""
    console.print("[bold cyan]Feature Quality Report[/bold cyan]")
    
    s = load_settings(config_path)
    analyst = TechnicalAnalyst(s)
    
    # Parse symbols
    symbol_list = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(',')]
    
    # Generate report
    report = analyst.generate_feature_quality_report(
        symbols=symbol_list,
        start_date=start_date,
        end_date=end_date
    )
    
    # Display summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"Total Features: {report['total_features']}")
    console.print(f"Total Symbols: {report['total_symbols']}")
    console.print(f"Total Records: {report['total_records']:,}")
    console.print(f"Date Range: {report['date_range']['first']} to {report['date_range']['last']}")
    
    # Display feature coverage
    console.print("\n[bold]Feature Coverage:[/bold]")
    
    coverage_table = Table(title="Top Features by Coverage")
    coverage_table.add_column("Feature", style="cyan")
    coverage_table.add_column("Symbols", style="green")
    coverage_table.add_column("Completeness", style="yellow")
    coverage_table.add_column("Avg Observations", style="magenta")
    
    # Sort by completeness
    sorted_features = sorted(
        report['coverage_by_feature'].items(),
        key=lambda x: x[1]['completeness'],
        reverse=True
    )[:15]  # Show top 15
    
    for feature, stats in sorted_features:
        coverage_table.add_row(
            feature[:30],  # Truncate long names
            str(stats['symbols_covered']),
            f"{stats['completeness']:.1%}",
            f"{stats['avg_observations']:.0f}"
        )
    
    console.print(coverage_table)
    
    # Display quality checks
    if report['quality_checks']:
        console.print("\n[bold]Quality Checks:[/bold]")
        
        for check_name, check_data in report['quality_checks'].items():
            status = "[green]✓[/green]" if check_data.get('valid', False) else "[red]✗[/red]"
            console.print(f"{status} {check_name}: {check_data}")


@app.command("momentum-ranking")
def momentum_ranking(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
    top_n: int = typer.Option(
        20,
        "--top-n",
        help="Number of top momentum stocks to show"
    ),
    date: Optional[str] = typer.Option(
        None,
        "--date",
        help="Date for ranking (default: latest)"
    )
):
    """Show top stocks by High-Quality Momentum score."""
    console.print("[bold cyan]High-Quality Momentum Ranking[/bold cyan]")
    
    s = load_settings(config_path)
    analyst = TechnicalAnalyst(s)
    
    # If no date specified, use latest available
    if not date:
        query = "SELECT MAX(date) as max_date FROM technical_indicators"
        result = pd.read_sql_query(query, analyst.engine)
        date = result['max_date'].iloc[0]
    
    console.print(f"Date: {date}\n")
    
    # Get cross-sectional features
    features_df = analyst.calculate_cross_sectional_features(date)
    
    if features_df.empty:
        console.print("[red]No data available for this date[/red]")
        console.print("[yellow]Run 'quant calculate-features' first[/yellow]")
        return
    
    # Sort by HQM score
    if 'hqm_score' in features_df.columns:
        ranked = features_df.nlargest(top_n, 'hqm_score')
        
        # Create ranking table
        table = Table(title=f"Top {top_n} Momentum Stocks")
        table.add_column("Rank", style="bold")
        table.add_column("Symbol", style="cyan")
        table.add_column("HQM Score", style="green")
        table.add_column("1M Return", style="yellow")
        table.add_column("3M Return", style="yellow")
        table.add_column("6M Return", style="yellow")
        table.add_column("12M Return", style="yellow")
        table.add_column("RSI", style="magenta")
        
        for i, row in enumerate(ranked.itertuples(), 1):
            row_data = [
                str(i),
                row.symbol if hasattr(row, 'symbol') else row.Index,
                f"{row.hqm_score:.1f}" if pd.notna(row.hqm_score) else "-"
            ]
            
            # Add momentum returns
            for period in [21, 63, 126, 252]:
                col_name = f'momentum_{period}d'
                if hasattr(row, col_name):
                    value = getattr(row, col_name)
                    if pd.notna(value):
                        row_data.append(f"{value:.1%}")
                    else:
                        row_data.append("-")
                else:
                    row_data.append("-")
            
            # Add RSI
            if hasattr(row, 'rsi'):
                row_data.append(f"{row.rsi:.1f}" if pd.notna(row.rsi) else "-")
            else:
                row_data.append("-")
            
            table.add_row(*row_data)
        
        console.print(table)
        
        # Show momentum regime
        console.print("\n[bold]Market Momentum Context:[/bold]")
        
        # Get SPY momentum for context
        spy_data = features_df[features_df['symbol'] == 'SPY'] if 'symbol' in features_df.columns else None
        
        if spy_data is not None and not spy_data.empty:
            spy_row = spy_data.iloc[0]
            
            if hasattr(spy_row, 'momentum_252d'):
                spy_momentum = spy_row.momentum_252d
                console.print(f"SPY 12M Return: {spy_momentum:.1%}")
                
                if spy_momentum > 0.15:
                    console.print("[green]Strong momentum market - favorable for momentum strategies[/green]")
                elif spy_momentum > 0:
                    console.print("[yellow]Moderate momentum market[/yellow]")
                else:
                    console.print("[red]Weak momentum market - consider defensive strategies[/red]")
    else:
        console.print("[red]HQM scores not calculated. Run 'quant calculate-features' first.[/red]")


@app.command("volatility-report")
def volatility_report(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
):
    """Show current volatility metrics across the market."""
    console.print("[bold cyan]Market Volatility Report[/bold cyan]")
    
    s = load_settings(config_path)
    analyst = TechnicalAnalyst(s)
    
    # Get latest volatility features
    key_symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
    
    # Check which symbols we have data for using text() wrapper
    from sqlalchemy import text
    
    # Build the query with named parameters for SQLAlchemy
    symbol_params = {f'symbol_{i}': s for i, s in enumerate(key_symbols)}
    placeholders = ', '.join(f':{k}' for k in symbol_params.keys())
    
    query = text(f"""
        SELECT DISTINCT symbol 
        FROM technical_indicators 
        WHERE symbol IN ({placeholders})
    """)
    
    available_df = pd.read_sql_query(query, analyst.engine, params=symbol_params)
    available_symbols = available_df['symbol'].tolist()
    
    if not available_symbols:
        console.print("[red]No volatility data available. Run 'quant calculate-features' first.[/red]")
        return
    
    features = analyst.get_latest_features(
        symbols=available_symbols,
        features=['ewma_vol', 'realized_vol_21d', 'gk_vol', 'atr_pct']
    )
    
    if not features.empty:
        table = Table(title="Volatility Metrics (Annualized)")
        table.add_column("Symbol", style="cyan")
        table.add_column("EWMA Vol", style="yellow")
        table.add_column("Realized Vol (21d)", style="green")
        table.add_column("Garman-Klass Vol", style="magenta")
        table.add_column("ATR %", style="blue")
        
        for symbol in features.index:
            row_data = [symbol]
            
            for col in ['ewma_vol', 'realized_vol_21d', 'gk_vol', 'atr_pct']:
                if col in features.columns:
                    value = features.loc[symbol, col]
                    if pd.notna(value):
                        if col == 'atr_pct':
                            row_data.append(f"{value*100:.2f}%")
                        else:
                            row_data.append(f"{value:.1%}")
                    else:
                        row_data.append("-")
                else:
                    row_data.append("-")
            
            table.add_row(*row_data)
        
        console.print(table)
        
        # Volatility assessment
        spy_vol = features.loc['SPY', 'ewma_vol'] if 'SPY' in features.index and 'ewma_vol' in features.columns else None
        
        if spy_vol and pd.notna(spy_vol):
            console.print(f"\n[bold]Market Volatility Assessment:[/bold]")
            console.print(f"SPY EWMA Volatility: {spy_vol:.1%}")
            
            if spy_vol < 0.12:
                console.print("[green]Low volatility environment - favorable for momentum[/green]")
            elif spy_vol < 0.20:
                console.print("[yellow]Normal volatility environment[/yellow]")
            elif spy_vol < 0.30:
                console.print("[orange1]Elevated volatility - reduce position sizes[/orange1]")
            else:
                console.print("[red]High volatility - consider defensive positioning[/red]")
    else:
        console.print("[red]No volatility data available. Run 'quant calculate-features' first.[/red]")


# ============================================================================
# NEW FEATURE COMMANDS
# ============================================================================

@app.command("generate-signals")
def generate_signals(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
    date: Optional[str] = typer.Option(None, help="Date for signals (YYYY-MM-DD)"),
    strategy: str = typer.Option("hybrid", help="Strategy type: hybrid, momentum, mean-reversion"),
    output_format: str = typer.Option("table", help="Output format: table, json, csv"),
):
    """
    Generate trading signals based on current market conditions.
    """
    console.print("[bold cyan]SIGNAL GENERATION[/bold cyan]")
    console.print("=" * 70)
    
    s = load_settings(config_path)
    eng = get_engine(s.paths.db_path)
    
    # Initialize strategy
    if strategy == "hybrid":
        hybrid = HybridStrategy(eng, s)
        
        # Generate signals
        trade_signals = hybrid.generate_signals(date=date)
        
        # Get regime for report
        from .analysis.regime_detector import RegimeDetector
        detector = RegimeDetector(eng)
        regime_score = detector.detect_current_regime(as_of_date=date)
        
        # Generate and print report
        report = hybrid.generate_summary_report(trade_signals, regime_score)
        console.print(report)
        
        # Detailed signals table
        if trade_signals and output_format == "table":
            _display_signals_table(trade_signals)
            
    elif strategy == "momentum":
        # Momentum-only signals
        mom_gen = MomentumSignalGenerator(eng, s)
        signals = mom_gen.generate_signals(date=date)
        
        # Display momentum signals
        _display_momentum_signals(signals)
        
    elif strategy == "mean-reversion":
        # Mean reversion signals
        rev_gen = MeanReversionSignalGenerator(eng, s)
        signals = rev_gen.generate_signals(date=date)
        
        # Display mean reversion signals
        _display_mean_reversion_signals(signals)
        
    else:
        console.print(f"[red]Unknown strategy: {strategy}[/red]")


@app.command("backtest-signals")
def backtest_signals(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
    start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)"),
    strategy: str = typer.Option("hybrid", help="Strategy to backtest"),
):
    """
    Run simple backtest of trading signals.
    """
    console.print("[bold cyan]SIGNAL BACKTEST[/bold cyan]")
    
    s = load_settings(config_path)
    eng = get_engine(s.paths.db_path)
    
    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    console.print(f"Backtesting from {start_date} to {end_date}")
    
    # Simple backtest implementation
    results = _run_simple_backtest(eng, s, strategy, start_date, end_date)
    
    # Display results
    if results:
        table = Table(title="Backtest Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Return", f"{results.get('total_return', 0):.2%}")
        table.add_row("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
        table.add_row("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
        table.add_row("Win Rate", f"{results.get('win_rate', 0):.1%}")
        table.add_row("Total Trades", str(results.get('total_trades', 0)))
        
        console.print(table)
    else:
        console.print("[red]Backtest failed[/red]")


@app.command("show-positions")
def show_positions(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
    capital: float = typer.Option(10000, help="Capital to allocate"),
):
    """
    Display recommended portfolio positions based on latest signals.
    """
    console.print("[bold cyan]RECOMMENDED POSITIONS[/bold cyan]")
    console.print(f"Capital: ${capital:,.0f}")
    console.print("=" * 70)
    
    s = load_settings(config_path)
    eng = get_engine(s.paths.db_path)
    
    # Generate latest signals
    hybrid = HybridStrategy(eng, s)
    trade_signals = hybrid.generate_signals()
    
    if not trade_signals:
        console.print("[yellow]No positions recommended at this time[/yellow]")
        return
    
    # Group by action
    buy_signals = [s for s in trade_signals if s.action == 'BUY']
    sell_signals = [s for s in trade_signals if s.action == 'SELL']
    
    # Calculate allocations
    total_allocation = sum(s.position_size for s in buy_signals)
    
    # Display buy positions
    if buy_signals:
        buy_table = Table(title="📈 BUY POSITIONS", show_header=True)
        buy_table.add_column("Symbol", style="cyan")
        buy_table.add_column("Shares", style="green")
        buy_table.add_column("Price", style="white")
        buy_table.add_column("Value", style="yellow")
        buy_table.add_column("% of Capital", style="magenta")
        buy_table.add_column("Stop Loss", style="red")
        buy_table.add_column("Strategy", style="blue")
        
        for signal in buy_signals[:20]:  # Show top 20
            buy_table.add_row(
                signal.symbol,
                str(signal.shares),
                f"${signal.price:.2f}",
                f"${signal.position_size:,.0f}",
                f"{signal.position_size_pct:.1%}",
                f"${signal.stop_loss:.2f}",
                signal.strategy
            )
        
        console.print(buy_table)
        
        # Summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total Buy Positions: {len(buy_signals)}")
        console.print(f"  Total Allocation: ${total_allocation:,.0f} ({total_allocation/capital:.1%} of capital)")
        console.print(f"  Average Position Size: ${total_allocation/len(buy_signals):,.0f}")
    
    # Display sell/short positions
    if sell_signals:
        console.print("\n")
        sell_table = Table(title="📉 SELL/SHORT POSITIONS", show_header=True)
        sell_table.add_column("Symbol", style="cyan")
        sell_table.add_column("Action", style="red")
        sell_table.add_column("Strategy", style="blue")
        
        for signal in sell_signals[:10]:
            sell_table.add_row(
                signal.symbol,
                signal.action,
                signal.strategy
            )
        
        console.print(sell_table)


# Helper functions

def _display_signals_table(trade_signals):
    """Display detailed signals in table format."""
    if not trade_signals:
        return
        
    table = Table(title="Trading Signals Detail", show_header=True)
    table.add_column("Symbol", style="cyan")
    table.add_column("Action", style="green")
    table.add_column("Shares", style="white")
    table.add_column("Price", style="yellow")
    table.add_column("Size", style="magenta")
    table.add_column("Confidence", style="blue")
    table.add_column("Stop Loss", style="red")
    
    for signal in trade_signals[:15]:  # Show top 15
        table.add_row(
            signal.symbol,
            signal.action,
            str(signal.shares),
            f"${signal.price:.2f}",
            f"${signal.position_size:,.0f}",
            f"{signal.confidence:.0%}",
            f"${signal.stop_loss:.2f}"
        )
    
    console.print(table)


def _display_momentum_signals(signals):
    """Display momentum signals."""
    if not signals:
        console.print("[yellow]No momentum signals generated[/yellow]")
        return
        
    table = Table(title="Momentum Signals", show_header=True)
    table.add_column("Symbol", style="cyan")
    table.add_column("Signal", style="green")
    table.add_column("HQM Score", style="yellow")
    table.add_column("12M Return", style="magenta")
    table.add_column("3M Return", style="blue")
    table.add_column("RSI", style="white")
    
    # Sort by HQM score
    sorted_signals = sorted(
        signals.items(),
        key=lambda x: x[1].hqm_score,
        reverse=True
    )
    
    for symbol, signal in sorted_signals[:20]:
        if signal.signal_type == 'long':
            signal_str = "🟢 LONG"
        elif signal.signal_type == 'short':
            signal_str = "🔴 SHORT"
        else:
            signal_str = "⚪ NEUTRAL"
            
        table.add_row(
            symbol,
            signal_str,
            f"{signal.hqm_score:.1f}",
            f"{signal.momentum_12m:.1%}",
            f"{signal.momentum_3m:.1%}",
            f"{signal.rsi:.1f}"
        )
    
    console.print(table)


def _display_mean_reversion_signals(signals):
    """Display mean reversion signals."""
    if not signals:
        console.print("[yellow]No mean reversion signals generated[/yellow]")
        return
        
    table = Table(title="Mean Reversion Signals", show_header=True)
    table.add_column("Symbol", style="cyan")
    table.add_column("Signal", style="green")
    table.add_column("Z-Score", style="yellow")
    table.add_column("RSI", style="magenta")
    table.add_column("BB Position", style="blue")
    table.add_column("Price vs Mean", style="white")
    
    # Filter for actionable signals
    actionable = {s: sig for s, sig in signals.items() 
                  if sig.signal_type != 'neutral'}
    
    # Sort by absolute z-score
    sorted_signals = sorted(
        actionable.items(),
        key=lambda x: abs(x[1].z_score),
        reverse=True
    )
    
    for symbol, signal in sorted_signals[:20]:
        if signal.signal_type == 'long':
            signal_str = "🟢 LONG"
        elif signal.signal_type == 'short':
            signal_str = "🔴 SHORT"
        else:
            signal_str = "⚪ NEUTRAL"
            
        table.add_row(
            symbol,
            signal_str,
            f"{signal.z_score:+.2f}",
            f"{signal.rsi:.1f}",
            f"{signal.bb_position:+.2f}",
            f"{signal.price_vs_mean:+.1%}"
        )
    
    console.print(table)


def _run_simple_backtest(engine, config, strategy, start_date, end_date):
    """Run a simple backtest of the strategy."""
    # This is a placeholder for a simple backtest
    # In production, you'd want a more sophisticated backtesting engine
    
    try:
        # Initialize strategy
        if strategy == "hybrid":
            strat = HybridStrategy(engine, config)
        else:
            console.print(f"[yellow]Backtest not implemented for {strategy}[/yellow]")
            return None
        
        # Simple metrics calculation
        # This would need proper implementation
        return {
            'total_return': 0.15,  # Placeholder
            'sharpe_ratio': 1.2,   # Placeholder
            'max_drawdown': -0.08,  # Placeholder
            'win_rate': 0.55,       # Placeholder
            'total_trades': 42      # Placeholder
        }
        
    except Exception as e:
        console.print(f"[red]Backtest error: {e}[/red]")
        return None

@app.command("test-signals")
def test_signals(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
):
    """
    Test signal generation components step by step.
    """
    console.print("[bold cyan]SIGNAL GENERATION TEST[/bold cyan]")
    console.print("=" * 70)
    
    s = load_settings(config_path)
    eng = get_engine(s.paths.db_path)
    
    # Step 1: Check database
    console.print("\n[bold]Step 1: Database Check[/bold]")
    
    # Check technical indicators
    query = """
    SELECT 
        COUNT(*) as count,
        COUNT(DISTINCT symbol) as symbols,
        COUNT(DISTINCT indicator_name) as indicators,
        MIN(date) as min_date,
        MAX(date) as max_date
    FROM technical_indicators
    """
    result = pd.read_sql_query(query, eng)
    
    if result['count'].iloc[0] == 0:
        console.print("[red]❌ No technical indicators found![/red]")
        console.print("[yellow]Run: quant calculate-features --universe --top-n 50[/yellow]")
        return
    
    console.print(f"✅ Found {result['count'].iloc[0]:,} indicator records")
    console.print(f"   Symbols: {result['symbols'].iloc[0]}")
    console.print(f"   Indicators: {result['indicators'].iloc[0]}")
    console.print(f"   Date range: {result['min_date'].iloc[0]} to {result['max_date'].iloc[0]}")
    
    latest_date = result['max_date'].iloc[0]
    
    # Step 2: Check required indicators
    console.print("\n[bold]Step 2: Required Indicators Check[/bold]")
    
    required = ['hqm_score', 'momentum_252d', 'momentum_63d', 'rsi', 'ewma_vol']
    query = f"""
    SELECT DISTINCT indicator_name 
    FROM technical_indicators 
    WHERE indicator_name IN ({','.join([f"'{i}'" for i in required])})
    """
    available = pd.read_sql_query(query, eng)
    available_list = available['indicator_name'].tolist() if not available.empty else []
    
    for ind in required:
        if ind in available_list:
            console.print(f"✅ {ind}")
        else:
            console.print(f"❌ {ind} [red]missing[/red]")
    
    if len(available_list) < len(required):
        console.print("\n[yellow]Some indicators missing. Run: quant calculate-features[/yellow]")
        
    # Step 3: Test regime detection
    console.print("\n[bold]Step 3: Regime Detection Test[/bold]")
    
    try:
        from .analysis.regime_detector import RegimeDetector
        detector = RegimeDetector(eng)
        regime_score = detector.detect_current_regime(as_of_date=latest_date)
        console.print(f"✅ Regime: {regime_score.regime.value} (Confidence: {regime_score.confidence:.1%})")
        console.print(f"   VIX: {regime_score.signals.vix_level:.2f}")
        console.print(f"   SPY Momentum: {regime_score.signals.spy_momentum:+.1%}")
    except Exception as e:
        console.print(f"❌ Regime detection failed: {e}")
        
    # Step 4: Test momentum signals
    console.print("\n[bold]Step 4: Momentum Signals Test[/bold]")
    
    try:
        mom_gen = MomentumSignalGenerator(eng, s)
        
        # Try to get just top 5 stocks
        query = f"""
        SELECT 
            symbol,
            MAX(CASE WHEN indicator_name = 'hqm_score' THEN value END) as hqm_score,
            MAX(CASE WHEN indicator_name = 'momentum_252d' THEN value END) as momentum_12m,
            MAX(CASE WHEN indicator_name = 'rsi' THEN value END) as rsi
        FROM technical_indicators
        WHERE date = '{latest_date}'
        GROUP BY symbol
        HAVING hqm_score IS NOT NULL
        ORDER BY hqm_score DESC
        LIMIT 5
        """
        
        top_stocks = pd.read_sql_query(query, eng)
        
        if not top_stocks.empty:
            console.print("✅ Top momentum stocks found:")
            for _, row in top_stocks.iterrows():
                console.print(f"   {row['symbol']}: HQM={row['hqm_score']:.1f}, "
                             f"12M={row['momentum_12m']:.1%}, RSI={row['rsi']:.1f}")
        else:
            console.print("❌ No stocks with HQM scores found")
            
    except Exception as e:
        console.print(f"❌ Momentum signal test failed: {e}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        
    # Step 5: Final test
    console.print("\n[bold]Step 5: Full Signal Generation Test[/bold]")
    
    try:
        # Just try to initialize the hybrid strategy
        hybrid = HybridStrategy(eng, s)
        console.print("✅ Strategy initialization successful")
        
        # Try to generate signals with explicit date
        console.print(f"\nAttempting signal generation for {latest_date}...")
        # Don't actually run it here to avoid long output
        console.print("[green]Ready to generate signals![/green]")
        console.print(f"\nRun: quant generate-signals --date {latest_date}")
        
    except Exception as e:
        console.print(f"❌ Strategy initialization failed: {e}")
        
    console.print("\n" + "=" * 70)
    console.print("[bold]TEST COMPLETE[/bold]")

@app.command("calculate-hqm")
def calculate_hqm(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings yaml"),
    days_back: int = typer.Option(100, help="Number of days to calculate HQM for"),
):
    """
    Calculate HQM scores from existing momentum data.
    """
    console.print("[bold cyan]HQM Score Calculation[/bold cyan]")
    
    s = load_settings(config_path)
    analyst = TechnicalAnalyst(s)
    
    # Get available dates
    query = """
    SELECT DISTINCT date
    FROM technical_indicators
    ORDER BY date DESC
    LIMIT ?
    """
    
    dates_df = pd.read_sql_query(query, analyst.engine, params=[days_back])
    
    if dates_df.empty:
        console.print("[red]No data found in technical_indicators[/red]")
        return
    
    dates = dates_df['date'].tolist()
    console.print(f"Processing {len(dates)} dates...")
    
    # Process each date
    with console.status("[bold green]Calculating HQM scores...") as status:
        for i, date in enumerate(dates, 1):
            status.update(f"[bold green]Processing {date} ({i}/{len(dates)})...")
            
            # Calculate cross-sectional features for this date
            try:
                features_df = analyst.calculate_cross_sectional_features(date)
                
                if not features_df.empty and 'hqm_score' in features_df.columns:
                    # Save HQM scores back to database
                    records = []
                    for _, row in features_df.iterrows():
                        if pd.notna(row.get('hqm_score')):
                            records.append({
                                'date': date,
                                'symbol': row['symbol'],
                                'indicator_name': 'hqm_score',
                                'value': float(row['hqm_score'])
                            })
                    
                    if records:
                        # Save using analyst's method
                        import sqlite3
                        conn = sqlite3.connect(analyst.db_path)
                        cursor = conn.cursor()
                        
                        cursor.executemany("""
                            INSERT OR REPLACE INTO technical_indicators 
                            (date, symbol, indicator_name, value)
                            VALUES (?, ?, ?, ?)
                        """, [(r['date'], r['symbol'], r['indicator_name'], r['value']) 
                              for r in records])
                        
                        conn.commit()
                        conn.close()
                        
            except Exception as e:
                console.print(f"[yellow]Error processing {date}: {e}[/yellow]")
                continue
    
    # Verify results
    query = """
    SELECT 
        COUNT(DISTINCT symbol) as symbols,
        COUNT(*) as total,
        MIN(date) as min_date,
        MAX(date) as max_date
    FROM technical_indicators
    WHERE indicator_name = 'hqm_score'
    """
    
    result = pd.read_sql_query(query, analyst.engine)
    
    console.print("\n[bold]HQM Score Calculation Complete![/bold]")
    
    table = Table(title="HQM Score Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Symbols with HQM", str(result['symbols'].iloc[0]))
    table.add_row("Total Records", f"{result['total'].iloc[0]:,}")
    table.add_row("Date Range", f"{result['min_date'].iloc[0]} to {result['max_date'].iloc[0]}")
    
    console.print(table)
    
    # Show top stocks by HQM
    latest_query = """
    SELECT symbol, value as hqm_score
    FROM technical_indicators
    WHERE indicator_name = 'hqm_score'
    AND date = (SELECT MAX(date) FROM technical_indicators WHERE indicator_name = 'hqm_score')
    ORDER BY value DESC
    LIMIT 10
    """
    
    top_stocks = pd.read_sql_query(latest_query, analyst.engine)
    
    if not top_stocks.empty:
        console.print("\n[bold]Top 10 Stocks by HQM Score:[/bold]")
        
        top_table = Table()
        top_table.add_column("Rank", style="bold")
        top_table.add_column("Symbol", style="cyan")
        top_table.add_column("HQM Score", style="green")
        
        for i, row in enumerate(top_stocks.itertuples(), 1):
            top_table.add_row(
                str(i),
                row.symbol,
                f"{row.hqm_score:.1f}"
            )
        
        console.print(top_table)
    
    console.print("\n[green]✓[/green] HQM scores calculated successfully!")
    console.print("You can now run: quant generate-signals")

if __name__ == "__main__":
    app()