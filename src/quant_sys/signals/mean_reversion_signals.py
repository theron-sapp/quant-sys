"""
Mean reversion signal generation for trading strategies.
Location: src/quant_sys/signals/mean_reversion_signals.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class MeanReversionSignal:
    """Container for mean reversion trading signals."""
    symbol: str
    signal_type: str  # 'long', 'short', 'neutral'
    signal_strength: float  # 0-1 scale
    z_score: float
    rsi: float
    bb_position: float  # Position within Bollinger Bands (-1 to 1)
    price_vs_mean: float  # % deviation from mean
    reasons: List[str]


class MeanReversionSignalGenerator:
    """Generate mean reversion trading signals."""
    
    def __init__(self, engine, config):
        """
        Initialize mean reversion signal generator.
        
        Args:
            engine: SQLAlchemy engine
            config: Settings object with signal parameters
        """
        self.engine = engine
        self.config = config
        
        # Mean reversion parameters from config
        self.lookback_days = config.signals.mean_rev.lookback_days
        self.z_entry_threshold = config.signals.mean_rev.z_entry
        
        # RSI thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
    def generate_signals(
        self,
        date: Optional[str] = None,
        universe: Optional[List[str]] = None,
        min_volume: float = 1000000
    ) -> Dict[str, MeanReversionSignal]:
        """
        Generate mean reversion signals.
        
        Args:
            date: Date for signal generation
            universe: List of symbols to consider
            min_volume: Minimum average dollar volume for liquidity
            
        Returns:
            Dictionary of symbol -> MeanReversionSignal
        """
        if date is None:
            date = self._get_latest_date()
            
        logger.info(f"Generating mean reversion signals for {date}")
        
        # Fetch recent price and feature data
        features_df = self._fetch_features(date, universe)
        price_df = self._fetch_price_history(date, universe)
        
        if features_df.empty or price_df.empty:
            logger.warning("Insufficient data for mean reversion signals")
            return {}
            
        # Calculate z-scores
        z_scores = self._calculate_z_scores(price_df)
        
        # Generate signals
        signals = self._generate_reversion_signals(features_df, z_scores)
        
        # Filter by liquidity
        signals = self._filter_by_liquidity(signals, min_volume)
        
        return signals
    
    def _get_latest_date(self) -> str:
        """Get the latest date with data."""
        # First try technical_indicators
        query = "SELECT MAX(date) as max_date FROM technical_indicators"
        result = pd.read_sql_query(query, self.engine)
        ti_date = result['max_date'].iloc[0]
        
        # Also check prices_daily
        query = "SELECT MAX(date) as max_date FROM prices_daily"
        result = pd.read_sql_query(query, self.engine)
        pd_date = result['max_date'].iloc[0]
        
        # Use the earlier of the two (to ensure we have both price and indicator data)
        if ti_date and pd_date:
            return min(ti_date, pd_date)
        elif pd_date:
            return pd_date
        else:
            raise ValueError("No data found. Run 'quant ingest' and 'quant calculate-features' first.")

    
    def _fetch_features(
        self,
        date: str,
        universe: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fetch technical features for mean reversion analysis."""
        
        if universe:
            symbols_str = "','".join(universe)
            where_clause = f"AND symbol IN ('{symbols_str}')"
        else:
            where_clause = ""
            
        query = f"""
        SELECT 
            symbol,
            MAX(CASE WHEN indicator_name = 'rsi' THEN value END) as rsi,
            MAX(CASE WHEN indicator_name = 'bb_percent_b' THEN value END) as bb_position,
            MAX(CASE WHEN indicator_name = 'bb_upper' THEN value END) as bb_upper,
            MAX(CASE WHEN indicator_name = 'bb_lower' THEN value END) as bb_lower,
            MAX(CASE WHEN indicator_name = 'sma_20' THEN value END) as sma_20,
            MAX(CASE WHEN indicator_name = 'volume_ratio' THEN value END) as volume_ratio,
            MAX(CASE WHEN indicator_name = 'atr_pct' THEN value END) as atr_pct
        FROM technical_indicators
        WHERE date = '{date}'
        {where_clause}
        GROUP BY symbol
        """
        
        return pd.read_sql_query(query, self.engine)
    
    def _fetch_price_history(
        self,
        date: str,
        universe: Optional[List[str]] = None,
        lookback: int = 20
    ) -> pd.DataFrame:
        """Fetch recent price history for z-score calculation."""
        
        # Calculate start date
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=lookback * 2)  # Extra buffer for trading days
        
        if universe:
            symbols_str = "','".join(universe)
            where_clause = f"AND symbol IN ('{symbols_str}')"
        else:
            where_clause = ""
            
        query = f"""
        SELECT 
            date,
            symbol,
            close,
            volume,
            close * volume as dollar_volume
        FROM prices_daily
        WHERE date BETWEEN '{start_date.strftime('%Y-%m-%d')}' 
              AND '{date}'
        {where_clause}
        ORDER BY symbol, date
        """
        
        df = pd.read_sql_query(query, self.engine)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def _calculate_z_scores(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate z-scores for each symbol.
        
        Z-score = (price - mean) / std_dev
        """
        z_scores = []
        
        for symbol in price_df['symbol'].unique():
            symbol_data = price_df[price_df['symbol'] == symbol].copy()
            
            if len(symbol_data) < self.lookback_days:
                continue
                
            # Get recent prices
            recent_prices = symbol_data.tail(self.lookback_days)['close']
            current_price = recent_prices.iloc[-1]
            
            # Calculate statistics
            mean_price = recent_prices.mean()
            std_price = recent_prices.std()
            
            if std_price > 0:
                z_score = (current_price - mean_price) / std_price
                price_vs_mean = (current_price - mean_price) / mean_price
            else:
                z_score = 0
                price_vs_mean = 0
                
            # Calculate average dollar volume
            avg_dollar_volume = symbol_data.tail(self.lookback_days)['dollar_volume'].mean()
            
            z_scores.append({
                'symbol': symbol,
                'z_score': z_score,
                'price_vs_mean': price_vs_mean,
                'current_price': current_price,
                'mean_price': mean_price,
                'std_price': std_price,
                'avg_dollar_volume': avg_dollar_volume
            })
            
        return pd.DataFrame(z_scores)
    
    def _generate_reversion_signals(
        self,
        features_df: pd.DataFrame,
        z_scores_df: pd.DataFrame
    ) -> Dict[str, MeanReversionSignal]:
        """Generate mean reversion signals based on z-scores and technical indicators."""
        
        # Merge dataframes
        df = pd.merge(
            features_df,
            z_scores_df,
            on='symbol',
            how='inner'
        )
        
        signals = {}
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            
            # Get values with safe defaults
            z_score = row.get('z_score', 0)
            rsi = row.get('rsi', 50)
            bb_position = row.get('bb_position', 0)
            price_vs_mean = row.get('price_vs_mean', 0)
            
            # Handle None/NaN values - convert to appropriate defaults
            if pd.isna(z_score) or z_score is None:
                z_score = 0.0
            if pd.isna(rsi) or rsi is None:
                rsi = 50.0
            if pd.isna(bb_position) or bb_position is None:
                bb_position = 0.0
            if pd.isna(price_vs_mean) or price_vs_mean is None:
                price_vs_mean = 0.0
                
            # Ensure all values are floats
            z_score = float(z_score)
            rsi = float(rsi)
            bb_position = float(bb_position)
            price_vs_mean = float(price_vs_mean)
            
            # Initialize signal variables
            signal_type = 'neutral'
            signal_strength = 0.0
            reasons = []
            
            # Strong mean reversion buy signal
            if z_score < -self.z_entry_threshold:
                signal_type = 'long'
                signal_strength = min(abs(z_score) / 3, 1.0)
                reasons.append(f"Z-score: {z_score:.2f} (oversold)")
                
                # Confirm with RSI
                if rsi < self.rsi_oversold:
                    signal_strength = min(signal_strength * 1.2, 1.0)
                    reasons.append(f"RSI oversold: {rsi:.1f}")
                    
                # Confirm with Bollinger Bands
                if bb_position < -0.8:
                    signal_strength = min(signal_strength * 1.1, 1.0)
                    reasons.append("Below lower Bollinger Band")
                    
            # Strong mean reversion sell signal
            elif z_score > self.z_entry_threshold:
                signal_type = 'short'
                signal_strength = min(z_score / 3, 1.0)
                reasons.append(f"Z-score: {z_score:.2f} (overbought)")
                
                # Confirm with RSI
                if rsi > self.rsi_overbought:
                    signal_strength = min(signal_strength * 1.2, 1.0)
                    reasons.append(f"RSI overbought: {rsi:.1f}")
                    
                # Confirm with Bollinger Bands
                if bb_position > 0.8:
                    signal_strength = min(signal_strength * 1.1, 1.0)
                    reasons.append("Above upper Bollinger Band")
                    
            # Moderate signals based on RSI extremes
            elif rsi < 35 and z_score < -0.5:
                signal_type = 'long'
                signal_strength = 0.5
                reasons.append(f"RSI moderately oversold: {rsi:.1f}")
                
            elif rsi > 65 and z_score > 0.5:
                signal_type = 'short'
                signal_strength = 0.5
                reasons.append(f"RSI moderately overbought: {rsi:.1f}")
                
            # Add price deviation information
            if abs(price_vs_mean) > 0.05:
                reasons.append(f"Price vs mean: {price_vs_mean:.1%}")
                
            # Create signal object
            signals[symbol] = MeanReversionSignal(
                symbol=symbol,
                signal_type=signal_type,
                signal_strength=signal_strength,
                z_score=z_score,
                rsi=rsi,
                bb_position=bb_position,
                price_vs_mean=price_vs_mean,
                reasons=reasons
            )
            
        return signals
    
    def _filter_by_liquidity(
        self,
        signals: Dict[str, MeanReversionSignal],
        min_volume: float
    ) -> Dict[str, MeanReversionSignal]:
        """
        Filter signals by minimum liquidity requirements.
        
        Mean reversion requires sufficient liquidity for entry/exit.
        """
        # Get volume data for signal symbols
        symbols = list(signals.keys())
        if not symbols:
            return signals
            
        symbols_str = "','".join(symbols)
        
        query = f"""
        SELECT 
            symbol,
            AVG(close * volume) as avg_dollar_volume
        FROM prices_daily
        WHERE symbol IN ('{symbols_str}')
        AND date >= date('now', '-20 days')
        GROUP BY symbol
        """
        
        vol_df = pd.read_sql_query(query, self.engine)
        
        # Filter signals
        filtered_signals = {}
        for symbol, signal in signals.items():
            vol_data = vol_df[vol_df['symbol'] == symbol]
            
            if not vol_data.empty:
                avg_volume = vol_data['avg_dollar_volume'].iloc[0]
                
                if avg_volume >= min_volume:
                    filtered_signals[symbol] = signal
                else:
                    logger.info(f"Filtering {symbol} due to low liquidity: ${avg_volume:,.0f}")
            else:
                # Keep signal if we can't verify volume (conservative)
                filtered_signals[symbol] = signal
                
        return filtered_signals
    
    def combine_with_trend(
        self,
        signals: Dict[str, MeanReversionSignal],
        trend_window: int = 50
    ) -> Dict[str, MeanReversionSignal]:
        """
        Adjust mean reversion signals based on trend.
        
        Mean reversion works better in ranging markets,
        less effective in strong trends.
        """
        symbols = list(signals.keys())
        if not symbols:
            return signals
            
        symbols_str = "','".join(symbols)
        
        # Get trend data (50-day SMA vs 200-day SMA)
        query = f"""
        SELECT 
            symbol,
            MAX(CASE WHEN indicator_name = 'sma_50' THEN value END) as sma_50,
            MAX(CASE WHEN indicator_name = 'sma_200' THEN value END) as sma_200
        FROM technical_indicators
        WHERE symbol IN ('{symbols_str}')
        AND date = (SELECT MAX(date) FROM technical_indicators)
        GROUP BY symbol
        """
        
        trend_df = pd.read_sql_query(query, self.engine)
        
        # Adjust signals based on trend
        for _, row in trend_df.iterrows():
            symbol = row['symbol']
            if symbol not in signals:
                continue
                
            signal = signals[symbol]
            sma_50 = row.get('sma_50')
            sma_200 = row.get('sma_200')
            
            if sma_50 and sma_200:
                trend_strength = (sma_50 - sma_200) / sma_200
                
                # Strong uptrend - reduce short signals
                if trend_strength > 0.05 and signal.signal_type == 'short':
                    signal.signal_strength *= 0.5
                    signal.reasons.append("Signal reduced - strong uptrend")
                    
                # Strong downtrend - reduce long signals
                elif trend_strength < -0.05 and signal.signal_type == 'long':
                    signal.signal_strength *= 0.5
                    signal.reasons.append("Signal reduced - strong downtrend")
                    
                # Ranging market - increase signal strength
                elif abs(trend_strength) < 0.02:
                    signal.signal_strength = min(signal.signal_strength * 1.2, 1.0)
                    signal.reasons.append("Ranging market - favorable for mean reversion")
                    
        return signals