"""
Momentum signal generation for trading strategies.
Location: src/quant_sys/signals/momentum_signals.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class MomentumSignal:
    """Container for momentum trading signals."""
    symbol: str
    signal_type: str  # 'long', 'short', 'neutral'
    signal_strength: float  # 0-1 scale
    hqm_score: float
    momentum_12m: float
    momentum_3m: float
    rsi: float
    percentile_rank: float
    reasons: List[str]


class MomentumSignalGenerator:
    """Generate momentum-based trading signals."""
    
    def __init__(self, engine, config):
        """
        Initialize momentum signal generator.
        
        Args:
            engine: SQLAlchemy engine
            config: Settings object with signal parameters
        """
        self.engine = engine
        self.config = config
        
        # Signal parameters from config
        self.lookback_weeks = config.signals.ts_mom.lookback_weeks
        self.skip_recent_weeks = config.signals.ts_mom.skip_recent_weeks
        self.momentum_threshold = config.signals.ts_mom.threshold
        
        # Cross-sectional parameters
        self.xs_lookback = config.signals.xs_mom.lookback_weeks
        self.num_buckets = config.signals.xs_mom.bucket
        self.long_bucket = config.signals.xs_mom.long_bucket
        self.short_bucket = config.signals.xs_mom.short_bucket
        
    def generate_signals(
        self,
        date: Optional[str] = None,
        universe: Optional[List[str]] = None,
        top_n: int = 20
    ) -> Dict[str, MomentumSignal]:
        """
        Generate momentum signals for given date and universe.
        
        Args:
            date: Date for signal generation (default: latest)
            universe: List of symbols to consider (default: all)
            top_n: Number of top/bottom stocks to signal
            
        Returns:
            Dictionary of symbol -> MomentumSignal
        """
        # Get latest features if no date specified
        if date is None:
            date = self._get_latest_date()
            
        logger.info(f"Generating momentum signals for {date}")
        
        # Fetch feature data
        features_df = self._fetch_features(date, universe)
        
        if features_df.empty:
            logger.warning("No feature data available")
            return {}
            
        # Calculate time-series momentum signals
        ts_signals = self._calculate_ts_momentum(features_df)
        
        # Calculate cross-sectional momentum signals
        xs_signals = self._calculate_xs_momentum(features_df)
        
        # Generate HQM-based signals
        hqm_signals = self._calculate_hqm_signals(features_df, top_n)
        
        # Combine all momentum signals
        combined_signals = self._combine_momentum_signals(
            features_df, ts_signals, xs_signals, hqm_signals
        )
        
        return combined_signals
    
    def _get_latest_date(self) -> str:
        """Get the latest date with feature data."""
        query = "SELECT MAX(date) as max_date FROM technical_indicators"
        result = pd.read_sql_query(query, self.engine)
        latest = result['max_date'].iloc[0]
        
        if latest is None:
            raise ValueError("No data found in technical_indicators table. Run 'quant calculate-features' first.")
            
        logger.info(f"Using latest available date: {latest}")
        return latest
    
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
            
        # Note: Using COALESCE to provide default values for NULL results
        query = f"""
        SELECT 
            symbol,
            COALESCE(MAX(CASE WHEN indicator_name = 'rsi' THEN value END), 50) as rsi,
            COALESCE(MAX(CASE WHEN indicator_name = 'bb_percent_b' THEN value END), 0) as bb_position,
            MAX(CASE WHEN indicator_name = 'bb_upper' THEN value END) as bb_upper,
            MAX(CASE WHEN indicator_name = 'bb_lower' THEN value END) as bb_lower,
            MAX(CASE WHEN indicator_name = 'bb_middle' THEN value END) as sma_20,
            MAX(CASE WHEN indicator_name = 'relative_volume' THEN value END) as volume_ratio,
            COALESCE(MAX(CASE WHEN indicator_name = 'atr_pct' THEN value END), 0.02) as atr_pct
        FROM technical_indicators
        WHERE date = '{date}'
        {where_clause}
        GROUP BY symbol
        """
        
        df = pd.read_sql_query(query, self.engine)
        
        # Calculate 12-1 momentum (skip recent month)
        if 'momentum_12m' in df.columns and 'momentum_1m' in df.columns:
            df['momentum_12_1'] = df['momentum_12m'] - df['momentum_1m']
            
        return df
    
    def _calculate_ts_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate time-series momentum signals.
        
        Uses 12-month minus 1-month momentum to avoid reversal effects.
        """
        if 'momentum_12_1' not in df.columns:
            return pd.Series(dtype=float)
            
        # Generate signals based on absolute momentum
        ts_signals = pd.Series(index=df['symbol'], dtype=float)
        
        for _, row in df.iterrows():
            mom = row['momentum_12_1']
            
            if pd.isna(mom):
                signal = 0.0
            elif mom > 0.20:  # Strong positive momentum (>20%)
                signal = 1.0
            elif mom > 0.10:  # Moderate positive momentum
                signal = 0.7
            elif mom > self.momentum_threshold:  # Weak positive
                signal = 0.3
            elif mom < -0.10:  # Strong negative momentum
                signal = -1.0
            elif mom < 0:  # Weak negative
                signal = -0.3
            else:
                signal = 0.0
                
            ts_signals[row['symbol']] = signal
            
        return ts_signals
    
    def _calculate_xs_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate cross-sectional momentum signals.
        
        Ranks stocks into quintiles and signals top/bottom quintiles.
        """
        if 'momentum_12m' not in df.columns:
            return pd.Series(dtype=float)
            
        # Rank stocks by momentum
        df['momentum_rank'] = df['momentum_12m'].rank(pct=True)
        
        # Assign to buckets (quintiles by default)
        df['bucket'] = pd.qcut(
            df['momentum_rank'],
            q=self.num_buckets,
            labels=range(1, self.num_buckets + 1)
        )
        
        # Generate signals
        xs_signals = pd.Series(index=df['symbol'], dtype=float)
        
        for _, row in df.iterrows():
            bucket = row['bucket']
            
            if bucket == self.long_bucket:  # Top quintile
                signal = 1.0
            elif bucket == self.long_bucket - 1:  # Second quintile
                signal = 0.5
            elif bucket == self.short_bucket:  # Bottom quintile
                signal = -1.0
            elif bucket == self.short_bucket + 1:  # Second worst quintile
                signal = -0.5
            else:  # Middle buckets
                signal = 0.0
                
            xs_signals[row['symbol']] = signal
            
        return xs_signals
    
    def _calculate_hqm_signals(
        self,
        df: pd.DataFrame,
        top_n: int
    ) -> pd.Series:
        """
        Generate signals based on High-Quality Momentum scores.
        
        Selects top N stocks by HQM score.
        """
        if 'hqm_score' not in df.columns:
            return pd.Series(dtype=float)
            
        # Sort by HQM score
        df_sorted = df.sort_values('hqm_score', ascending=False)
        
        # Generate signals
        hqm_signals = pd.Series(index=df['symbol'], dtype=float)
        hqm_signals[:] = 0.0  # Initialize all to neutral
        
        # Long signals for top N
        top_stocks = df_sorted.head(top_n)['symbol'].tolist()
        for symbol in top_stocks:
            hqm_signals[symbol] = 1.0
            
        # Short signals for bottom N (if momentum is negative)
        bottom_stocks = df_sorted.tail(top_n)
        for _, row in bottom_stocks.iterrows():
            if row['momentum_12m'] < -0.10:  # Only short if clearly negative
                hqm_signals[row['symbol']] = -0.5
                
        return hqm_signals
    
    def _combine_momentum_signals(
        self,
        df: pd.DataFrame,
        ts_signals: pd.Series,
        xs_signals: pd.Series,
        hqm_signals: pd.Series
    ) -> Dict[str, MomentumSignal]:
        """
        Combine different momentum signals into final signals.
        
        Weights:
        - HQM: 40%
        - Cross-sectional: 35%
        - Time-series: 25%
        """
        signals = {}
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            
            # Get individual signals
            ts_sig = ts_signals.get(symbol, 0.0)
            xs_sig = xs_signals.get(symbol, 0.0)
            hqm_sig = hqm_signals.get(symbol, 0.0)
            
            # Calculate weighted composite signal
            composite_signal = (
                0.40 * hqm_sig +
                0.35 * xs_sig +
                0.25 * ts_sig
            )
            
            # Apply RSI filter (avoid overbought/oversold extremes)
            rsi = row.get('rsi', 50)
            if rsi > 80 and composite_signal > 0:
                composite_signal *= 0.5  # Reduce long signal if overbought
            elif rsi < 20 and composite_signal < 0:
                composite_signal *= 0.5  # Reduce short signal if oversold
                
            # Determine signal type and strength
            if composite_signal > 0.5:
                signal_type = 'long'
                signal_strength = min(composite_signal, 1.0)
            elif composite_signal < -0.3:
                signal_type = 'short'
                signal_strength = min(abs(composite_signal), 1.0)
            else:
                signal_type = 'neutral'
                signal_strength = 0.0
                
            # Build reasons list
            reasons = []
            if hqm_sig > 0:
                reasons.append(f"HQM score: {row.get('hqm_score', 0):.1f}")
            if abs(ts_sig) > 0.5:
                reasons.append(f"12-1 momentum: {row.get('momentum_12_1', 0):.1%}")
            if abs(xs_sig) > 0.5:
                reasons.append(f"Top quintile performer")
            if rsi > 70:
                reasons.append(f"RSI high: {rsi:.1f}")
            elif rsi < 30:
                reasons.append(f"RSI low: {rsi:.1f}")
                
            # Create signal object
            signals[symbol] = MomentumSignal(
                symbol=symbol,
                signal_type=signal_type,
                signal_strength=signal_strength,
                hqm_score=row.get('hqm_score', 0),
                momentum_12m=row.get('momentum_12m', 0),
                momentum_3m=row.get('momentum_3m', 0),
                rsi=rsi,
                percentile_rank=row.get('momentum_rank', 0),
                reasons=reasons
            )
            
        return signals
    
    def filter_by_volatility(
        self,
        signals: Dict[str, MomentumSignal],
        max_volatility: float = 0.40
    ) -> Dict[str, MomentumSignal]:
        """
        Filter signals by volatility threshold.
        
        High volatility stocks may be excluded or have reduced signals.
        """
        # Fetch volatility data
        symbols = list(signals.keys())
        symbols_str = "','".join(symbols)
        
        query = f"""
        SELECT symbol, value as volatility
        FROM technical_indicators
        WHERE indicator_name = 'ewma_vol'
        AND symbol IN ('{symbols_str}')
        AND date = (SELECT MAX(date) FROM technical_indicators)
        """
        
        vol_df = pd.read_sql_query(query, self.engine)
        vol_dict = dict(zip(vol_df['symbol'], vol_df['volatility']))
        
        # Filter signals
        filtered_signals = {}
        for symbol, signal in signals.items():
            vol = vol_dict.get(symbol, 0.20)  # Default 20% if missing
            
            if vol > max_volatility:
                # Skip very high volatility stocks
                logger.info(f"Filtering {symbol} due to high volatility: {vol:.1%}")
                continue
            elif vol > 0.30:
                # Reduce signal strength for elevated volatility
                signal.signal_strength *= 0.7
                signal.reasons.append(f"Signal reduced - volatility: {vol:.1%}")
                
            filtered_signals[symbol] = signal
            
        return filtered_signals