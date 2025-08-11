"""
Core technical indicators for feature engineering.

This module implements fundamental technical analysis indicators used for
stock selection and signal generation.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple, Dict
import warnings


def calculate_returns(
    prices: pd.Series, 
    periods: List[int] = [1, 5, 21, 63, 126, 252]
) -> pd.DataFrame:
    """
    Calculate returns over multiple periods.
    
    Args:
        prices: Series of prices with DatetimeIndex
        periods: List of lookback periods in days
        
    Returns:
        DataFrame with returns for each period
    """
    returns_dict = {}
    
    for period in periods:
        returns_dict[f'return_{period}d'] = prices.pct_change(period)
        
    return pd.DataFrame(returns_dict, index=prices.index)


def calculate_rsi(
    prices: pd.Series, 
    period: int = 14
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    
    Args:
        prices: Series of prices
        period: Lookback period for RSI calculation
        
    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Use exponential weighted average (standard RSI calculation)
    avg_gain = gain.ewm(com=period-1, adjust=True, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, adjust=True, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_atr(
    high: pd.Series,
    low: pd.Series, 
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = EMA of True Range
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: Lookback period for ATR
        
    Returns:
        Series of ATR values
    """
    prev_close = close.shift(1)
    
    # Calculate true range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR as exponential moving average of true range
    atr = true_range.ewm(com=period-1, adjust=True, min_periods=period).mean()
    
    return atr


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Series of prices
        period: Moving average period
        std_dev: Number of standard deviations for bands
        
    Returns:
        DataFrame with columns: middle, upper, lower, %b, bandwidth
    """
    middle = prices.rolling(window=period, min_periods=period).mean()
    std = prices.rolling(window=period, min_periods=period).std()
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    # %B indicates where price is relative to the bands
    # 1 = at upper band, 0 = at lower band
    percent_b = (prices - lower) / (upper - lower)
    
    # Bandwidth indicates volatility
    bandwidth = (upper - lower) / middle
    
    return pd.DataFrame({
        'bb_middle': middle,
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_percent_b': percent_b,
        'bb_bandwidth': bandwidth
    }, index=prices.index)


def calculate_zscore(
    returns: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """
    Calculate rolling z-score for mean reversion signals.
    
    Z-score = (value - rolling_mean) / rolling_std
    
    Args:
        returns: Series of returns
        lookback: Lookback period for rolling statistics
        
    Returns:
        Series of z-scores
    """
    rolling_mean = returns.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = returns.rolling(window=lookback, min_periods=lookback).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    zscore = (returns - rolling_mean) / rolling_std
    
    return zscore


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD = 12-day EMA - 26-day EMA
    Signal = 9-day EMA of MACD
    Histogram = MACD - Signal
    
    Args:
        prices: Series of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        
    Returns:
        DataFrame with MACD, signal, and histogram
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Normalize MACD by price for comparability across stocks
    macd_normalized = macd_line / prices
    signal_normalized = signal_line / prices
    histogram_normalized = histogram / prices
    
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram,
        'macd_norm': macd_normalized,
        'macd_signal_norm': signal_normalized,
        'macd_histogram_norm': histogram_normalized
    }, index=prices.index)


def calculate_dollar_volume(
    prices: pd.Series,
    volume: pd.Series,
    period: int = 20
) -> pd.DataFrame:
    """
    Calculate dollar volume metrics for liquidity analysis.
    
    Args:
        prices: Series of prices
        volume: Series of volume
        period: Period for rolling average
        
    Returns:
        DataFrame with dollar volume metrics
    """
    dollar_volume = prices * volume
    
    # Scale to millions for readability
    dollar_volume_mm = dollar_volume / 1_000_000
    
    # Rolling average dollar volume
    avg_dollar_volume = dollar_volume_mm.rolling(
        window=period, 
        min_periods=period
    ).mean()
    
    # Relative volume (current vs average)
    relative_volume = dollar_volume_mm / avg_dollar_volume
    
    return pd.DataFrame({
        'dollar_volume_mm': dollar_volume_mm,
        'avg_dollar_volume_mm': avg_dollar_volume,
        'relative_volume': relative_volume
    }, index=prices.index)


def normalize_indicator(
    indicator: pd.Series,
    method: str = 'zscore',
    lookback: Optional[int] = None
) -> pd.Series:
    """
    Normalize an indicator for machine learning.
    
    Args:
        indicator: Series to normalize
        method: 'zscore', 'minmax', or 'percentile'
        lookback: Period for rolling normalization (None = use entire series)
        
    Returns:
        Normalized series
    """
    if lookback:
        if method == 'zscore':
            mean = indicator.rolling(lookback, min_periods=lookback).mean()
            std = indicator.rolling(lookback, min_periods=lookback).std()
            return (indicator - mean) / std.replace(0, np.nan)
        
        elif method == 'minmax':
            min_val = indicator.rolling(lookback, min_periods=lookback).min()
            max_val = indicator.rolling(lookback, min_periods=lookback).max()
            return (indicator - min_val) / (max_val - min_val).replace(0, np.nan)
        
        elif method == 'percentile':
            return indicator.rolling(lookback, min_periods=lookback).rank(pct=True)
    
    else:
        if method == 'zscore':
            return (indicator - indicator.mean()) / indicator.std()
        
        elif method == 'minmax':
            return (indicator - indicator.min()) / (indicator.max() - indicator.min())
        
        elif method == 'percentile':
            return indicator.rank(pct=True)
    
    raise ValueError(f"Unknown normalization method: {method}")


def calculate_all_technical_features(
    df: pd.DataFrame,
    periods_returns: List[int] = [1, 5, 21, 63, 126, 252],
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    macd_params: Dict = None
) -> pd.DataFrame:
    """
    Calculate all technical features for a stock.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
        periods_returns: Periods for return calculation
        rsi_period: RSI period
        atr_period: ATR period
        bb_period: Bollinger Bands period
        macd_params: MACD parameters dict
        
    Returns:
        DataFrame with all technical features
    """
    if macd_params is None:
        macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
    
    features = pd.DataFrame(index=df.index)
    
    # Returns
    returns_df = calculate_returns(df['close'], periods_returns)
    features = pd.concat([features, returns_df], axis=1)
    
    # RSI
    features['rsi'] = calculate_rsi(df['close'], rsi_period)
    
    # ATR
    features['atr'] = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    features['atr_pct'] = features['atr'] / df['close']  # ATR as % of price
    
    # Bollinger Bands
    bb_df = calculate_bollinger_bands(df['close'], bb_period)
    features = pd.concat([features, bb_df], axis=1)
    
    # MACD
    macd_df = calculate_macd(df['close'], **macd_params)
    features = pd.concat([features, macd_df], axis=1)
    
    # Dollar Volume
    if 'volume' in df.columns:
        dv_df = calculate_dollar_volume(df['close'], df['volume'])
        features = pd.concat([features, dv_df], axis=1)
    
    # Z-scores for mean reversion
    if 'return_5d' in features.columns:
        features['zscore_5d'] = calculate_zscore(features['return_5d'], lookback=20)
    
    return features

def calculate_momentum_multiperiod(
    prices: pd.Series,
    periods: List[int] = [21, 63, 126, 252]
) -> pd.DataFrame:
    """
    Calculate momentum (returns) across multiple timeframes.
    
    Args:
        prices: Series of prices with DatetimeIndex
        periods: List of lookback periods (trading days)
        
    Returns:
        DataFrame with momentum for each period
    """
    momentum_dict = {}
    
    for period in periods:
        # Calculate return over period
        momentum_dict[f'momentum_{period}d'] = (
            prices / prices.shift(period) - 1
        )
    
    return pd.DataFrame(momentum_dict, index=prices.index)

def calculate_momentum_percentiles(
    returns_df: pd.DataFrame,
    window: int = 252,
    min_observations: int = 100
) -> pd.DataFrame:
    """
    Calculate percentile ranks for each momentum metric.
    
    Args:
        returns_df: DataFrame with return columns
        window: Rolling window for percentile calculation
        min_observations: Minimum observations required
        
    Returns:
        DataFrame with percentile ranks (0-100)
    """
    percentile_df = pd.DataFrame(index=returns_df.index)
    
    for col in returns_df.columns:
        if 'momentum' in col or 'return' in col:
            # Calculate rolling percentile rank
            percentile_col = f'{col}_percentile'
            
            # Use rolling rank to get percentiles within window
            percentile_df[percentile_col] = returns_df[col].rolling(
                window=window, 
                min_periods=min_observations
            ).rank(pct=True) * 100
    
    return percentile_df

def calculate_hqm_score(
    momentum_dict: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate High-Quality Momentum score from multiple momentum metrics.
    
    Args:
        momentum_dict: Dict of momentum metric names to percentile values
        weights: Optional weights for each timeframe
        
    Returns:
        HQM score (0-100)
    """
    if weights is None:
        # Equal weight to all timeframes
        weights = {k: 1.0 / len(momentum_dict) for k in momentum_dict}
    
    # Filter out NaN values
    valid_scores = {k: v for k, v in momentum_dict.items() 
                   if pd.notna(v)}
    
    if not valid_scores:
        return np.nan
    
    # Recalculate weights for valid scores
    total_weight = sum(weights.get(k, 1.0 / len(momentum_dict)) 
                      for k in valid_scores)
    
    # Calculate weighted average
    hqm_score = sum(
        v * weights.get(k, 1.0 / len(momentum_dict)) / total_weight
        for k, v in valid_scores.items()
    )
    
    return hqm_score


def rank_by_hqm(
    universe_df: pd.DataFrame,
    momentum_columns: List[str],
    percentile_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Rank stocks by their HQM scores.
    
    Args:
        universe_df: DataFrame with stocks as rows and features as columns
        momentum_columns: List of momentum/return column names
        percentile_columns: Optional list of percentile column names
        
    Returns:
        DataFrame sorted by HQM score with added HQM columns
    """
    df = universe_df.copy()
    
    # If percentiles not provided, calculate them
    if percentile_columns is None:
        percentile_columns = []
        for col in momentum_columns:
            pct_col = f'{col}_percentile'
            # Calculate cross-sectional percentiles
            df[pct_col] = df[col].rank(pct=True) * 100
            percentile_columns.append(pct_col)
    
    # Calculate HQM score for each stock
    hqm_scores = []
    for idx in df.index:
        momentum_dict = {
            col: df.loc[idx, col] 
            for col in percentile_columns
        }
        hqm_scores.append(calculate_hqm_score(momentum_dict))
    
    df['hqm_score'] = hqm_scores
    
    # Rank by HQM score
    df['hqm_rank'] = df['hqm_score'].rank(ascending=False, method='dense')
    
    # Sort by HQM score
    df = df.sort_values('hqm_score', ascending=False)
    
    return df