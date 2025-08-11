"""
Volatility modeling for risk management and position sizing.

Implements various volatility estimators including EWMA, Garman-Klass,
and GARCH models for volatility forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import warnings


def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate simple realized (historical) volatility.
    
    Args:
        returns: Series of returns
        window: Rolling window period
        annualize: Whether to annualize volatility
        
    Returns:
        Series of volatility values
    """
    vol = returns.rolling(window=window, min_periods=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def calculate_ewma_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) volatility.
    
    EWMA gives more weight to recent observations, making it more responsive
    to volatility changes. Lambda=0.94 is the RiskMetrics standard.
    
    Args:
        returns: Series of returns
        lambda_: Decay factor (0.94 is RiskMetrics standard)
        annualize: Whether to annualize volatility
        
    Returns:
        Series of EWMA volatility
    """
    # Calculate squared returns
    returns_squared = returns ** 2
    
    # Initialize with first observation or expanding window variance
    ewma_var = returns_squared.ewm(
        alpha=1-lambda_, 
        adjust=False
    ).mean()
    
    # Convert variance to volatility
    ewma_vol = np.sqrt(ewma_var)
    
    if annualize:
        ewma_vol = ewma_vol * np.sqrt(252)
    
    return ewma_vol


def calculate_garman_klass_volatility(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate Garman-Klass volatility estimator.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        open_: Series of open prices
        window: Rolling window for averaging
        annualize: Whether to annualize
        
    Returns:
        Series of GK volatility estimates
    """
    # Ensure no zero or negative prices
    high = high.replace(0, np.nan)
    low = low.replace(0, np.nan)
    close = close.replace(0, np.nan)
    open_ = open_.replace(0, np.nan)
    
    # Calculate components with proper handling
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    
    # Garman-Klass formula - note the correction here
    # The daily variance is the sum of squared log returns
    gk_daily_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    
    # Take the square root of the average variance to get volatility
    # Use rolling mean of variance, then take square root
    gk_var = gk_daily_var.rolling(window=window, min_periods=window).mean()
    
    # Handle negative variances (shouldn't happen but just in case)
    gk_var = gk_var.clip(lower=0)
    
    # Convert to volatility
    gk_vol = np.sqrt(gk_var)
    
    if annualize:
        gk_vol = gk_vol * np.sqrt(252)
    
    # Cap at reasonable maximum (e.g., 200% annualized volatility)
    gk_vol = gk_vol.clip(upper=2.0)
    
    return gk_vol

def calculate_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate Parkinson volatility estimator.
    
    Parkinson uses only high and low prices, making it simpler than
    Garman-Klass but still more efficient than close-to-close volatility.
    
    Formula:
    Parkinson = sqrt((log(H/L))^2 / (4 * log(2)))
    
    Args:
        high: Series of high prices
        low: Series of low prices
        window: Rolling window for averaging
        annualize: Whether to annualize
        
    Returns:
        Series of Parkinson volatility estimates
    """
    # Ensure no zero or negative prices
    high = high.replace(0, np.nan)
    low = low.replace(0, np.nan)
    
    # Calculate daily Parkinson volatility
    log_hl = np.log(high / low)
    parkinson_daily = log_hl / (2 * np.sqrt(np.log(2)))
    
    # Rolling average
    parkinson_vol = parkinson_daily.rolling(
        window=window, 
        min_periods=window
    ).mean()
    
    if annualize:
        parkinson_vol = parkinson_vol * np.sqrt(252)
    
    return parkinson_vol


def calculate_yang_zhang_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate Yang-Zhang volatility estimator.
    
    Yang-Zhang is considered one of the most accurate volatility estimators,
    combining overnight and intraday volatility with drift correction.
    
    Args:
        open_: Series of open prices
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        window: Rolling window
        annualize: Whether to annualize
        
    Returns:
        Series of Yang-Zhang volatility estimates
    """
    # Overnight volatility (close to open)
    log_co = np.log(open_ / close.shift(1))
    overnight_var = log_co.rolling(window=window).var()
    
    # Open-to-close volatility
    log_oc = np.log(close / open_)
    open_close_var = log_oc.rolling(window=window).var()
    
    # Rogers-Satchell volatility (high-low-close)
    log_hc = np.log(high / close)
    log_lc = np.log(low / close)
    rs_var = (log_hc * np.log(high / open_) + 
              log_lc * np.log(low / open_)).rolling(window=window).mean()
    
    # Yang-Zhang combination with optimal weights
    k = 0.34 / (1 + (window + 1) / (window - 1))
    yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
    
    yz_vol = np.sqrt(yz_var)
    
    if annualize:
        yz_vol = yz_vol * np.sqrt(252)
    
    return yz_vol


def volatility_regime_detection(
    volatility: pd.Series,
    lookback: int = 252,
    thresholds: Tuple[float, float] = (25, 75)
) -> pd.Series:
    """
    Classify volatility into low, medium, or high regimes.
    
    Args:
        volatility: Series of volatility values
        lookback: Period for percentile calculation
        thresholds: (low, high) percentile thresholds
        
    Returns:
        Series with regime labels
    """
    # Calculate rolling percentiles
    vol_percentile = volatility.rolling(
        window=lookback,
        min_periods=lookback // 2
    ).rank(pct=True) * 100
    
    # Classify regimes
    regimes = pd.Series(index=volatility.index, dtype='object')
    regimes[vol_percentile < thresholds[0]] = 'low_vol'
    regimes[(vol_percentile >= thresholds[0]) & 
            (vol_percentile < thresholds[1])] = 'medium_vol'
    regimes[vol_percentile >= thresholds[1]] = 'high_vol'
    
    return regimes


def calculate_volatility_forecast(
    returns: pd.Series,
    method: str = 'ewma',
    horizon: int = 1,
    params: Optional[Dict] = None
) -> float:
    """
    Forecast future volatility using various methods.
    
    Args:
        returns: Historical returns
        method: 'ewma', 'garch', or 'historical'
        horizon: Forecast horizon in days
        params: Method-specific parameters
        
    Returns:
        Volatility forecast
    """
    if params is None:
        params = {}
    
    if method == 'ewma':
        lambda_ = params.get('lambda', 0.94)
        current_vol = calculate_ewma_volatility(
            returns, 
            lambda_=lambda_, 
            annualize=False
        ).iloc[-1]
        
        # EWMA forecast is just the current estimate
        forecast = current_vol * np.sqrt(horizon)
        
    elif method == 'historical':
        window = params.get('window', 21)
        current_vol = returns.tail(window).std()
        forecast = current_vol * np.sqrt(horizon)
        
    elif method == 'garch':
        # Simplified GARCH(1,1) forecast
        # Note: For production, use arch library
        warnings.warn("Using simplified GARCH. Consider arch library for production.")
        
        omega = params.get('omega', 0.000001)
        alpha = params.get('alpha', 0.1)
        beta = params.get('beta', 0.85)
        
        # Current variance estimate
        returns_squared = returns ** 2
        current_var = returns_squared.mean()
        
        # Multi-step forecast
        forecast_var = current_var
        for _ in range(horizon):
            long_run_var = omega / (1 - alpha - beta)
            forecast_var = omega + (alpha + beta) * forecast_var
        
        forecast = np.sqrt(forecast_var)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Annualize if needed
    if params.get('annualize', True):
        forecast = forecast * np.sqrt(252 / horizon)
    
    return forecast


def calculate_volatility_risk_metrics(
    volatility: pd.Series,
    returns: pd.Series,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate volatility-based risk metrics.
    
    Args:
        volatility: Series of volatility estimates
        returns: Series of returns
        confidence_level: Confidence level for VaR
        
    Returns:
        Dict with risk metrics
    """
    current_vol = volatility.iloc[-1]
    
    # Assuming normal distribution
    z_score = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }.get(confidence_level, 1.96)
    
    # Value at Risk (daily)
    daily_var = -z_score * current_vol / np.sqrt(252)
    
    # Expected Shortfall (Conditional VaR)
    # For normal distribution: ES = -σ * φ(z) / Φ(z)
    from scipy.stats import norm
    es_multiplier = norm.pdf(z_score) / norm.cdf(-z_score)
    daily_es = -es_multiplier * current_vol / np.sqrt(252)
    
    # Volatility percentile (current vs history)
    vol_percentile = (volatility < current_vol).mean() * 100
    
    # Sharpe ratio using current volatility
    annual_return = returns.mean() * 252
    sharpe = annual_return / current_vol if current_vol > 0 else 0
    
    return {
        'current_volatility': current_vol,
        'daily_var': daily_var,
        'daily_expected_shortfall': daily_es,
        'volatility_percentile': vol_percentile,
        'volatility_sharpe': sharpe,
        'weekly_var': daily_var * np.sqrt(5),
        'monthly_var': daily_var * np.sqrt(21)
    }


def calculate_all_volatility_features(
    df: pd.DataFrame,
    window: int = 21,
    ewma_lambda: float = 0.94
) -> pd.DataFrame:
    """
    Calculate comprehensive volatility features for a stock.
    
    Args:
        df: DataFrame with OHLC prices and returns
        window: Default window for calculations
        ewma_lambda: Lambda parameter for EWMA
        
    Returns:
        DataFrame with volatility features
    """
    features = pd.DataFrame(index=df.index)
    
    # Calculate returns if not present
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()
    
    # Realized volatility
    features['realized_vol'] = calculate_realized_volatility(
        df['returns'], window=window
    )
    
    # EWMA volatility
    features['ewma_vol'] = calculate_ewma_volatility(
        df['returns'], lambda_=ewma_lambda
    )
    
    # Garman-Klass volatility
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        features['gk_vol'] = calculate_garman_klass_volatility(
            df['high'], df['low'], df['close'], df['open'], 
            window=window
        )
        
        # Parkinson volatility
        features['parkinson_vol'] = calculate_parkinson_volatility(
            df['high'], df['low'], window=window
        )
        
        # Yang-Zhang volatility
        features['yz_vol'] = calculate_yang_zhang_volatility(
            df['open'], df['high'], df['low'], df['close'],
            window=window
        )
    
    # Volatility regime
    features['vol_regime'] = volatility_regime_detection(
        features['ewma_vol'].fillna(method='ffill')
    )
    
    # Volatility change metrics
    features['vol_change_1d'] = features['ewma_vol'].pct_change()
    features['vol_change_5d'] = features['ewma_vol'].pct_change(5)
    
    return features

def calculate_realized_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate simple realized (historical) volatility.
    
    Args:
        returns: Series of returns
        window: Rolling window period
        annualize: Whether to annualize volatility
        
    Returns:
        Series of volatility values
    """
    vol = returns.rolling(window=window, min_periods=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def calculate_ewma_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate Exponentially Weighted Moving Average (EWMA) volatility.
    
    Args:
        returns: Series of returns
        lambda_: Decay factor (0.94 is RiskMetrics standard)
        annualize: Whether to annualize volatility
        
    Returns:
        Series of EWMA volatility
    """
    # Calculate squared returns
    returns_squared = returns ** 2
    
    # Initialize with first observation or expanding window variance
    ewma_var = returns_squared.ewm(
        alpha=1-lambda_, 
        adjust=False
    ).mean()
    
    # Convert variance to volatility
    ewma_vol = np.sqrt(ewma_var)
    
    if annualize:
        ewma_vol = ewma_vol * np.sqrt(252)
    
    return ewma_vol


def calculate_garman_klass_volatility(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate Garman-Klass volatility estimator.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        open_: Series of open prices
        window: Rolling window for averaging
        annualize: Whether to annualize
        
    Returns:
        Series of GK volatility estimates
    """
    # Ensure no zero or negative prices
    high = high.replace(0, np.nan)
    low = low.replace(0, np.nan)
    close = close.replace(0, np.nan)
    open_ = open_.replace(0, np.nan)
    
    # Calculate components
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    
    # Garman-Klass formula
    gk_daily = np.sqrt(
        0.5 * log_hl**2 - 
        (2 * np.log(2) - 1) * log_co**2
    )
    
    # Rolling average
    gk_vol = gk_daily.rolling(window=window, min_periods=window).mean()
    
    if annualize:
        gk_vol = gk_vol * np.sqrt(252)
    
    return gk_vol


# Add this simplified version if parkinson is imported anywhere
def calculate_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate Parkinson volatility estimator.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        window: Rolling window for averaging
        annualize: Whether to annualize
        
    Returns:
        Series of Parkinson volatility estimates
    """
    # Ensure no zero or negative prices
    high = high.replace(0, np.nan)
    low = low.replace(0, np.nan)
    
    # Calculate daily Parkinson volatility
    log_hl = np.log(high / low)
    parkinson_daily = log_hl / (2 * np.sqrt(np.log(2)))
    
    # Rolling average
    parkinson_vol = parkinson_daily.rolling(
        window=window, 
        min_periods=window
    ).mean()
    
    if annualize:
        parkinson_vol = parkinson_vol * np.sqrt(252)
    
    return parkinson_vol