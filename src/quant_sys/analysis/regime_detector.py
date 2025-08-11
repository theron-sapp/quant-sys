"""
Market Regime Detection Module

Implements multi-signal regime detection to determine optimal strategy allocation
between dividend-focused and growth-focused approaches.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import select, text
from sqlalchemy.engine import Engine

class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_GROWTH = "strong_growth"      # VIX < 15, strong momentum
    GROWTH = "growth"                     # VIX < 20, positive momentum  
    NEUTRAL = "neutral"                   # VIX 20-30, mixed signals
    DIVIDEND = "dividend"                 # VIX > 30, defensive needed
    CRISIS = "crisis"                     # VIX > 40, maximum defense

@dataclass
class RegimeSignals:
    """Container for all regime detection signals"""
    vix_level: float
    vix_percentile: float
    vix_signal: float  # -1 to +1
    spy_momentum: float
    spy_rsi: float
    macd_histogram: float
    momentum_signal: float  # -1 to +1
    interest_rate: Optional[float] = None
    gdp_growth: Optional[float] = None
    inflation_rate: Optional[float] = None
    economic_signal: Optional[float] = None  # -1 to +1
    
@dataclass
class RegimeScore:
    """Composite regime scoring"""
    composite_score: float  # -1 to +1 scale
    regime: MarketRegime
    confidence: float  # 0 to 1
    signals: RegimeSignals
    
class RegimeDetector:
    """
    Detects market regimes using multiple signals:
    - VIX levels and percentiles
    - Market momentum (RSI, MACD)
    - Economic indicators (when available)
    """
    
    def __init__(self, engine: Engine, config: Dict = None):
        self.engine = engine
        self.config = config or self._default_config()
        self.regime_history: List[RegimeScore] = []
        self._current_regime: Optional[MarketRegime] = None
        self._regime_start_date: Optional[datetime] = None
        
    def _default_config(self) -> Dict:
        """Default configuration matching course materials"""
        return {
            "vix_thresholds": {
                "strong_growth": 15,
                "growth_max": 20,
                "neutral_range": [20, 30],
                "dividend_min": 30,
                "crisis_min": 40
            },
            "signal_weights": {
                "vix": 0.40,
                "momentum": 0.20,
                "macd": 0.20,
                "economic": 0.20
            },
            "persistence": {
                "min_days": 5,
                "confidence_threshold": 0.75,
                "lookback_period": 21
            },
            "momentum": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            }
        }
    
    def detect_current_regime(self, as_of_date: str = None) -> RegimeScore:
        """
        Main method to detect current market regime
        
        Returns:
            RegimeScore with regime classification and confidence
        """
        if as_of_date is None:
            as_of_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            
        # Get market data
        vix_data = self._get_vix_data(as_of_date)
        spy_data = self._get_spy_data(as_of_date)
        
        # Calculate individual signals
        signals = self._calculate_signals(vix_data, spy_data, as_of_date)
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(signals)
        
        # Determine regime and confidence
        regime = self._score_to_regime(composite_score, signals.vix_level)
        confidence = self._calculate_confidence(signals, composite_score)
        
        # Create regime score
        regime_score = RegimeScore(
            composite_score=composite_score,
            regime=regime,
            confidence=confidence,
            signals=signals
        )
        
        # Check persistence before confirming regime change
        if self._should_change_regime(regime_score):
            self._current_regime = regime
            self._regime_start_date = pd.Timestamp(as_of_date)
            
        # Store in history
        self.regime_history.append(regime_score)
        
        return regime_score
    
    def _get_vix_data(self, as_of_date: str) -> pd.DataFrame:
        """Fetch VIX data from database"""
        query = text("""
            SELECT date, adj_close as vix_close
            FROM prices_daily
            WHERE symbol = '^VIX'
            AND date <= :as_of_date
            ORDER BY date DESC
            LIMIT 252
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"as_of_date": as_of_date})
        
        if df.empty:
            raise ValueError("No VIX data available")
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df
    
    def _get_spy_data(self, as_of_date: str) -> pd.DataFrame:
        """Fetch SPY data from database"""
        query = text("""
            SELECT date, open, high, low, close, adj_close, volume
            FROM prices_daily
            WHERE symbol = 'SPY'
            AND date <= :as_of_date
            ORDER BY date DESC
            LIMIT 252
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"as_of_date": as_of_date})
            
        if df.empty:
            raise ValueError("No SPY data available")
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df
    
    def _calculate_signals(self, vix_data: pd.DataFrame, spy_data: pd.DataFrame, 
                          as_of_date: str) -> RegimeSignals:
        """Calculate all regime detection signals"""
        
        # VIX signals
        current_vix = vix_data['vix_close'].iloc[-1]
        vix_percentile = (vix_data['vix_close'] <= current_vix).mean()
        vix_signal = self._calculate_vix_signal(current_vix)
        
        # SPY momentum signals
        spy_momentum = self._calculate_momentum(spy_data)
        spy_rsi = self._calculate_rsi(spy_data['adj_close'])
        macd_hist = self._calculate_macd_histogram(spy_data['adj_close'])
        momentum_signal = self._calculate_momentum_signal(spy_momentum, spy_rsi, macd_hist)
        
        # Economic signals (placeholder - would need FRED data)
        economic_signal = None
        
        return RegimeSignals(
            vix_level=current_vix,
            vix_percentile=vix_percentile,
            vix_signal=vix_signal,
            spy_momentum=spy_momentum,
            spy_rsi=spy_rsi,
            macd_histogram=macd_hist,
            momentum_signal=momentum_signal,
            economic_signal=economic_signal
        )
    
    def _calculate_vix_signal(self, vix_level: float) -> float:
        """
        Convert VIX level to signal (-1 to +1)
        Low VIX (+1) = Growth favorable
        High VIX (-1) = Dividend favorable
        """
        thresholds = self.config["vix_thresholds"]
        
        if vix_level < thresholds["strong_growth"]:
            return 1.0  # Strong growth signal
        elif vix_level < thresholds["growth_max"]:
            # Linear interpolation between strong_growth and growth_max
            return 1.0 - (vix_level - thresholds["strong_growth"]) / (thresholds["growth_max"] - thresholds["strong_growth"])
        elif vix_level <= thresholds["neutral_range"][1]:
            # Linear interpolation in neutral range
            range_size = thresholds["neutral_range"][1] - thresholds["neutral_range"][0]
            position = (vix_level - thresholds["neutral_range"][0]) / range_size
            return 0.5 - position  # From +0.5 to -0.5
        elif vix_level < thresholds["crisis_min"]:
            # Linear interpolation between dividend and crisis
            return -0.5 - 0.5 * (vix_level - thresholds["neutral_range"][1]) / (thresholds["crisis_min"] - thresholds["neutral_range"][1])
        else:
            return -1.0  # Crisis signal
    
    def _calculate_momentum(self, spy_data: pd.DataFrame) -> float:
        """Calculate SPY momentum (21-day return)"""
        if len(spy_data) < 21:
            return 0.0
        return (spy_data['adj_close'].iloc[-1] / spy_data['adj_close'].iloc[-21] - 1) * 100
    
    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> float:
        """Calculate RSI"""
        if period is None:
            period = self.config["momentum"]["rsi_period"]
            
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if insufficient data
            
        # Calculate price changes
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def _calculate_macd_histogram(self, prices: pd.Series) -> float:
        """Calculate MACD histogram value"""
        cfg = self.config["momentum"]
        
        if len(prices) < cfg["macd_slow"] + cfg["macd_signal"]:
            return 0.0
            
        # Calculate MACD components
        exp1 = prices.ewm(span=cfg["macd_fast"], adjust=False).mean()
        exp2 = prices.ewm(span=cfg["macd_slow"], adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=cfg["macd_signal"], adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Normalize histogram to -1 to +1 range (approximate)
        hist_value = histogram.iloc[-1]
        hist_std = histogram.std()
        if hist_std > 0:
            normalized = np.clip(hist_value / (2 * hist_std), -1, 1)
        else:
            normalized = 0.0
            
        return normalized
    
    def _calculate_momentum_signal(self, momentum: float, rsi: float, macd_hist: float) -> float:
        """Combine momentum indicators into single signal"""
        
        # Momentum contribution (21-day return)
        if momentum > 5:
            mom_signal = 1.0
        elif momentum > 2:
            mom_signal = 0.5
        elif momentum < -5:
            mom_signal = -1.0
        elif momentum < -2:
            mom_signal = -0.5
        else:
            mom_signal = 0.0
            
        # RSI contribution
        if rsi > self.config["momentum"]["rsi_overbought"]:
            rsi_signal = -0.5  # Overbought = negative for mean reversion
        elif rsi < self.config["momentum"]["rsi_oversold"]:
            rsi_signal = 0.5   # Oversold = positive for mean reversion
        else:
            rsi_signal = (rsi - 50) / 50  # Normalize to -1 to +1
            
        # Weight the signals
        combined = (mom_signal * 0.5 + rsi_signal * 0.3 + macd_hist * 0.2)
        return np.clip(combined, -1, 1)
    
    def _calculate_composite_score(self, signals: RegimeSignals) -> float:
        """Calculate weighted composite regime score"""
        weights = self.config["signal_weights"]
        
        # Start with VIX and momentum signals (always available)
        score = (
            weights["vix"] * signals.vix_signal +
            weights["momentum"] * signals.momentum_signal +
            weights["macd"] * signals.macd_histogram
        )
        
        # Normalize weights if economic signal not available
        if signals.economic_signal is None:
            total_weight = weights["vix"] + weights["momentum"] + weights["macd"]
            score = score / total_weight
        else:
            score += weights["economic"] * signals.economic_signal
            
        return np.clip(score, -1, 1)
    
    def _score_to_regime(self, score: float, vix_level: float) -> MarketRegime:
        """Convert composite score to regime classification"""
        
        # Check for crisis first (VIX override)
        if vix_level >= self.config["vix_thresholds"]["crisis_min"]:
            return MarketRegime.CRISIS
            
        # Map score to regime
        if score > 0.5:
            return MarketRegime.STRONG_GROWTH
        elif score > 0.1:
            return MarketRegime.GROWTH
        elif score < -0.5:
            return MarketRegime.CRISIS
        elif score < -0.1:
            return MarketRegime.DIVIDEND
        else:
            return MarketRegime.NEUTRAL
    
    def _calculate_confidence(self, signals: RegimeSignals, composite_score: float) -> float:
        """Calculate confidence in regime detection"""
        
        # Start with base confidence from score magnitude
        confidence = abs(composite_score)
        
        # Boost confidence if signals align
        signal_agreement = 0
        signal_count = 0
        
        # Check VIX and momentum alignment
        if signals.vix_signal * signals.momentum_signal > 0:
            signal_agreement += 1
        signal_count += 1
        
        # Check VIX percentile extremes
        if signals.vix_percentile > 0.8 or signals.vix_percentile < 0.2:
            confidence *= 1.2
            
        # Adjust for signal agreement
        if signal_count > 0:
            agreement_ratio = signal_agreement / signal_count
            confidence = confidence * (0.7 + 0.3 * agreement_ratio)
            
        return np.clip(confidence, 0, 1)
    
    def _should_change_regime(self, new_score: RegimeScore) -> bool:
        """
        Determine if regime should change based on persistence rules
        Prevents whipsaw by requiring confidence and duration thresholds
        """
        
        # Always accept first regime
        if self._current_regime is None:
            return True
            
        # Check if regime is actually different
        if new_score.regime == self._current_regime:
            return False
            
        # Check confidence threshold
        if new_score.confidence < self.config["persistence"]["confidence_threshold"]:
            return False
            
        # Check persistence (need N days of consistent signal)
        min_days = self.config["persistence"]["min_days"]
        if len(self.regime_history) < min_days:
            return True  # Not enough history, accept change
            
        # Look at recent history
        recent_regimes = [r.regime for r in self.regime_history[-min_days:]]
        consistent = all(r == new_score.regime for r in recent_regimes[-min_days+1:])
        
        return consistent
    
    def get_regime_allocation(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get recommended strategy allocation based on regime
        
        Returns:
            Dict with 'dividend_weight' and 'growth_weight' (sum to 1.0)
        """
        allocations = {
            MarketRegime.STRONG_GROWTH: {"dividend": 0.2, "growth": 0.8},
            MarketRegime.GROWTH: {"dividend": 0.4, "growth": 0.6},
            MarketRegime.NEUTRAL: {"dividend": 0.5, "growth": 0.5},
            MarketRegime.DIVIDEND: {"dividend": 0.6, "growth": 0.4},
            MarketRegime.CRISIS: {"dividend": 0.8, "growth": 0.2}
        }
        return allocations[regime]
    
    def get_risk_scaling(self, regime: MarketRegime) -> float:
        """
        Get risk scaling factor based on regime
        Used to adjust gross exposure based on market conditions
        
        Returns:
            Scaling factor (0.3 to 1.0)
        """
        scaling = {
            MarketRegime.STRONG_GROWTH: 1.0,
            MarketRegime.GROWTH: 0.9,
            MarketRegime.NEUTRAL: 0.7,
            MarketRegime.DIVIDEND: 0.5,
            MarketRegime.CRISIS: 0.3
        }
        return scaling[regime]
    
    def get_regime_summary(self) -> pd.DataFrame:
        """Get summary of recent regime history"""
        if not self.regime_history:
            return pd.DataFrame()
            
        data = []
        for score in self.regime_history[-20:]:  # Last 20 observations
            data.append({
                'regime': score.regime.value,
                'composite_score': score.composite_score,
                'confidence': score.confidence,
                'vix_level': score.signals.vix_level,
                'vix_signal': score.signals.vix_signal,
                'momentum_signal': score.signals.momentum_signal
            })
            
        return pd.DataFrame(data)