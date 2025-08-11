"""
Signal combination and regime filtering.
Location: src/quant_sys/signals/signal_combiner.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from ..analysis.regime_detector import MarketRegime, RegimeScore

logger = logging.getLogger(__name__)

@dataclass
class CombinedSignal:
    """Final combined trading signal."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    position_type: str  # 'long', 'short', 'flat'
    signal_strength: float  # 0-1 scale for position sizing
    
    # Component signals
    momentum_signal: float
    mean_reversion_signal: float
    regime_adjusted_signal: float
    
    # Metadata
    regime: MarketRegime
    volatility: float
    correlation_score: float
    reasons: List[str]
    
    # Risk metrics
    suggested_position_size: float  # As % of capital
    stop_loss_pct: float
    confidence: float


class SignalCombiner:
    """Combine multiple signal types with regime awareness."""
    
    def __init__(self, engine, config, regime_detector):
        """
        Initialize signal combiner.
        
        Args:
            engine: SQLAlchemy engine
            config: Settings object
            regime_detector: RegimeDetector instance
        """
        self.engine = engine
        self.config = config
        self.regime_detector = regime_detector
        
        # Signal weights from config
        self.momentum_weight = config.signals.ts_mom.weight
        self.mean_rev_weight = config.signals.mean_rev.weight
        
        # Risk parameters
        self.max_position_size = 0.08  # 8% max per position
        self.base_stop_loss = 0.20  # 20% stop loss
        
    def combine_signals(
        self,
        momentum_signals: Dict,
        mean_reversion_signals: Dict,
        date: Optional[str] = None
    ) -> Dict[str, CombinedSignal]:
        """
        Combine momentum and mean reversion signals with regime filtering.
        
        Args:
            momentum_signals: Dictionary of momentum signals
            mean_reversion_signals: Dictionary of mean reversion signals
            date: Date for regime detection
            
        Returns:
            Dictionary of symbol -> CombinedSignal
        """
        # Get current regime
        regime_score = self.regime_detector.detect_current_regime(as_of_date=date)
        regime_weights = self._get_regime_weights(regime_score)
        
        logger.info(f"Combining signals for regime: {regime_score.regime.value}")
        logger.info(f"Weights - Momentum: {regime_weights['momentum']:.1%}, "
                   f"Mean Rev: {regime_weights['mean_reversion']:.1%}")
        
        # Get all unique symbols
        all_symbols = set(momentum_signals.keys()) | set(mean_reversion_signals.keys())
        
        # Fetch additional data
        volatility_data = self._fetch_volatility(list(all_symbols))
        correlation_data = self._calculate_correlation_scores(list(all_symbols))
        
        # Combine signals for each symbol
        combined_signals = {}
        
        for symbol in all_symbols:
            # Get component signals
            mom_signal = momentum_signals.get(symbol)
            rev_signal = mean_reversion_signals.get(symbol)
            
            # Convert to numeric values
            mom_value = self._signal_to_value(mom_signal) if mom_signal else 0
            rev_value = self._signal_to_value(rev_signal) if rev_signal else 0
            
            # Apply regime weights
            weighted_signal = (
                regime_weights['momentum'] * mom_value +
                regime_weights['mean_reversion'] * rev_value
            )
            
            # Apply regime-specific adjustments
            adjusted_signal = self._apply_regime_adjustments(
                weighted_signal,
                regime_score,
                symbol,
                volatility_data.get(symbol, 0.20)
            )
            
            # Apply risk filters
            final_signal = self._apply_risk_filters(
                adjusted_signal,
                symbol,
                volatility_data.get(symbol, 0.20),
                correlation_data.get(symbol, 0.5)
            )
            
            # Determine action and position type
            action, position_type = self._determine_action(final_signal)
            
            # Calculate position sizing
            position_size = self._calculate_position_size(
                abs(final_signal),
                volatility_data.get(symbol, 0.20),
                correlation_data.get(symbol, 0.5)
            )
            
            # Calculate stop loss
            stop_loss = self._calculate_stop_loss(
                volatility_data.get(symbol, 0.20),
                regime_score.regime
            )
            
            # Build reasons
            reasons = self._build_signal_reasons(
                mom_signal,
                rev_signal,
                regime_score,
                weighted_signal,
                adjusted_signal
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                mom_signal,
                rev_signal,
                regime_score,
                volatility_data.get(symbol, 0.20)
            )
            
            # Create combined signal
            combined_signals[symbol] = CombinedSignal(
                symbol=symbol,
                action=action,
                position_type=position_type,
                signal_strength=abs(final_signal),
                momentum_signal=mom_value,
                mean_reversion_signal=rev_value,
                regime_adjusted_signal=adjusted_signal,
                regime=regime_score.regime,
                volatility=volatility_data.get(symbol, 0.20),
                correlation_score=correlation_data.get(symbol, 0.5),
                reasons=reasons,
                suggested_position_size=position_size,
                stop_loss_pct=stop_loss,
                confidence=confidence
            )
            
        # Apply portfolio-level constraints
        combined_signals = self._apply_portfolio_constraints(combined_signals)
        
        return combined_signals
    
    def _get_regime_weights(self, regime_score: RegimeScore) -> Dict[str, float]:
        """
        Get signal weights based on market regime.
        
        Different regimes favor different strategies:
        - Growth regimes favor momentum
        - Crisis/defensive regimes favor mean reversion
        """
        regime = regime_score.regime
        
        if regime == MarketRegime.STRONG_GROWTH:
            return {
                'momentum': 0.80,
                'mean_reversion': 0.20
            }
        elif regime == MarketRegime.GROWTH:
            return {
                'momentum': 0.65,
                'mean_reversion': 0.35
            }
        elif regime == MarketRegime.NEUTRAL:
            return {
                'momentum': 0.50,
                'mean_reversion': 0.50
            }
        elif regime == MarketRegime.DIVIDEND:
            return {
                'momentum': 0.30,
                'mean_reversion': 0.70
            }
        else:  # CRISIS
            return {
                'momentum': 0.10,
                'mean_reversion': 0.90
            }
    
    def _signal_to_value(self, signal) -> float:
        """Convert signal object to numeric value."""
        if hasattr(signal, 'signal_strength'):
            strength = signal.signal_strength
            
            if hasattr(signal, 'signal_type'):
                if signal.signal_type == 'short':
                    return -strength
                elif signal.signal_type == 'long':
                    return strength
                else:
                    return 0
        return 0
    
    def _apply_regime_adjustments(
        self,
        signal: float,
        regime_score: RegimeScore,
        symbol: str,
        volatility: float
    ) -> float:
        """
        Apply regime-specific adjustments to signals.
        
        - In crisis: Reduce all signals, especially longs
        - In strong growth: Boost momentum signals
        - In high volatility: Reduce signal strength
        """
        adjusted = signal
        regime = regime_score.regime
        
        # Crisis adjustments
        if regime == MarketRegime.CRISIS:
            if signal > 0:  # Reduce long signals more
                adjusted *= 0.3
            else:  # Keep some short signals
                adjusted *= 0.7
                
        # Strong growth adjustments
        elif regime == MarketRegime.STRONG_GROWTH:
            if signal > 0:  # Boost long signals
                adjusted *= 1.2
            else:  # Reduce short signals
                adjusted *= 0.5
                
        # Volatility adjustments
        if regime_score.signals.vix_level > 30:
            adjusted *= 0.7  # Reduce in high VIX
        elif regime_score.signals.vix_level < 15:
            adjusted *= 1.1  # Boost in low VIX
            
        # Individual stock volatility adjustment
        if volatility > 0.40:  # Very high volatility
            adjusted *= 0.5
        elif volatility > 0.30:  # High volatility
            adjusted *= 0.7
            
        return np.clip(adjusted, -1.0, 1.0)
    
    def _apply_risk_filters(
        self,
        signal: float,
        symbol: str,
        volatility: float,
        correlation: float
    ) -> float:
        """
        Apply risk-based filters to signals.
        
        - High correlation: Reduce signal
        - Extreme volatility: Filter out
        - Position limits: Cap signal strength
        """
        filtered = signal
        
        # Correlation filter
        if correlation > 0.7:
            filtered *= 0.5  # High correlation with market
            
        # Extreme volatility filter
        if volatility > 0.50:
            filtered = 0  # Too risky
            
        # Signal strength cap
        filtered = np.clip(filtered, -0.95, 0.95)
        
        return filtered
    
    def _determine_action(self, signal: float) -> tuple[str, str]:
        """
        Determine trading action from signal value.
        
        Returns:
            (action, position_type) tuple
        """
        if signal > 0.3:
            return ('BUY', 'long')
        elif signal < -0.3:
            return ('SELL', 'short')
        else:
            return ('HOLD', 'flat')
    
    def _calculate_position_size(
        self,
        signal_strength: float,
        volatility: float,
        correlation: float
    ) -> float:
        """
        Calculate position size based on signal strength and risk.
        
        Uses inverse volatility weighting with signal strength scaling.
        """
        # Base position size from signal strength
        base_size = signal_strength * self.max_position_size
        
        # Volatility adjustment (inverse volatility)
        vol_scalar = min(0.15 / volatility, 1.5)  # Target 15% volatility
        
        # Correlation adjustment
        corr_scalar = 1.0 - (correlation * 0.3)  # Reduce for high correlation
        
        # Final position size
        position_size = base_size * vol_scalar * corr_scalar
        
        # Apply limits
        return min(position_size, self.max_position_size)
    
    def _calculate_stop_loss(
        self,
        volatility: float,
        regime: MarketRegime
    ) -> float:
        """
        Calculate stop loss percentage based on volatility and regime.
        """
        # Base stop loss
        stop_loss = self.base_stop_loss
        
        # Adjust for volatility (wider stops for volatile stocks)
        stop_loss = stop_loss * (1 + volatility)
        
        # Tighten in crisis regime
        if regime == MarketRegime.CRISIS:
            stop_loss *= 0.75
            
        # Minimum and maximum
        return np.clip(stop_loss, 0.10, 0.30)
    
    def _build_signal_reasons(
        self,
        mom_signal,
        rev_signal,
        regime_score: RegimeScore,
        weighted_signal: float,
        adjusted_signal: float
    ) -> List[str]:
        """Build list of reasons for the signal."""
        reasons = []
        
        # Regime reason
        reasons.append(f"Regime: {regime_score.regime.value} ({regime_score.confidence:.0%} confidence)")
        
        # Component signal reasons
        if mom_signal and hasattr(mom_signal, 'reasons'):
            for reason in mom_signal.reasons[:2]:  # Top 2 momentum reasons
                reasons.append(f"Momentum: {reason}")
                
        if rev_signal and hasattr(rev_signal, 'reasons'):
            for reason in rev_signal.reasons[:1]:  # Top mean reversion reason
                reasons.append(f"MeanRev: {reason}")
                
        # Signal adjustment reasons
        if abs(adjusted_signal - weighted_signal) > 0.1:
            if adjusted_signal < weighted_signal:
                reasons.append("Signal reduced by risk filters")
            else:
                reasons.append("Signal boosted by regime")
                
        return reasons
    
    def _calculate_confidence(
        self,
        mom_signal,
        rev_signal,
        regime_score: RegimeScore,
        volatility: float
    ) -> float:
        """
        Calculate overall signal confidence.
        
        Based on:
        - Agreement between signals
        - Regime confidence
        - Volatility levels
        """
        confidence = 0.5  # Base confidence
        
        # Signal agreement
        if mom_signal and rev_signal:
            mom_value = self._signal_to_value(mom_signal)
            rev_value = self._signal_to_value(rev_signal)
            
            if np.sign(mom_value) == np.sign(rev_value):
                confidence += 0.2  # Signals agree
            else:
                confidence -= 0.1  # Signals disagree
                
        # Regime confidence
        confidence += regime_score.confidence * 0.2
        
        # Volatility penalty
        if volatility > 0.30:
            confidence -= 0.1
        elif volatility < 0.15:
            confidence += 0.1
            
        return np.clip(confidence, 0.1, 0.95)
    
    def _fetch_volatility(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch volatility data for symbols."""
        if not symbols:
            return {}
            
        symbols_str = "','".join(symbols)
        
        query = f"""
        SELECT symbol, value as volatility
        FROM technical_indicators
        WHERE indicator_name = 'ewma_vol'
        AND symbol IN ('{symbols_str}')
        AND date = (SELECT MAX(date) FROM technical_indicators)
        """
        
        df = pd.read_sql_query(query, self.engine)
        return dict(zip(df['symbol'], df['volatility']))
    
    def _calculate_correlation_scores(self, symbols: List[str]) -> Dict[str, float]:
        """
        Calculate correlation with market (SPY) for each symbol.
        
        High correlation means less diversification benefit.
        """
        if not symbols:
            return {}
            
        # For now, return placeholder values
        # TODO: Implement actual correlation calculation
        return {symbol: 0.5 for symbol in symbols}
    
    def _apply_portfolio_constraints(
        self,
        signals: Dict[str, CombinedSignal]
    ) -> Dict[str, CombinedSignal]:
        """
        Apply portfolio-level constraints.
        
        - Maximum number of positions
        - Sector concentration limits
        - Total exposure limits
        """
        # Sort by signal strength
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1].signal_strength,
            reverse=True
        )
        
        # Keep only top N signals
        max_positions = 30
        if len(sorted_signals) > max_positions:
            # Keep top long and short signals
            long_signals = [(s, sig) for s, sig in sorted_signals 
                          if sig.position_type == 'long'][:20]
            short_signals = [(s, sig) for s, sig in sorted_signals 
                           if sig.position_type == 'short'][:10]
            
            sorted_signals = long_signals + short_signals
            
        # Check total exposure
        total_exposure = sum(sig.suggested_position_size 
                           for _, sig in sorted_signals)
        
        # Scale down if over limit
        if total_exposure > 1.0:
            scale_factor = 0.95 / total_exposure
            for _, sig in sorted_signals:
                sig.suggested_position_size *= scale_factor
                
        return dict(sorted_signals)