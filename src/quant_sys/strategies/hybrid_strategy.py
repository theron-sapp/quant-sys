"""
Hybrid trading strategy with regime-based switching.
Location: src/quant_sys/strategies/hybrid_strategy.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from ..analysis.regime_detector import RegimeDetector, MarketRegime
from ..signals.momentum_signals import MomentumSignalGenerator
from ..signals.mean_reversion_signals import MeanReversionSignalGenerator
from ..signals.signal_combiner import SignalCombiner

logger = logging.getLogger(__name__)

@dataclass
class PortfolioAllocation:
    """Portfolio allocation across strategies."""
    momentum_weight: float
    dividend_weight: float
    mean_reversion_weight: float
    cash_weight: float
    
    # Risk adjustments
    gross_exposure: float  # % of capital deployed
    leverage: float  # Gross/Net exposure
    
    # Metadata
    regime: MarketRegime
    confidence: float
    rebalance_needed: bool


@dataclass
class TradeSignal:
    """Final trade signal for execution."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    shares: int
    price: float
    position_size: float
    position_size_pct: float
    
    # Strategy attribution
    strategy: str  # 'momentum', 'dividend', 'mean_reversion', 'hybrid'
    signal_strength: float
    confidence: float
    
    # Risk parameters
    stop_loss: float
    take_profit: Optional[float]
    max_holding_days: int
    
    # Reasons
    reasons: List[str]


class HybridStrategy:
    """
    Master strategy that combines multiple sub-strategies based on market regime.
    
    Dynamically allocates between:
    - Momentum (growth regimes)
    - Dividend/Quality (defensive regimes)
    - Mean Reversion (ranging markets)
    """
    
    def __init__(self, engine, config):
        """
        Initialize hybrid strategy.
        
        Args:
            engine: SQLAlchemy engine
            config: Settings object
        """
        self.engine = engine
        self.config = config
        
        # Initialize components
        self.regime_detector = RegimeDetector(engine)
        self.momentum_generator = MomentumSignalGenerator(engine, config)
        self.mean_rev_generator = MeanReversionSignalGenerator(engine, config)
        self.signal_combiner = SignalCombiner(engine, config, self.regime_detector)
        
        # Strategy parameters
        self.capital_base = config.capital_base
        self.gross_exposure_cap = config.gross_exposure_cap
        self.max_drawdown = config.risk_management.max_drawdown
        
        # Regime transition parameters
        self.transition_days = 5  # Days to smooth strategy transitions
        self.min_regime_confidence = 0.60  # Minimum confidence for regime change
        
        # Track regime history
        self.regime_history = []
        self.last_regime_change = None
        
    def generate_signals(
        self,
        date: Optional[str] = None,
        current_positions: Optional[Dict] = None
    ) -> List[TradeSignal]:
        """
        Generate trading signals based on current market regime.
        
        Args:
            date: Date for signal generation
            current_positions: Current portfolio positions
            
        Returns:
            List of trade signals
        """
        logger.info("=" * 60)
        logger.info(f"HYBRID STRATEGY - Generating signals for {date or 'latest'}")
        logger.info("=" * 60)
        
        # Detect current regime
        regime_score = self.regime_detector.detect_current_regime(as_of_date=date)
        logger.info(f"Market Regime: {regime_score.regime.value} (Confidence: {regime_score.confidence:.1%})")
        
        # Update regime history
        self._update_regime_history(regime_score, date)
        
        # Get portfolio allocation based on regime
        allocation = self._calculate_allocation(regime_score)
        logger.info(f"Strategy Allocation - Momentum: {allocation.momentum_weight:.0%}, "
                   f"Mean Rev: {allocation.mean_reversion_weight:.0%}, "
                   f"Cash: {allocation.cash_weight:.0%}")
        
        # Generate component signals
        momentum_signals = self.momentum_generator.generate_signals(date=date)
        mean_rev_signals = self.mean_rev_generator.generate_signals(date=date)
        
        logger.info(f"Raw Signals - Momentum: {len(momentum_signals)}, "
                   f"Mean Reversion: {len(mean_rev_signals)}")
        
        # Combine signals with regime awareness
        combined_signals = self.signal_combiner.combine_signals(
            momentum_signals,
            mean_rev_signals,
            date=date
        )
        
        # Generate final trade signals
        trade_signals = self._generate_trade_signals(
            combined_signals,
            allocation,
            current_positions
        )
        
        logger.info(f"Final Trade Signals: {len(trade_signals)} "
                   f"(Buy: {sum(1 for s in trade_signals if s.action == 'BUY')}, "
                   f"Sell: {sum(1 for s in trade_signals if s.action == 'SELL')})")
        
        return trade_signals
    
    def _calculate_allocation(self, regime_score) -> PortfolioAllocation:
        """
        Calculate portfolio allocation based on regime.
        
        Smoothly transitions between strategies to avoid whipsaw.
        """
        regime = regime_score.regime
        confidence = regime_score.confidence
        
        # Base allocations by regime
        if regime == MarketRegime.STRONG_GROWTH:
            base_allocation = {
                'momentum': 0.70,
                'dividend': 0.10,
                'mean_reversion': 0.20,
                'cash': 0.00
            }
            gross_exposure = 1.0  # Full exposure
            
        elif regime == MarketRegime.GROWTH:
            base_allocation = {
                'momentum': 0.50,
                'dividend': 0.20,
                'mean_reversion': 0.30,
                'cash': 0.00
            }
            gross_exposure = 0.90
            
        elif regime == MarketRegime.NEUTRAL:
            base_allocation = {
                'momentum': 0.30,
                'dividend': 0.30,
                'mean_reversion': 0.40,
                'cash': 0.00
            }
            gross_exposure = 0.80
            
        elif regime == MarketRegime.DIVIDEND:
            base_allocation = {
                'momentum': 0.20,
                'dividend': 0.50,
                'mean_reversion': 0.20,
                'cash': 0.10
            }
            gross_exposure = 0.70
            
        else:  # CRISIS
            base_allocation = {
                'momentum': 0.05,
                'dividend': 0.35,
                'mean_reversion': 0.10,
                'cash': 0.50
            }
            gross_exposure = 0.30  # Minimal exposure
            
        # Adjust for confidence
        if confidence < 0.70:
            # Low confidence - move toward neutral allocation
            neutral = {'momentum': 0.33, 'dividend': 0.33, 'mean_reversion': 0.34, 'cash': 0.00}
            
            blend_factor = confidence / 0.70
            for key in base_allocation:
                base_allocation[key] = (
                    base_allocation[key] * blend_factor +
                    neutral[key] * (1 - blend_factor)
                )
                
        # Check if rebalance needed
        rebalance_needed = self._check_rebalance_trigger(regime_score)
        
        return PortfolioAllocation(
            momentum_weight=base_allocation['momentum'],
            dividend_weight=base_allocation['dividend'],
            mean_reversion_weight=base_allocation['mean_reversion'],
            cash_weight=base_allocation['cash'],
            gross_exposure=gross_exposure,
            leverage=1.0,  # No leverage for now
            regime=regime,
            confidence=confidence,
            rebalance_needed=rebalance_needed
        )
    
    def _generate_trade_signals(
        self,
        combined_signals: Dict,
        allocation: PortfolioAllocation,
        current_positions: Optional[Dict]
    ) -> List[TradeSignal]:
        """
        Generate final trade signals based on combined signals and allocation.
        """
        trade_signals = []
        
        # Calculate available capital
        available_capital = self.gross_exposure_cap * allocation.gross_exposure
        
        # Sort signals by strength
        sorted_signals = sorted(
            combined_signals.items(),
            key=lambda x: x[1].signal_strength,
            reverse=True
        )
        
        # Track capital allocation
        capital_allocated = 0
        positions_added = 0
        max_positions = 30
        
        for symbol, signal in sorted_signals:
            # Skip if no action needed
            if signal.action == 'HOLD':
                continue
                
            # Check position limit
            if positions_added >= max_positions:
                break
                
            # Calculate position size based on signal and allocation
            base_position_size = signal.suggested_position_size * available_capital
            
            # Adjust for strategy allocation
            if signal.momentum_signal > signal.mean_reversion_signal:
                # Momentum-driven signal
                position_size = base_position_size * allocation.momentum_weight
                strategy = 'momentum'
                max_holding = 63  # 3 months for momentum
                
            else:
                # Mean reversion signal
                position_size = base_position_size * allocation.mean_reversion_weight
                strategy = 'mean_reversion'
                max_holding = 21  # 3 weeks for mean reversion
                
            # Check capital constraint
            if capital_allocated + position_size > available_capital:
                position_size = available_capital - capital_allocated
                if position_size < 100:  # Minimum position size
                    break
                    
            # Get current price
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue
                
            # Calculate shares
            shares = int(position_size / current_price)
            if shares == 0:
                continue
                
            # Determine action
            if signal.action == 'BUY':
                action = 'BUY'
            elif signal.action == 'SELL':
                action = 'SELL'
                shares = -shares  # Negative for short positions
            else:
                continue
                
            # Create trade signal
            trade_signal = TradeSignal(
                symbol=symbol,
                action=action,
                shares=abs(shares),
                price=current_price,
                position_size=abs(shares * current_price),
                position_size_pct=position_size / self.gross_exposure_cap,
                strategy=strategy,
                signal_strength=signal.signal_strength,
                confidence=signal.confidence,
                stop_loss=current_price * (1 - signal.stop_loss_pct),
                take_profit=None,  # Can be added based on strategy
                max_holding_days=max_holding,
                reasons=signal.reasons
            )
            
            trade_signals.append(trade_signal)
            capital_allocated += position_size
            positions_added += 1
            
        return trade_signals
    
    def _update_regime_history(self, regime_score, date):
        """Track regime changes over time."""
        self.regime_history.append({
            'date': date,
            'regime': regime_score.regime,
            'confidence': regime_score.confidence,
            'composite_score': regime_score.composite_score
        })
        
        # Keep only recent history (last 60 days)
        if len(self.regime_history) > 60:
            self.regime_history = self.regime_history[-60:]
            
        # Check for regime change
        if len(self.regime_history) >= 2:
            if self.regime_history[-1]['regime'] != self.regime_history[-2]['regime']:
                self.last_regime_change = date
                logger.info(f"REGIME CHANGE DETECTED: {self.regime_history[-2]['regime'].value} "
                          f"-> {self.regime_history[-1]['regime'].value}")
    
    def _check_rebalance_trigger(self, regime_score) -> bool:
        """
        Check if portfolio rebalancing is needed.
        
        Triggers:
        1. Regime change with high confidence
        2. Monthly schedule
        3. Risk limits breached
        """
        # Always rebalance on regime change with high confidence
        if self.last_regime_change and regime_score.confidence > self.min_regime_confidence:
            return True
            
        # Check other triggers (monthly, risk, etc.)
        # TODO: Implement additional triggers
        
        return False
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        query = f"""
        SELECT close
        FROM prices_daily
        WHERE symbol = '{symbol}'
        ORDER BY date DESC
        LIMIT 1
        """
        
        result = pd.read_sql_query(query, self.engine)
        if not result.empty:
            return result['close'].iloc[0]
        return None
    
    def generate_summary_report(
        self,
        trade_signals: List[TradeSignal],
        regime_score
    ) -> str:
        """
        Generate a summary report of signals and strategy.
        """
        report = []
        report.append("=" * 70)
        report.append("HYBRID STRATEGY TRADING SIGNALS REPORT")
        report.append("=" * 70)
        
        # Regime information
        report.append(f"\n📊 MARKET REGIME: {regime_score.regime.value}")
        report.append(f"   Confidence: {regime_score.confidence:.1%}")
        report.append(f"   VIX Level: {regime_score.signals.vix_level:.2f}")
        report.append(f"   SPY Momentum: {regime_score.signals.spy_momentum:+.1f}%")
        
        # Allocation
        allocation = self._calculate_allocation(regime_score)
        report.append(f"\n💼 STRATEGY ALLOCATION:")
        report.append(f"   Momentum: {allocation.momentum_weight:.0%}")
        report.append(f"   Mean Reversion: {allocation.mean_reversion_weight:.0%}")
        report.append(f"   Cash: {allocation.cash_weight:.0%}")
        report.append(f"   Gross Exposure: {allocation.gross_exposure:.0%}")
        
        # Trade signals
        buy_signals = [s for s in trade_signals if s.action == 'BUY']
        sell_signals = [s for s in trade_signals if s.action == 'SELL']
        
        report.append(f"\n📈 BUY SIGNALS ({len(buy_signals)}):")
        for i, signal in enumerate(buy_signals[:5], 1):
            report.append(f"   {i}. {signal.symbol}")
            report.append(f"      Shares: {signal.shares} @ ${signal.price:.2f}")
            report.append(f"      Size: ${signal.position_size:,.0f} ({signal.position_size_pct:.1%})")
            report.append(f"      Strategy: {signal.strategy}")
            report.append(f"      Confidence: {signal.confidence:.0%}")
            if signal.reasons:
                report.append(f"      Reasons: {signal.reasons[0]}")
                
        if sell_signals:
            report.append(f"\n📉 SELL SIGNALS ({len(sell_signals)}):")
            for i, signal in enumerate(sell_signals[:3], 1):
                report.append(f"   {i}. {signal.symbol}")
                report.append(f"      Strategy: {signal.strategy}")
                
        # Risk metrics
        total_exposure = sum(s.position_size for s in buy_signals)
        report.append(f"\n⚠️ RISK METRICS:")
        report.append(f"   Total Exposure: ${total_exposure:,.0f}")
        report.append(f"   Number of Positions: {len(buy_signals)}")
        report.append(f"   Average Position Size: ${total_exposure/len(buy_signals):,.0f}" if buy_signals else "   No positions")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)