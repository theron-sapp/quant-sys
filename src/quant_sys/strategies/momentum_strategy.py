"""
Momentum trading strategy implementation.
Location: src/quant_sys/strategies/momentum_strategy.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ..signals.momentum_signals import MomentumSignalGenerator, MomentumSignal
from ..analysis.regime_detector import MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class MomentumPosition:
    """Momentum strategy position."""
    symbol: str
    position_type: str  # 'long' or 'short'
    entry_date: str
    entry_price: float
    current_price: float
    position_size: float  # Dollar amount
    shares: int
    
    # Performance metrics
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Risk metrics
    stop_loss_price: float
    trailing_stop_pct: float
    max_gain_pct: float  # For trailing stop calculation
    
    # Signal metrics
    hqm_score: float
    signal_strength: float
    
    # Exit conditions
    days_held: int
    exit_signal: bool
    exit_reason: str


class MomentumStrategy:
    """
    Momentum-based trading strategy.
    
    Focuses on stocks with strong price momentum,
    particularly effective in growth market regimes.
    """
    
    def __init__(self, engine, config):
        """
        Initialize momentum strategy.
        
        Args:
            engine: SQLAlchemy engine
            config: Settings object
        """
        self.engine = engine
        self.config = config
        
        # Initialize signal generator
        self.signal_generator = MomentumSignalGenerator(engine, config)
        
        # Strategy parameters
        self.max_positions = 20
        self.position_size_pct = 0.05  # 5% per position
        self.stop_loss_pct = 0.15  # 15% stop loss
        self.trailing_stop_pct = 0.10  # 10% trailing stop
        self.min_hqm_score = 70  # Minimum HQM score for entry
        self.max_holding_period = 63  # Maximum 3 months
        
    def generate_portfolio(
        self,
        capital: float,
        current_positions: Dict[str, MomentumPosition],
        date: Optional[str] = None,
        regime: Optional[MarketRegime] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate momentum portfolio with entry and exit signals.
        
        Args:
            capital: Total capital available
            current_positions: Current open positions
            date: Date for signal generation
            regime: Current market regime
            
        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        logger.info(f"Generating momentum portfolio for capital: ${capital:,.0f}")
        
        # Check if momentum strategy should be active
        if not self._should_trade(regime):
            logger.info(f"Momentum strategy inactive in regime: {regime.value if regime else 'Unknown'}")
            return [], self._generate_all_exits(current_positions, "Regime unfavorable")
            
        # Generate momentum signals
        signals = self.signal_generator.generate_signals(date=date, top_n=30)
        
        # Filter signals by quality
        filtered_signals = self._filter_signals(signals, regime)
        
        # Check existing positions for exits
        exit_signals = self._check_exits(current_positions, signals, date)
        
        # Generate new entries
        available_capital = self._calculate_available_capital(
            capital,
            current_positions,
            exit_signals
        )
        entry_signals = self._generate_entries(
            filtered_signals,
            available_capital,
            current_positions
        )
        
        return entry_signals, exit_signals
    
    def _should_trade(self, regime: Optional[MarketRegime]) -> bool:
        """
        Determine if momentum strategy should be active.
        
        Momentum works best in growth regimes with low volatility.
        """
        if regime is None:
            return True  # Default to active if regime unknown
            
        favorable_regimes = [
            MarketRegime.STRONG_GROWTH,
            MarketRegime.GROWTH,
            MarketRegime.NEUTRAL
        ]
        
        return regime in favorable_regimes
    
    def _filter_signals(
        self,
        signals: Dict[str, MomentumSignal],
        regime: Optional[MarketRegime]
    ) -> Dict[str, MomentumSignal]:
        """
        Filter momentum signals based on quality criteria.
        """
        filtered = {}
        
        for symbol, signal in signals.items():
            # Check signal type
            if signal.signal_type != 'long':
                continue  # Momentum strategy is long-only in most regimes
                
            # Check HQM score
            if signal.hqm_score < self.min_hqm_score:
                continue
                
            # Check signal strength
            if signal.signal_strength < 0.5:
                continue
                
            # Additional filters for defensive regimes
            if regime in [MarketRegime.DIVIDEND, MarketRegime.CRISIS]:
                # In defensive regimes, require higher quality
                if signal.hqm_score < 80:
                    continue
                if signal.rsi > 70:  # Avoid overbought
                    continue
                    
            filtered[symbol] = signal
            
        return filtered
    
    def _check_exits(
        self,
        current_positions: Dict[str, MomentumPosition],
        new_signals: Dict[str, MomentumSignal],
        date: str
    ) -> List[Dict]:
        """
        Check existing positions for exit conditions.
        
        Exit conditions:
        1. Stop loss hit
        2. Trailing stop triggered
        3. Momentum deterioration
        4. Maximum holding period reached
        5. Signal reversal
        """
        exit_signals = []
        
        for symbol, position in current_positions.items():
            exit_reason = None
            exit_price = position.current_price
            
            # Update position metrics
            position = self._update_position_metrics(position, date)
            
            # Check stop loss
            if position.current_price <= position.stop_loss_price:
                exit_reason = "Stop loss triggered"
                
            # Check trailing stop
            elif position.max_gain_pct > 0.10:  # If gain > 10%, activate trailing stop
                trailing_stop = position.entry_price * (1 + position.max_gain_pct - self.trailing_stop_pct)
                if position.current_price <= trailing_stop:
                    exit_reason = f"Trailing stop triggered (max gain: {position.max_gain_pct:.1%})"
                    
            # Check holding period
            elif position.days_held >= self.max_holding_period:
                exit_reason = f"Max holding period reached ({position.days_held} days)"
                
            # Check momentum deterioration
            elif symbol in new_signals:
                new_signal = new_signals[symbol]
                
                # Exit if momentum has reversed
                if new_signal.signal_type == 'short':
                    exit_reason = "Momentum reversed to short"
                    
                # Exit if HQM score dropped significantly
                elif new_signal.hqm_score < position.hqm_score - 20:
                    exit_reason = f"HQM deteriorated: {position.hqm_score:.0f} -> {new_signal.hqm_score:.0f}"
                    
                # Exit if signal strength weakened
                elif new_signal.signal_strength < 0.3:
                    exit_reason = "Signal strength weakened"
                    
            # Generate exit signal if needed
            if exit_reason:
                exit_signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': position.shares,
                    'exit_price': exit_price,
                    'entry_price': position.entry_price,
                    'pnl': position.unrealized_pnl,
                    'pnl_pct': position.unrealized_pnl_pct,
                    'days_held': position.days_held,
                    'reason': exit_reason
                })
                
        return exit_signals
    
    def _generate_entries(
        self,
        signals: Dict[str, MomentumSignal],
        available_capital: float,
        current_positions: Dict[str, MomentumPosition]
    ) -> List[Dict]:
        """
        Generate entry signals for new positions.
        """
        entry_signals = []
        
        # Sort signals by HQM score
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1].hqm_score,
            reverse=True
        )
        
        # Calculate position size
        position_size = min(
            available_capital * self.position_size_pct,
            available_capital / (self.max_positions - len(current_positions))
        )
        
        # Generate entries
        positions_to_add = self.max_positions - len(current_positions)
        capital_used = 0
        
        for symbol, signal in sorted_signals:
            # Skip if already have position
            if symbol in current_positions:
                continue
                
            # Check capital constraint
            if capital_used + position_size > available_capital:
                break
                
            # Check position limit
            if len(entry_signals) >= positions_to_add:
                break
                
            # Get current price
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue
                
            # Calculate shares
            shares = int(position_size / current_price)
            if shares == 0:
                continue
                
            # Generate entry signal
            entry_signals.append({
                'symbol': symbol,
                'action': 'BUY',
                'shares': shares,
                'price': current_price,
                'position_size': shares * current_price,
                'position_size_pct': position_size / available_capital,
                'stop_loss': current_price * (1 - self.stop_loss_pct),
                'hqm_score': signal.hqm_score,
                'signal_strength': signal.signal_strength,
                'momentum_12m': signal.momentum_12m,
                'momentum_3m': signal.momentum_3m,
                'reasons': signal.reasons
            })
            
            capital_used += shares * current_price
            
        return entry_signals
    
    def _update_position_metrics(
        self,
        position: MomentumPosition,
        date: str
    ) -> MomentumPosition:
        """Update position with current metrics."""
        # Get current price
        current_price = self._get_current_price(position.symbol)
        if current_price:
            position.current_price = current_price
            
            # Update P&L
            position.unrealized_pnl = (current_price - position.entry_price) * position.shares
            position.unrealized_pnl_pct = (current_price / position.entry_price - 1)
            
            # Update max gain for trailing stop
            position.max_gain_pct = max(
                position.max_gain_pct,
                position.unrealized_pnl_pct
            )
            
        # Update days held
        entry_date = pd.to_datetime(position.entry_date)
        current_date = pd.to_datetime(date)
        position.days_held = (current_date - entry_date).days
        
        return position
    
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
    
    def _calculate_available_capital(
        self,
        total_capital: float,
        current_positions: Dict[str, MomentumPosition],
        exit_signals: List[Dict]
    ) -> float:
        """Calculate capital available for new positions."""
        # Start with total capital
        available = total_capital
        
        # Subtract current positions (that aren't being exited)
        exiting_symbols = [sig['symbol'] for sig in exit_signals]
        for symbol, position in current_positions.items():
            if symbol not in exiting_symbols:
                available -= position.position_size
                
        # Add back capital from exits
        for exit_sig in exit_signals:
            if exit_sig['symbol'] in current_positions:
                position = current_positions[exit_sig['symbol']]
                available += position.shares * exit_sig['exit_price']
                
        return max(available, 0)
    
    def _generate_all_exits(
        self,
        current_positions: Dict[str, MomentumPosition],
        reason: str
    ) -> List[Dict]:
        """Generate exit signals for all positions."""
        exit_signals = []
        
        for symbol, position in current_positions.items():
            exit_signals.append({
                'symbol': symbol,
                'action': 'SELL',
                'shares': position.shares,
                'exit_price': position.current_price,
                'entry_price': position.entry_price,
                'pnl': position.unrealized_pnl,
                'pnl_pct': position.unrealized_pnl_pct,
                'days_held': position.days_held,
                'reason': reason
            })
            
        return exit_signals
    
    def calculate_strategy_metrics(
        self,
        positions: Dict[str, MomentumPosition]
    ) -> Dict:
        """
        Calculate momentum strategy performance metrics.
        """
        if not positions:
            return {
                'total_positions': 0,
                'total_value': 0,
                'total_pnl': 0,
                'avg_pnl_pct': 0,
                'win_rate': 0,
                'avg_hqm_score': 0
            }
            
        # Calculate metrics
        total_value = sum(p.position_size for p in positions.values())
        total_pnl = sum(p.unrealized_pnl for p in positions.values())
        
        pnl_pcts = [p.unrealized_pnl_pct for p in positions.values()]
        avg_pnl_pct = np.mean(pnl_pcts) if pnl_pcts else 0
        
        winners = sum(1 for p in pnl_pcts if p > 0)
        win_rate = winners / len(pnl_pcts) if pnl_pcts else 0
        
        avg_hqm = np.mean([p.hqm_score for p in positions.values()])
        
        return {
            'total_positions': len(positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl / total_value if total_value > 0 else 0,
            'avg_pnl_pct': avg_pnl_pct,
            'win_rate': win_rate,
            'avg_hqm_score': avg_hqm,
            'best_performer': max(positions.items(), key=lambda x: x[1].unrealized_pnl_pct)[0] if positions else None,
            'worst_performer': min(positions.items(), key=lambda x: x[1].unrealized_pnl_pct)[0] if positions else None
        }