from __future__ import annotations

def equity_commission(shares: int, commission_per_share: float = 0.005) -> float:
    return abs(shares) * commission_per_share

def equity_slippage(notional: float, slippage_bps: int = 8) -> float:
    return abs(notional) * (slippage_bps / 10_000)
