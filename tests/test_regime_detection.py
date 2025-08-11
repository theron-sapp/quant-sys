"""
Test script for regime detection functionality
Save as: test_regime_detection.py in repo root
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, 'src')

from quant_sys.core.config import load_settings
from quant_sys.core.storage import get_engine
from quant_sys.analysis.regime_detector import RegimeDetector, MarketRegime

def test_regime_detection():
    """Test regime detection on various dates"""
    
    print("Testing Regime Detection System")
    print("=" * 50)
    
    # Load config and get engine
    settings = load_settings()
    engine = get_engine(settings.paths.db_path)
    
    # Initialize detector
    detector = RegimeDetector(engine)
    
    # Test dates - these should cover different market regimes
    test_dates = [
        "2024-01-15",  # Recent date
        "2023-10-27",  # Market correction period
        "2023-07-15",  # Summer rally
        "2023-03-15",  # Banking crisis
        "2022-06-15",  # Bear market
        "2022-01-15",  # Start of 2022 decline
        "2021-11-15",  # Peak growth period
        "2020-03-23",  # COVID crash bottom
        "2020-02-19",  # Pre-COVID peak
    ]
    
    results = []
    
    for date in test_dates:
        try:
            print(f"\nTesting date: {date}")
            regime_score = detector.detect_current_regime(date)
            
            result = {
                'date': date,
                'regime': regime_score.regime.value,
                'confidence': regime_score.confidence,
                'composite_score': regime_score.composite_score,
                'vix': regime_score.signals.vix_level,
                'vix_signal': regime_score.signals.vix_signal,
                'momentum': regime_score.signals.spy_momentum,
                'rsi': regime_score.signals.spy_rsi
            }
            results.append(result)
            
            # Get allocation
            allocation = detector.get_regime_allocation(regime_score.regime)
            risk_scale = detector.get_risk_scaling(regime_score.regime)
            
            print(f"  Regime: {regime_score.regime.value}")
            print(f"  Confidence: {regime_score.confidence:.1%}")
            print(f"  VIX: {regime_score.signals.vix_level:.2f}")
            print(f"  Allocation: Dividend={allocation['dividend']:.0%}, Growth={allocation['growth']:.0%}")
            print(f"  Risk Scaling: {risk_scale:.0%}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            
    # Create summary DataFrame
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 50)
        print("SUMMARY TABLE")
        print("=" * 50)
        print(df.to_string(index=False))
        
        # Check regime persistence
        print("\n" + "=" * 50)
        print("REGIME PERSISTENCE CHECK")
        print("=" * 50)
        
        # Simulate continuous detection over a period
        print("\nTesting regime persistence over 10 days...")
        base_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
        
        for i in range(10):
            test_date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                regime_score = detector.detect_current_regime(test_date)
                print(f"  Day {i}: {regime_score.regime.value} (confidence: {regime_score.confidence:.1%})")
            except:
                print(f"  Day {i}: No data")
                
        # Show final regime history summary
        print("\n" + "=" * 50)
        print("REGIME HISTORY SUMMARY")
        print("=" * 50)
        history_df = detector.get_regime_summary()
        if not history_df.empty:
            print(history_df.tail(10).to_string(index=False))

def test_edge_cases():
    """Test edge cases and error handling"""
    
    print("\n" + "=" * 50)
    print("EDGE CASE TESTS")
    print("=" * 50)
    
    settings = load_settings()
    engine = get_engine(settings.paths.db_path)
    detector = RegimeDetector(engine)
    
    # Test with future date (should use most recent available)
    print("\n1. Testing future date...")
    try:
        future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        regime_score = detector.detect_current_regime(future_date)
        print(f"   ✅ Handled future date: {regime_score.regime.value}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        
    # Test with very old date
    print("\n2. Testing very old date...")
    try:
        old_date = "2005-01-01"
        regime_score = detector.detect_current_regime(old_date)
        print(f"   ✅ Handled old date: {regime_score.regime.value}")
    except Exception as e:
        print(f"   ⚠️ Expected failure for insufficient data: {e}")
        
    # Test regime transitions
    print("\n3. Testing regime transition thresholds...")
    test_vix_levels = [10, 15, 20, 25, 30, 35, 40, 50]
    
    for vix in test_vix_levels:
        signal = detector._calculate_vix_signal(vix)
        print(f"   VIX={vix:2d} → Signal={signal:+.2f}")

if __name__ == "__main__":
    print("REGIME DETECTION TEST SUITE")
    print("=" * 50)
    
    try:
        # Run main tests
        test_regime_detection()
        
        # Run edge case tests
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        print("\nMake sure you have run:")
        print("  1. quant ingest --top-n 50")
        print("  2. quant transform")
        import traceback
        traceback.print_exc()