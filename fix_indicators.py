"""
Script to diagnose and fix missing indicators (hqm_score and momentum_252d).
Save as: fix_indicators.py
Run with: python fix_indicators.py
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def diagnose_indicators():
    """Check what indicators we actually have in the database."""
    
    conn = sqlite3.connect('db/quant.sqlite')
    
    print("=" * 70)
    print("INDICATOR DIAGNOSTIC")
    print("=" * 70)
    
    # 1. Check all unique indicator names
    print("\n1. ALL AVAILABLE INDICATORS:")
    query = """
    SELECT DISTINCT indicator_name
    FROM technical_indicators
    ORDER BY indicator_name
    """
    indicators = pd.read_sql_query(query, conn)
    
    for ind in indicators['indicator_name']:
        print(f"   - {ind}")
    
    # 2. Check for momentum indicators specifically
    print("\n2. MOMENTUM INDICATORS:")
    momentum_indicators = [ind for ind in indicators['indicator_name'] if 'momentum' in ind.lower()]
    
    if momentum_indicators:
        for ind in momentum_indicators:
            # Count how many symbols have this indicator
            count_query = f"""
            SELECT COUNT(DISTINCT symbol) as count
            FROM technical_indicators
            WHERE indicator_name = '{ind}'
            """
            count = pd.read_sql_query(count_query, conn)['count'].iloc[0]
            print(f"   - {ind}: {count} symbols")
    else:
        print("   ❌ No momentum indicators found!")
    
    # 3. Check what periods of returns we have
    print("\n3. RETURN PERIODS AVAILABLE:")
    return_indicators = [ind for ind in indicators['indicator_name'] if ind.startswith('return_')]
    
    for ind in return_indicators:
        count_query = f"""
        SELECT COUNT(DISTINCT symbol) as count
        FROM technical_indicators
        WHERE indicator_name = '{ind}'
        """
        count = pd.read_sql_query(count_query, conn)['count'].iloc[0]
        print(f"   - {ind}: {count} symbols")
    
    # 4. Check for HQM-related indicators
    print("\n4. HQM-RELATED INDICATORS:")
    hqm_indicators = [ind for ind in indicators['indicator_name'] if 'hqm' in ind.lower() or 'percentile' in ind.lower()]
    
    if hqm_indicators:
        for ind in hqm_indicators:
            print(f"   - {ind}")
    else:
        print("   ❌ No HQM indicators found!")
    
    conn.close()
    
    return indicators['indicator_name'].tolist()


def calculate_missing_momentum():
    """Calculate momentum_252d from return_252d if it exists."""
    
    conn = sqlite3.connect('db/quant.sqlite')
    
    print("\n" + "=" * 70)
    print("FIXING MOMENTUM_252D")
    print("=" * 70)
    
    # Check if we have return_252d
    query = """
    SELECT COUNT(*) as count
    FROM technical_indicators
    WHERE indicator_name = 'return_252d'
    """
    result = pd.read_sql_query(query, conn)
    
    if result['count'].iloc[0] > 0:
        print(f"✅ Found {result['count'].iloc[0]} return_252d records")
        
        # Copy return_252d to momentum_252d
        insert_query = """
        INSERT OR REPLACE INTO technical_indicators (date, symbol, indicator_name, value)
        SELECT date, symbol, 'momentum_252d' as indicator_name, value
        FROM technical_indicators
        WHERE indicator_name = 'return_252d'
        """
        
        cursor = conn.cursor()
        cursor.execute(insert_query)
        conn.commit()
        
        # Verify
        verify_query = """
        SELECT COUNT(*) as count
        FROM technical_indicators
        WHERE indicator_name = 'momentum_252d'
        """
        verify_result = pd.read_sql_query(verify_query, conn)
        print(f"✅ Created {verify_result['count'].iloc[0]} momentum_252d records")
        
        # Also create other momentum indicators from returns
        for period in [21, 63, 126]:
            return_col = f'return_{period}d'
            momentum_col = f'momentum_{period}d'
            
            # Check if return exists but momentum doesn't
            check_query = f"""
            SELECT 
                (SELECT COUNT(*) FROM technical_indicators WHERE indicator_name = '{return_col}') as return_count,
                (SELECT COUNT(*) FROM technical_indicators WHERE indicator_name = '{momentum_col}') as momentum_count
            """
            check_result = pd.read_sql_query(check_query, conn)
            
            if check_result['return_count'].iloc[0] > 0 and check_result['momentum_count'].iloc[0] == 0:
                print(f"\nCreating {momentum_col} from {return_col}...")
                
                insert_query = f"""
                INSERT OR REPLACE INTO technical_indicators (date, symbol, indicator_name, value)
                SELECT date, symbol, '{momentum_col}' as indicator_name, value
                FROM technical_indicators
                WHERE indicator_name = '{return_col}'
                """
                
                cursor.execute(insert_query)
                conn.commit()
                print(f"✅ Created {momentum_col}")
    else:
        print("❌ No return_252d data found. Need to recalculate features.")
        print("   Run: quant calculate-features --universe --top-n 50")
    
    conn.close()


def calculate_hqm_scores():
    """Calculate HQM scores from existing momentum percentiles."""
    
    conn = sqlite3.connect('db/quant.sqlite')
    
    print("\n" + "=" * 70)
    print("CALCULATING HQM SCORES")
    print("=" * 70)
    
    # Get all unique dates
    query = """
    SELECT DISTINCT date
    FROM technical_indicators
    WHERE indicator_name IN ('momentum_21d', 'momentum_63d', 'momentum_126d', 'momentum_252d')
    ORDER BY date DESC
    LIMIT 100
    """
    dates = pd.read_sql_query(query, conn)
    
    if dates.empty:
        print("❌ No momentum data found to calculate HQM scores")
        return
    
    print(f"Processing {len(dates)} dates...")
    
    records_to_insert = []
    
    for date_val in dates['date']:
        # Get momentum data for this date
        momentum_query = f"""
        SELECT 
            symbol,
            MAX(CASE WHEN indicator_name = 'momentum_21d' THEN value END) as momentum_21d,
            MAX(CASE WHEN indicator_name = 'momentum_63d' THEN value END) as momentum_63d,
            MAX(CASE WHEN indicator_name = 'momentum_126d' THEN value END) as momentum_126d,
            MAX(CASE WHEN indicator_name = 'momentum_252d' THEN value END) as momentum_252d
        FROM technical_indicators
        WHERE date = '{date_val}'
        AND indicator_name IN ('momentum_21d', 'momentum_63d', 'momentum_126d', 'momentum_252d')
        GROUP BY symbol
        HAVING momentum_21d IS NOT NULL 
           AND momentum_63d IS NOT NULL 
           AND momentum_126d IS NOT NULL 
           AND momentum_252d IS NOT NULL
        """
        
        momentum_df = pd.read_sql_query(momentum_query, conn)
        
        if momentum_df.empty:
            continue
        
        # Calculate percentiles for each momentum period
        for col in ['momentum_21d', 'momentum_63d', 'momentum_126d', 'momentum_252d']:
            momentum_df[f'{col}_percentile'] = momentum_df[col].rank(pct=True) * 100
        
        # Calculate HQM score as average of percentiles
        percentile_cols = [col for col in momentum_df.columns if 'percentile' in col]
        momentum_df['hqm_score'] = momentum_df[percentile_cols].mean(axis=1)
        
        # Prepare records for insertion
        for _, row in momentum_df.iterrows():
            records_to_insert.append((
                date_val,
                row['symbol'],
                'hqm_score',
                float(row['hqm_score'])
            ))
            
            # Also save the percentiles
            for col in percentile_cols:
                records_to_insert.append((
                    date_val,
                    row['symbol'],
                    col,
                    float(row[col])
                ))
    
    # Insert all records
    if records_to_insert:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO technical_indicators (date, symbol, indicator_name, value)
            VALUES (?, ?, ?, ?)
        """, records_to_insert)
        conn.commit()
        
        print(f"✅ Inserted {len(records_to_insert)} HQM-related records")
        
        # Verify HQM scores
        verify_query = """
        SELECT COUNT(DISTINCT symbol) as symbols, COUNT(*) as total
        FROM technical_indicators
        WHERE indicator_name = 'hqm_score'
        """
        verify_result = pd.read_sql_query(verify_query, conn)
        print(f"✅ HQM scores calculated for {verify_result['symbols'].iloc[0]} symbols "
              f"({verify_result['total'].iloc[0]} total records)")
    else:
        print("❌ Could not calculate HQM scores - insufficient momentum data")
    
    conn.close()


def verify_fixes():
    """Verify that the required indicators now exist."""
    
    conn = sqlite3.connect('db/quant.sqlite')
    
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    required = ['hqm_score', 'momentum_252d', 'momentum_63d', 'rsi', 'ewma_vol']
    
    for indicator in required:
        query = f"""
        SELECT 
            COUNT(DISTINCT symbol) as symbols,
            COUNT(*) as records,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM technical_indicators
        WHERE indicator_name = '{indicator}'
        """
        result = pd.read_sql_query(query, conn)
        
        if result['records'].iloc[0] > 0:
            print(f"✅ {indicator}: {result['symbols'].iloc[0]} symbols, "
                  f"{result['records'].iloc[0]} records "
                  f"({result['min_date'].iloc[0]} to {result['max_date'].iloc[0]})")
        else:
            print(f"❌ {indicator}: NOT FOUND")
    
    # Show sample HQM scores
    print("\n5 SAMPLE HQM SCORES (Latest Date):")
    sample_query = """
    SELECT symbol, value as hqm_score
    FROM technical_indicators
    WHERE indicator_name = 'hqm_score'
    AND date = (SELECT MAX(date) FROM technical_indicators WHERE indicator_name = 'hqm_score')
    ORDER BY value DESC
    LIMIT 5
    """
    
    sample = pd.read_sql_query(sample_query, conn)
    if not sample.empty:
        for _, row in sample.iterrows():
            print(f"   {row['symbol']}: {row['hqm_score']:.1f}")
    
    conn.close()


if __name__ == "__main__":
    print("FIXING MISSING INDICATORS FOR SIGNAL GENERATION")
    print("=" * 70)
    
    # Step 1: Diagnose what we have
    available_indicators = diagnose_indicators()
    
    # Step 2: Fix momentum_252d (copy from return_252d if it exists)
    calculate_missing_momentum()
    
    # Step 3: Calculate HQM scores
    calculate_hqm_scores()
    
    # Step 4: Verify everything is fixed
    verify_fixes()
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run: quant test-signals")
    print("2. If all indicators show ✅, run: quant generate-signals")
    print("\nIf indicators are still missing:")
    print("  Run: quant calculate-features --universe --top-n 50")