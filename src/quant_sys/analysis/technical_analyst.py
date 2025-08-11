"""
Technical Analyst - Fixed version with proper SQLAlchemy usage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import logging
from sqlalchemy import text

from ..core.config import Settings
from ..core.storage import get_engine
from ..features.technical import (
    calculate_returns,
    calculate_rsi,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_dollar_volume,
    calculate_zscore
)
from ..features.high_quality_momentum import (
    calculate_momentum_multiperiod,
    calculate_hqm_score
)
from ..features.vol_models import (
    calculate_ewma_volatility,
    calculate_garman_klass_volatility,
    calculate_realized_volatility
)

logger = logging.getLogger(__name__)


class TechnicalAnalyst:
    """Coordinates technical analysis and feature engineering."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize with configuration."""
        self.settings = settings or Settings()
        self.engine = get_engine(self.settings.paths.db_path)
        self.db_path = Path(self.settings.paths.db_path)
        
        # Create technical_indicators table if it doesn't exist
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Create technical_indicators table if it doesn't exist."""
        # Use text() wrapper for SQLAlchemy
        create_table_sql = text("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                indicator_name TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (date, symbol, indicator_name)
            )
        """)
        
        with self.engine.connect() as conn:
            conn.execute(create_table_sql)
            conn.commit()
    
    def calculate_features_for_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate all technical features for a single symbol.
        """
        # Fetch price data
        query = """
        SELECT date, open, high, low, close, volume
        FROM prices_daily
        WHERE symbol = :symbol
        """
        
        params = {'symbol': symbol}
        if start_date:
            query += " AND date >= :start_date"
            params['start_date'] = start_date
        if end_date:
            query += " AND date <= :end_date"
            params['end_date'] = end_date
            
        query += " ORDER BY date"
        
        df = pd.read_sql_query(
            text(query), 
            self.engine,
            params=params,
            parse_dates=['date'],
            index_col='date'
        )
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Initialize features dataframe
        features = pd.DataFrame(index=df.index)
        
        try:
            # 1. Returns (multiple periods)
            returns_df = calculate_returns(
                df['close'], 
                periods=[1, 5, 21, 63, 126, 252]
            )
            features = pd.concat([features, returns_df], axis=1)
            
            # 2. RSI
            features['rsi'] = calculate_rsi(df['close'], period=14)
            
            # 3. ATR
            features['atr'] = calculate_atr(
                df['high'], df['low'], df['close'], period=14
            )
            features['atr_pct'] = features['atr'] / df['close']
            
            # 4. Bollinger Bands
            bb_df = calculate_bollinger_bands(df['close'], period=20)
            features = pd.concat([features, bb_df], axis=1)
            
            # 5. MACD
            macd_df = calculate_macd(df['close'])
            features = pd.concat([features, macd_df], axis=1)
            
            # 6. Dollar Volume
            if 'volume' in df.columns:
                dv_df = calculate_dollar_volume(df['close'], df['volume'])
                features = pd.concat([features, dv_df], axis=1)
            
            # 7. Z-scores for mean reversion
            for period in [5, 21]:
                col_name = f'return_{period}d'
                if col_name in features.columns:
                    features[f'zscore_{period}d'] = calculate_zscore(
                        features[col_name], lookback=20
                    )
            
            # 8. High-Quality Momentum
            momentum_df = calculate_momentum_multiperiod(
                df['close'],
                periods=[21, 63, 126, 252]
            )
            features = pd.concat([features, momentum_df], axis=1)
            
            # IMPORTANT: Also create momentum_XXXd columns from return_XXXd
            # This ensures we have both naming conventions
            for period in [21, 63, 126, 252]:
                return_col = f'return_{period}d'
                momentum_col = f'momentum_{period}d'
                
                # If we have the return but not the momentum, create it
                if return_col in features.columns and momentum_col not in features.columns:
                    features[momentum_col] = features[return_col]
            
            # 9. Volatility measures
            if 'return_1d' in features.columns:
                features['realized_vol_21d'] = calculate_realized_volatility(
                    features['return_1d'], window=21
                )
                features['ewma_vol'] = calculate_ewma_volatility(
                    features['return_1d'], lambda_=0.94
                )
            
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                features['gk_vol'] = calculate_garman_klass_volatility(
                    df['high'], df['low'], df['close'], df['open']
                )
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}")
            
        # Add symbol column
        features['symbol'] = symbol
        
        return features
    
    def batch_calculate_features(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_to_db: bool = True
    ) -> pd.DataFrame:
        """
        Calculate features for multiple symbols.
        """
        all_features = []
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Calculating features for {symbol} ({i}/{len(symbols)})")
            print(f"Processing {symbol} ({i}/{len(symbols)})...")
            
            try:
                features = self.calculate_features_for_symbol(
                    symbol, start_date, end_date
                )
                
                if not features.empty:
                    all_features.append(features)
                    
                    if save_to_db:
                        self._save_features_to_db(features, symbol)
                        
            except Exception as e:
                logger.error(f"Error calculating features for {symbol}: {e}")
                print(f"  Error: {e}")
                continue
        
        if all_features:
            combined = pd.concat(all_features, axis=0)
            return combined
        
        return pd.DataFrame()
    
    def _save_features_to_db(self, features: pd.DataFrame, symbol: str):
        """Save calculated features to database."""
        # Reshape from wide to long format for database storage
        records = []
        
        for date in features.index:
            for col in features.columns:
                if col == 'symbol':
                    continue
                    
                value = features.loc[date, col]
                
                # Skip NaN values
                if pd.isna(value):
                    continue
                
                records.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'indicator_name': col,
                    'value': float(value)
                })
        
        if records:
            # Use raw SQL for efficiency with sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if not exists (for sqlite3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    indicator_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (date, symbol, indicator_name)
                )
            """)
            
            # Delete existing records for this symbol and date range
            if records:
                min_date = min(r['date'] for r in records)
                max_date = max(r['date'] for r in records)
                
                cursor.execute("""
                    DELETE FROM technical_indicators
                    WHERE symbol = ? AND date BETWEEN ? AND ?
                """, (symbol, min_date, max_date))
            
            # Insert new records
            cursor.executemany("""
                INSERT OR REPLACE INTO technical_indicators 
                (date, symbol, indicator_name, value)
                VALUES (?, ?, ?, ?)
            """, [(r['date'], r['symbol'], r['indicator_name'], r['value']) 
                  for r in records])
            
            conn.commit()
            conn.close()
            
            print(f"  Saved {len(records)} feature records for {symbol}")
    
    def calculate_cross_sectional_features(
        self,
        date: str,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate cross-sectional features (percentiles, ranks) for a date.
        """
        # Build query with proper parameter binding
        base_query = """
        SELECT symbol, indicator_name, value
        FROM technical_indicators
        WHERE date = :date
        """
        
        params = {'date': date}
        
        if symbols:
            # Use IN clause with named parameters
            symbol_params = {f'symbol_{i}': s for i, s in enumerate(symbols)}
            placeholders = ', '.join(f':{k}' for k in symbol_params.keys())
            base_query += f" AND symbol IN ({placeholders})"
            params.update(symbol_params)
        
        df = pd.read_sql_query(text(base_query), self.engine, params=params)
        
        if df.empty:
            logger.warning(f"No features found for {date}")
            return pd.DataFrame()
        
        # Pivot to wide format
        pivot = df.pivot(index='symbol', columns='indicator_name', values='value')
        
        # Calculate percentiles for momentum indicators
        momentum_cols = [col for col in pivot.columns 
                        if any(x in col for x in ['momentum', 'return'])]
        
        for col in momentum_cols:
            if col in pivot.columns:
                pivot[f'{col}_percentile'] = pivot[col].rank(pct=True) * 100
        
        # Calculate HQM scores
        hqm_scores = []
        percentile_cols = [col for col in pivot.columns if 'percentile' in col]
        
        for symbol in pivot.index:
            score_dict = {
                col: pivot.loc[symbol, col] 
                for col in percentile_cols
                if pd.notna(pivot.loc[symbol, col])
            }
            
            if score_dict:
                hqm_scores.append(calculate_hqm_score(score_dict))
            else:
                hqm_scores.append(np.nan)
        
        pivot['hqm_score'] = hqm_scores
        pivot['hqm_rank'] = pivot['hqm_score'].rank(ascending=False, method='dense')
        
        # Add date column
        pivot['date'] = date
        
        return pivot.reset_index()
    
    def generate_feature_quality_report(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Generate quality report for calculated features.
        """
        # Get feature coverage
        base_query = """
        SELECT 
            symbol,
            indicator_name,
            COUNT(*) as count,
            MIN(date) as first_date,
            MAX(date) as last_date,
            AVG(value) as mean_value,
            COUNT(DISTINCT date) as unique_dates
        FROM technical_indicators
        WHERE 1=1
        """
        
        params = {}
        
        if symbols:
            symbol_params = {f'symbol_{i}': s for i, s in enumerate(symbols)}
            placeholders = ', '.join(f':{k}' for k in symbol_params.keys())
            base_query += f" AND symbol IN ({placeholders})"
            params.update(symbol_params)
        
        if start_date:
            base_query += " AND date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            base_query += " AND date <= :end_date"
            params['end_date'] = end_date
        
        base_query += " GROUP BY symbol, indicator_name"
        
        coverage_df = pd.read_sql_query(text(base_query), self.engine, params=params)
        
        # Calculate quality metrics
        report = {
            'total_features': coverage_df['indicator_name'].nunique() if not coverage_df.empty else 0,
            'total_symbols': coverage_df['symbol'].nunique() if not coverage_df.empty else 0,
            'total_records': int(coverage_df['count'].sum()) if not coverage_df.empty else 0,
            'date_range': {
                'first': coverage_df['first_date'].min() if not coverage_df.empty else None,
                'last': coverage_df['last_date'].max() if not coverage_df.empty else None
            },
            'coverage_by_feature': {},
            'quality_checks': {}
        }
        
        if not coverage_df.empty:
            # Coverage by feature type
            for feature in coverage_df['indicator_name'].unique():
                feature_data = coverage_df[coverage_df['indicator_name'] == feature]
                report['coverage_by_feature'][feature] = {
                    'symbols_covered': int(feature_data['symbol'].nunique()),
                    'avg_observations': float(feature_data['count'].mean()),
                    'completeness': float(feature_data['symbol'].nunique() / report['total_symbols']) if report['total_symbols'] > 0 else 0
                }
            
            # Quality checks
            # Check for reasonable RSI values
            rsi_check = coverage_df[coverage_df['indicator_name'] == 'rsi']
            if not rsi_check.empty:
                report['quality_checks']['rsi_range'] = {
                    'min': float(rsi_check['mean_value'].min()),
                    'max': float(rsi_check['mean_value'].max()),
                    'valid': (0 <= rsi_check['mean_value'].min()) and 
                            (rsi_check['mean_value'].max() <= 100)
                }
            
            # Check for reasonable volatility values
            vol_features = coverage_df[
                coverage_df['indicator_name'].str.contains('vol')
            ]
            if not vol_features.empty:
                report['quality_checks']['volatility_range'] = {
                    'min': float(vol_features['mean_value'].min()),
                    'max': float(vol_features['mean_value'].max()),
                    'valid': (0 <= vol_features['mean_value'].min()) and 
                            (vol_features['mean_value'].max() <= 2.0)  # 200% annualized
                }
        
        return report
    
    def get_latest_features(
        self,
        symbols: List[str],
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get the most recent features for given symbols.
        """
        # Build params dict
        symbol_params = {f'symbol_{i}': s for i, s in enumerate(symbols)}
        placeholders = ', '.join(f':{k}' for k in symbol_params.keys())
        
        query = f"""
        WITH latest_dates AS (
            SELECT symbol, MAX(date) as max_date
            FROM technical_indicators
            WHERE symbol IN ({placeholders})
            GROUP BY symbol
        )
        SELECT ti.symbol, ti.date, ti.indicator_name, ti.value
        FROM technical_indicators ti
        JOIN latest_dates ld 
            ON ti.symbol = ld.symbol 
            AND ti.date = ld.max_date
        """
        
        params = symbol_params.copy()
        
        if features:
            feature_params = {f'feature_{i}': f for i, f in enumerate(features)}
            feature_placeholders = ', '.join(f':{k}' for k in feature_params.keys())
            query += f" WHERE ti.indicator_name IN ({feature_placeholders})"
            params.update(feature_params)
        
        df = pd.read_sql_query(text(query), self.engine, params=params)
        
        if df.empty:
            return pd.DataFrame()
        
        # Pivot to wide format
        pivot = df.pivot(
            index='symbol', 
            columns='indicator_name', 
            values='value'
        )
        
        return pivot