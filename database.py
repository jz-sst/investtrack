"""
Database module for the AI Stock Analysis Bot
Handles PostgreSQL database operations through database service
"""

import pandas as pd
import json
import logging
from datetime import datetime, timedelta
import os

# Import the new database service
from database_service import db_service

class Database:
    def __init__(self, db_path='stock_analysis.db'):
        self.db_path = db_path  # Keep for backward compatibility
        self.logger = logging.getLogger(__name__)
        self.db_service = db_service
        self.use_sqlite_fallback = False
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            # Use new PostgreSQL database service
            health = self.db_service.health_check()
            if health['status'] == 'healthy':
                self.logger.info("PostgreSQL database initialized successfully")
            else:
                self.logger.error(f"Database health check failed: {health.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            # Fallback to SQLite for backward compatibility
            self.init_sqlite_fallback()
    
    def init_sqlite_fallback(self):
        """Initialize SQLite database as fallback"""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create basic tables for backward compatibility
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    period TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, period)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS company_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    info TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    period TEXT NOT NULL,
                    results TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, analysis_type, period)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("SQLite fallback database initialized successfully")
            self.use_sqlite_fallback = True
            
        except Exception as e:
            self.logger.error(f"Error initializing SQLite fallback: {str(e)}")
            self.use_sqlite_fallback = False
    
    def get_cached_data(self, ticker, period, max_age_hours=1):
        """
        Get cached stock data if available and not expired
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period
            max_age_hours (int): Maximum age in hours before considering data stale
            
        Returns:
            pd.DataFrame or None: Cached data if available and fresh
        """
        try:
            if hasattr(self, 'use_sqlite_fallback') and self.use_sqlite_fallback:
                return self.get_cached_data_sqlite(ticker, period, max_age_hours)
            else:
                # Try PostgreSQL first
                data = self.db_service.get_cached_stock_data(ticker, period, max_age_hours)
                return data if data is not None else pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error retrieving cached data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_cached_data_sqlite(self, ticker, period, max_age_hours=1):
        """SQLite fallback method for getting cached data"""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            cursor.execute('''
                SELECT data, updated_at FROM stock_data 
                WHERE ticker = ? AND period = ? AND updated_at > ?
            ''', (ticker, period, cutoff_time))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                data_json, updated_at = result
                data_dict = json.loads(data_json)
                
                # Convert back to DataFrame
                df = pd.DataFrame(data_dict)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                
                self.logger.info(f"Retrieved cached data for {ticker} ({period}) - last updated: {updated_at}")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def cache_data(self, ticker, period, data):
        """
        Cache stock data
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period
            data (pd.DataFrame): Stock data to cache
        """
        try:
            if data.empty:
                return
            
            if hasattr(self, 'use_sqlite_fallback') and self.use_sqlite_fallback:
                self.cache_data_sqlite(ticker, period, data)
            else:
                # Try PostgreSQL first
                self.db_service.cache_stock_data(ticker, period, data)
                
        except Exception as e:
            self.logger.error(f"Error caching data for {ticker}: {str(e)}")
    
    def cache_data_sqlite(self, ticker, period, data):
        """SQLite fallback method for caching data"""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert DataFrame to JSON
            data_copy = data.copy()
            data_copy.reset_index(inplace=True)
            
            # Convert datetime columns to string for JSON serialization
            for col in data_copy.columns:
                if data_copy[col].dtype == 'datetime64[ns]':
                    data_copy[col] = data_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            data_json = data_copy.to_json(orient='records')
            
            # Insert or update data
            cursor.execute('''
                INSERT OR REPLACE INTO stock_data (ticker, period, data, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (ticker, period, data_json))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cached data for {ticker} ({period}) - {len(data)} rows")
            
        except Exception as e:
            self.logger.error(f"Error caching data for {ticker}: {str(e)}")
    
    def get_cached_info(self, ticker, max_age_hours=24):
        """
        Get cached company information
        
        Args:
            ticker (str): Stock ticker symbol
            max_age_hours (int): Maximum age in hours before considering data stale
            
        Returns:
            dict or None: Cached company info if available and fresh
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            cursor.execute('''
                SELECT info, updated_at FROM company_info 
                WHERE ticker = ? AND updated_at > ?
            ''', (ticker, cutoff_time))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                info_json, updated_at = result
                info_dict = json.loads(info_json)
                
                self.logger.info(f"Retrieved cached company info for {ticker} - last updated: {updated_at}")
                return info_dict
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached company info for {ticker}: {str(e)}")
            return None
    
    def cache_info(self, ticker, info):
        """
        Cache company information
        
        Args:
            ticker (str): Stock ticker symbol
            info (dict): Company information to cache
        """
        try:
            if not info:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert to JSON
            info_json = json.dumps(info, default=str)
            
            # Insert or update info
            cursor.execute('''
                INSERT OR REPLACE INTO company_info (ticker, info, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (ticker, info_json))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cached company info for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error caching company info for {ticker}: {str(e)}")
    
    def get_cached_analysis(self, ticker, analysis_type, period, max_age_hours=1):
        """
        Get cached analysis results
        
        Args:
            ticker (str): Stock ticker symbol
            analysis_type (str): Type of analysis (TA, FA, etc.)
            period (str): Time period
            max_age_hours (int): Maximum age in hours before considering data stale
            
        Returns:
            dict or None: Cached analysis results if available and fresh
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            cursor.execute('''
                SELECT results, updated_at FROM analysis_results 
                WHERE ticker = ? AND analysis_type = ? AND period = ? AND updated_at > ?
            ''', (ticker, analysis_type, period, cutoff_time))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                results_json, updated_at = result
                results_dict = json.loads(results_json)
                
                self.logger.info(f"Retrieved cached {analysis_type} analysis for {ticker} - last updated: {updated_at}")
                return results_dict
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached analysis for {ticker}: {str(e)}")
            return None
    
    def cache_analysis(self, ticker, analysis_type, period, results):
        """
        Cache analysis results
        
        Args:
            ticker (str): Stock ticker symbol
            analysis_type (str): Type of analysis (TA, FA, etc.)
            period (str): Time period
            results (dict): Analysis results to cache
        """
        try:
            if not results:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert to JSON
            results_json = json.dumps(results, default=str)
            
            # Insert or update results
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_results (ticker, analysis_type, period, results, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (ticker, analysis_type, period, results_json))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cached {analysis_type} analysis for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error caching analysis for {ticker}: {str(e)}")
    
    def clear_cache(self, ticker=None, max_age_days=30):
        """
        Clear cached data
        
        Args:
            ticker (str, optional): Specific ticker to clear, or None for all
            max_age_days (int): Clear data older than this many days
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            if ticker:
                # Clear specific ticker
                cursor.execute('DELETE FROM stock_data WHERE ticker = ?', (ticker,))
                cursor.execute('DELETE FROM company_info WHERE ticker = ?', (ticker,))
                cursor.execute('DELETE FROM analysis_results WHERE ticker = ?', (ticker,))
                self.logger.info(f"Cleared cache for {ticker}")
            else:
                # Clear old data
                cursor.execute('DELETE FROM stock_data WHERE updated_at < ?', (cutoff_time,))
                cursor.execute('DELETE FROM company_info WHERE updated_at < ?', (cutoff_time,))
                cursor.execute('DELETE FROM analysis_results WHERE updated_at < ?', (cutoff_time,))
                self.logger.info(f"Cleared cache older than {max_age_days} days")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
    
    def get_cache_stats(self):
        """
        Get cache statistics
        
        Returns:
            dict: Cache statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Stock data stats
            cursor.execute('SELECT COUNT(*) FROM stock_data')
            stats['stock_data_count'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT ticker) FROM stock_data')
            stats['unique_tickers'] = cursor.fetchone()[0]
            
            # Company info stats
            cursor.execute('SELECT COUNT(*) FROM company_info')
            stats['company_info_count'] = cursor.fetchone()[0]
            
            # Analysis results stats
            cursor.execute('SELECT COUNT(*) FROM analysis_results')
            stats['analysis_results_count'] = cursor.fetchone()[0]
            
            # Database size
            stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            
            conn.close()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            return {}
    
    def cleanup_database(self):
        """
        Perform database maintenance
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Vacuum to reclaim space
            cursor.execute('VACUUM')
            
            # Analyze to update statistics
            cursor.execute('ANALYZE')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {str(e)}")
    
    def export_data(self, ticker, output_file):
        """
        Export cached data for a ticker to CSV
        
        Args:
            ticker (str): Stock ticker symbol
            output_file (str): Output file path
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get stock data
            query = '''
                SELECT ticker, period, data, updated_at 
                FROM stock_data 
                WHERE ticker = ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(ticker,))
            
            conn.close()
            
            if not df.empty:
                df.to_csv(output_file, index=False)
                self.logger.info(f"Exported data for {ticker} to {output_file}")
            else:
                self.logger.warning(f"No data found for {ticker}")
                
        except Exception as e:
            self.logger.error(f"Error exporting data for {ticker}: {str(e)}")
    
    def __del__(self):
        """Cleanup on object destruction"""
        try:
            # Close any remaining connections
            pass
        except:
            pass
