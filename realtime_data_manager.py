"""
Real-time Data Manager for Stock Analysis
- Multiple free API sources for redundancy
- Minute-by-minute updates where possible
- Data validation and fallback mechanisms
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any
import streamlit as st
import time

class RealTimeDataManager:
    """Manage real-time stock data from multiple free sources"""
    
    def __init__(self):
        self.primary_source = 'yfinance'
        self.fallback_sources = ['alpha_vantage_free', 'finnhub_free']
        self.cache_duration = 60  # Cache for 1 minute
        self.data_cache = {}
    
    def get_live_stock_data(self, ticker: str, period: str = '1d') -> Optional[pd.DataFrame]:
        """Get live stock data with caching and fallback"""
        
        cache_key = f"{ticker}_{period}_{int(time.time() // self.cache_duration)}"
        
        # Check cache first
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Try primary source (yfinance)
        data = self._get_yfinance_data(ticker, period)
        
        if data is not None and not data.empty:
            self.data_cache[cache_key] = data
            return data
        
        # Try fallback sources if primary fails
        for source in self.fallback_sources:
            if source == 'alpha_vantage_free':
                data = self._get_alpha_vantage_data(ticker)
            elif source == 'finnhub_free':
                data = self._get_finnhub_data(ticker)
            
            if data is not None and not data.empty:
                self.data_cache[cache_key] = data
                return data
        
        return None
    
    def _get_yfinance_data(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Get data from Yahoo Finance via yfinance"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval='1m' if period == '1d' else '1d')
            
            if not data.empty:
                # Add metadata
                data.attrs['source'] = 'yfinance'
                data.attrs['last_updated'] = datetime.now().isoformat()
                return data
                
        except Exception as e:
            st.warning(f"yfinance error for {ticker}: {str(e)}")
        
        return None
    
    def _get_alpha_vantage_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get data from Alpha Vantage (free tier)"""
        try:
            # Note: This would require an API key in production
            # For now, return None to indicate unavailable
            return None
            
        except Exception as e:
            return None
    
    def _get_finnhub_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get data from Finnhub (free tier)"""
        try:
            # Note: This would require an API key in production
            # For now, return None to indicate unavailable
            return None
            
        except Exception as e:
            return None
    
    def get_real_time_quote(self, ticker: str) -> Dict[str, Any]:
        """Get real-time quote data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get most recent price data
            hist = stock.history(period='1d', interval='1m')
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100 if prev_close else 0
                
                return {
                    'ticker': ticker,
                    'current_price': current_price,
                    'previous_close': prev_close,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': hist['Volume'].iloc[-1],
                    'high': hist['High'].max(),
                    'low': hist['Low'].min(),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'yfinance'
                }
        
        except Exception as e:
            return {
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            # Use SPY as market indicator
            spy = yf.Ticker('SPY')
            spy_data = spy.history(period='1d', interval='1m')
            
            if not spy_data.empty:
                current_time = datetime.now()
                last_trade_time = spy_data.index[-1].tz_localize(None)
                
                # Check if market is open (simple heuristic)
                time_diff = (current_time - last_trade_time).total_seconds()
                is_open = time_diff < 300  # Within 5 minutes
                
                return {
                    'is_open': is_open,
                    'last_trade_time': last_trade_time.isoformat(),
                    'spy_price': spy_data['Close'].iloc[-1],
                    'timestamp': current_time.isoformat()
                }
        
        except Exception as e:
            return {
                'is_open': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_trending_stocks(self) -> List[Dict[str, Any]]:
        """Get trending stocks from multiple sources"""
        
        # Predefined list of popular stocks for demo
        # In production, this would pull from APIs
        trending_tickers = [
            'AAPL', 'TSLA', 'NVDA', 'GOOGL', 'MSFT', 
            'AMZN', 'META', 'NFLX', 'AMD', 'CRM'
        ]
        
        trending_data = []
        
        for ticker in trending_tickers:
            quote = self.get_real_time_quote(ticker)
            if 'error' not in quote:
                trending_data.append({
                    'ticker': ticker,
                    'price': quote['current_price'],
                    'change_percent': quote['change_percent'],
                    'volume': quote['volume'],
                    'trend_score': abs(quote['change_percent'])  # Simple trending score
                })
        
        # Sort by trend score (highest volatility)
        trending_data.sort(key=lambda x: x['trend_score'], reverse=True)
        
        return trending_data[:10]
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if ticker exists and has data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            return not hist.empty
        except:
            return False
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get company information for fundamental analysis"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'roe': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'revenue': info.get('totalRevenue', 0),
                'net_income': info.get('netIncomeToCommon', 0),
                'timestamp': datetime.now().isoformat(),
                'source': 'yfinance'
            }
            
        except Exception as e:
            return {
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_data_freshness(self, ticker: str) -> Dict[str, Any]:
        """Check how fresh the data is"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d', interval='1m')
            
            if not hist.empty:
                last_update = hist.index[-1]
                time_diff = datetime.now() - last_update.tz_localize(None)
                
                return {
                    'ticker': ticker,
                    'last_update': last_update.isoformat(),
                    'minutes_old': int(time_diff.total_seconds() / 60),
                    'is_stale': time_diff.total_seconds() > 300,  # More than 5 minutes
                    'data_points': len(hist)
                }
        
        except Exception as e:
            return {
                'ticker': ticker,
                'error': str(e),
                'is_stale': True
            }
    
    def clear_cache(self):
        """Clear the data cache"""
        self.data_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.data_cache),
            'cache_keys': list(self.data_cache.keys()),
            'cache_duration_seconds': self.cache_duration
        }

# Global instance
data_manager = RealTimeDataManager()