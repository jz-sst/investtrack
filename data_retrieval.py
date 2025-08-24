"""
Data retrieval module for the AI Stock Analysis Bot
Handles fetching stock data from yfinance with SQLite caching
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import logging

class DataRetrieval:
    def __init__(self, db):
        self.db = db
        self.logger = logging.getLogger(__name__)
        
    def get_stock_data(self, ticker, period='1y', force_refresh=False):
        """
        Get stock data with caching
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            force_refresh (bool): Force refresh from API
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Check cache first
            if not force_refresh:
                cached_data = self.db.get_cached_data(ticker, period)
                if cached_data is not None:
                    self.logger.info(f"Retrieved cached data for {ticker}")
                    return cached_data
            
            # Fetch from yfinance
            self.logger.info(f"Fetching data for {ticker} from yfinance")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                self.logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Clean and prepare data
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error(f"Missing required column {col} for {ticker}")
                    return pd.DataFrame()
            
            # Cache the data
            self.db.cache_data(ticker, period, data)
            
            self.logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_company_info(self, ticker):
        """
        Get company information
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Company information
        """
        try:
            # Check cache first
            cached_info = self.db.get_cached_info(ticker)
            if cached_info:
                return cached_info
            
            # Fetch from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                return {}
            
            # Extract relevant information
            company_info = {
                'symbol': info.get('symbol', ticker),
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'eps': info.get('trailingEps', 0),
                'revenue': info.get('totalRevenue', 0),
                'gross_margins': info.get('grossMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'profit_margins': info.get('profitMargins', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'current_price': info.get('currentPrice', 0),
                'target_price': info.get('targetMeanPrice', 0),
                'recommendation': info.get('recommendationMean', 0),
                'last_updated': datetime.now().isoformat()
            }
            
            # Cache the info
            self.db.cache_info(ticker, company_info)
            
            return company_info
            
        except Exception as e:
            self.logger.error(f"Error fetching company info for {ticker}: {str(e)}")
            return {}
    
    def get_financial_data(self, ticker):
        """
        Get financial statements data
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Financial data including income statement, balance sheet, cash flow
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            financial_data = {
                'income_statement': financials.to_dict() if not financials.empty else {},
                'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                'cash_flow': cash_flow.to_dict() if not cash_flow.empty else {},
                'last_updated': datetime.now().isoformat()
            }
            
            return financial_data
            
        except Exception as e:
            self.logger.error(f"Error fetching financial data for {ticker}: {str(e)}")
            return {}
    
    def get_analyst_recommendations(self, ticker):
        """
        Get analyst recommendations
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Analyst recommendations and price targets
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get recommendations
            recommendations = stock.recommendations
            
            if recommendations is None or recommendations.empty:
                return {}
            
            # Get latest recommendation
            latest_rec = recommendations.iloc[-1] if len(recommendations) > 0 else None
            
            rec_data = {
                'latest_recommendation': latest_rec.to_dict() if latest_rec is not None else {},
                'all_recommendations': recommendations.to_dict('records') if not recommendations.empty else [],
                'last_updated': datetime.now().isoformat()
            }
            
            return rec_data
            
        except Exception as e:
            self.logger.error(f"Error fetching analyst recommendations for {ticker}: {str(e)}")
            return {}
    
    def get_earnings_data(self, ticker):
        """
        Get earnings data
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Earnings data
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get earnings
            earnings = stock.earnings
            quarterly_earnings = stock.quarterly_earnings
            
            earnings_data = {
                'annual_earnings': earnings.to_dict() if not earnings.empty else {},
                'quarterly_earnings': quarterly_earnings.to_dict() if not quarterly_earnings.empty else {},
                'last_updated': datetime.now().isoformat()
            }
            
            return earnings_data
            
        except Exception as e:
            self.logger.error(f"Error fetching earnings data for {ticker}: {str(e)}")
            return {}
    
    def get_multiple_stocks_data(self, tickers, period='1y'):
        """
        Get data for multiple stocks efficiently
        
        Args:
            tickers (list): List of stock ticker symbols
            period (str): Time period
            
        Returns:
            dict: Dictionary with ticker as key and data as value
        """
        results = {}
        
        for ticker in tickers:
            try:
                data = self.get_stock_data(ticker, period)
                if not data.empty:
                    results[ticker] = data
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        return results
    
    def refresh_cache(self, ticker, period='1y'):
        """
        Force refresh cache for a ticker
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period
            
        Returns:
            pd.DataFrame: Refreshed stock data
        """
        return self.get_stock_data(ticker, period, force_refresh=True)
    
    def get_market_data(self):
        """
        Get general market data (indices)
        
        Returns:
            dict: Market indices data
        """
        try:
            indices = {
                'S&P 500': '^GSPC',
                'NASDAQ': '^IXIC',
                'Dow Jones': '^DJI',
                'Russell 2000': '^RUT',
                'VIX': '^VIX'
            }
            
            market_data = {}
            
            for name, symbol in indices.items():
                try:
                    data = self.get_stock_data(symbol, '5d')
                    if not data.empty:
                        latest = data.iloc[-1]
                        prev = data.iloc[-2] if len(data) > 1 else latest
                        change = latest['Close'] - prev['Close']
                        change_pct = (change / prev['Close']) * 100
                        
                        market_data[name] = {
                            'symbol': symbol,
                            'price': latest['Close'],
                            'change': change,
                            'change_percent': change_pct,
                            'volume': latest['Volume']
                        }
                except Exception as e:
                    self.logger.error(f"Error fetching market data for {name}: {str(e)}")
                    continue
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return {}
