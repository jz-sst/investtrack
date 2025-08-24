"""
Fundamental analysis module for the AI Stock Analysis Bot
Analyzes financial metrics and company fundamentals
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

class FundamentalAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_financial_metrics(self, ticker):
        """
        Get key financial metrics for a stock
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Financial metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                return {}
            
            metrics = {
                # Valuation metrics
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'ev_to_revenue': info.get('enterpriseToRevenue', 0),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 0),
                
                # Profitability metrics
                'profit_margins': info.get('profitMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'gross_margins': info.get('grossMargins', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'roic': info.get('returnOnCapital', 0),
                
                # Growth metrics
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0),
                
                # Financial health
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'cash_ratio': info.get('cashRatio', 0),
                
                # Per share metrics
                'eps': info.get('trailingEps', 0),
                'forward_eps': info.get('forwardEps', 0),
                'book_value': info.get('bookValue', 0),
                'revenue_per_share': info.get('revenuePerShare', 0),
                
                # Dividend metrics
                'dividend_yield': info.get('dividendYield', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                
                # Market metrics
                'beta': info.get('beta', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'shares_short': info.get('sharesShort', 0),
                'short_ratio': info.get('shortRatio', 0),
                
                # Company info
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'employees': info.get('fullTimeEmployees', 0),
                'country': info.get('country', ''),
                'exchange': info.get('exchange', ''),
                
                # Analyst metrics
                'target_price': info.get('targetMeanPrice', 0),
                'recommendation': info.get('recommendationMean', 0),
                'number_of_analysts': info.get('numberOfAnalystOpinions', 0),
                
                # Recent financials
                'total_revenue': info.get('totalRevenue', 0),
                'total_debt': info.get('totalDebt', 0),
                'total_cash': info.get('totalCash', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'operating_cash_flow': info.get('operatingCashflow', 0),
                
                'last_updated': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting financial metrics for {ticker}: {str(e)}")
            return {}
    
    def calculate_intrinsic_value(self, ticker, metrics):
        """
        Calculate intrinsic value using DCF model (simplified)
        
        Args:
            ticker (str): Stock ticker symbol
            metrics (dict): Financial metrics
            
        Returns:
            float: Estimated intrinsic value
        """
        try:
            # Simplified DCF calculation
            free_cash_flow = metrics.get('free_cash_flow', 0)
            revenue_growth = metrics.get('revenue_growth', 0)
            shares_outstanding = metrics.get('shares_outstanding', 0)
            
            if free_cash_flow <= 0 or shares_outstanding <= 0:
                return 0
            
            # Assume growth rate decreases over time
            growth_rate = max(0, min(0.15, revenue_growth))  # Cap at 15%
            terminal_growth = 0.03  # 3% terminal growth
            discount_rate = 0.1  # 10% discount rate
            
            # Project 5 years of cash flows
            projected_fcf = []
            current_fcf = free_cash_flow
            
            for year in range(5):
                # Decrease growth rate each year
                year_growth = growth_rate * (0.8 ** year)
                current_fcf *= (1 + year_growth)
                projected_fcf.append(current_fcf)
            
            # Calculate terminal value
            terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
            
            # Discount all cash flows to present value
            present_value = 0
            for i, fcf in enumerate(projected_fcf):
                present_value += fcf / ((1 + discount_rate) ** (i + 1))
            
            # Add discounted terminal value
            present_value += terminal_value / ((1 + discount_rate) ** 5)
            
            # Calculate per share value
            intrinsic_value = present_value / shares_outstanding
            
            return intrinsic_value
            
        except Exception as e:
            self.logger.error(f"Error calculating intrinsic value for {ticker}: {str(e)}")
            return 0
    
    def analyze_financial_health(self, metrics):
        """
        Analyze financial health based on key metrics
        
        Args:
            metrics (dict): Financial metrics
            
        Returns:
            dict: Financial health analysis
        """
        try:
            health_score = 0
            max_score = 100
            
            # Debt analysis (20 points)
            debt_to_equity = metrics.get('debt_to_equity', 0)
            if debt_to_equity == 0:
                health_score += 20
            elif debt_to_equity < 0.3:
                health_score += 18
            elif debt_to_equity < 0.5:
                health_score += 15
            elif debt_to_equity < 1.0:
                health_score += 10
            else:
                health_score += 5
            
            # Liquidity analysis (20 points)
            current_ratio = metrics.get('current_ratio', 0)
            if current_ratio > 2.0:
                health_score += 20
            elif current_ratio > 1.5:
                health_score += 18
            elif current_ratio > 1.0:
                health_score += 15
            else:
                health_score += 5
            
            # Profitability analysis (30 points)
            roe = metrics.get('roe', 0)
            roa = metrics.get('roa', 0)
            profit_margins = metrics.get('profit_margins', 0)
            
            if roe > 0.15:
                health_score += 10
            elif roe > 0.10:
                health_score += 8
            elif roe > 0.05:
                health_score += 5
            
            if roa > 0.05:
                health_score += 10
            elif roa > 0.03:
                health_score += 8
            elif roa > 0.01:
                health_score += 5
            
            if profit_margins > 0.15:
                health_score += 10
            elif profit_margins > 0.10:
                health_score += 8
            elif profit_margins > 0.05:
                health_score += 5
            
            # Growth analysis (20 points)
            revenue_growth = metrics.get('revenue_growth', 0)
            earnings_growth = metrics.get('earnings_growth', 0)
            
            if revenue_growth > 0.15:
                health_score += 10
            elif revenue_growth > 0.10:
                health_score += 8
            elif revenue_growth > 0.05:
                health_score += 5
            elif revenue_growth > 0:
                health_score += 3
            
            if earnings_growth > 0.15:
                health_score += 10
            elif earnings_growth > 0.10:
                health_score += 8
            elif earnings_growth > 0.05:
                health_score += 5
            elif earnings_growth > 0:
                health_score += 3
            
            # Cash flow analysis (10 points)
            free_cash_flow = metrics.get('free_cash_flow', 0)
            operating_cash_flow = metrics.get('operating_cash_flow', 0)
            
            if free_cash_flow > 0 and operating_cash_flow > 0:
                health_score += 10
            elif operating_cash_flow > 0:
                health_score += 5
            
            return {
                'score': health_score,
                'max_score': max_score,
                'rating': self.get_health_rating(health_score),
                'strengths': self.identify_strengths(metrics),
                'weaknesses': self.identify_weaknesses(metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing financial health: {str(e)}")
            return {
                'score': 50,
                'max_score': 100,
                'rating': 'Average',
                'strengths': [],
                'weaknesses': []
            }
    
    def get_health_rating(self, score):
        """Get health rating based on score"""
        if score >= 80:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 60:
            return 'Average'
        elif score >= 50:
            return 'Below Average'
        else:
            return 'Poor'
    
    def identify_strengths(self, metrics):
        """Identify financial strengths"""
        strengths = []
        
        # High ROE
        if metrics.get('roe', 0) > 0.15:
            strengths.append("High return on equity (>15%)")
        
        # Low debt
        if metrics.get('debt_to_equity', 0) < 0.3:
            strengths.append("Low debt-to-equity ratio")
        
        # Strong liquidity
        if metrics.get('current_ratio', 0) > 2.0:
            strengths.append("Strong liquidity position")
        
        # High profit margins
        if metrics.get('profit_margins', 0) > 0.15:
            strengths.append("High profit margins (>15%)")
        
        # Strong growth
        if metrics.get('revenue_growth', 0) > 0.10:
            strengths.append("Strong revenue growth")
        
        # Positive free cash flow
        if metrics.get('free_cash_flow', 0) > 0:
            strengths.append("Positive free cash flow")
        
        # Dividend yield
        if metrics.get('dividend_yield', 0) > 0.03:
            strengths.append("Attractive dividend yield")
        
        return strengths
    
    def identify_weaknesses(self, metrics):
        """Identify financial weaknesses"""
        weaknesses = []
        
        # High debt
        if metrics.get('debt_to_equity', 0) > 1.0:
            weaknesses.append("High debt-to-equity ratio")
        
        # Poor liquidity
        if metrics.get('current_ratio', 0) < 1.0:
            weaknesses.append("Poor liquidity position")
        
        # Low profitability
        if metrics.get('roe', 0) < 0.05:
            weaknesses.append("Low return on equity")
        
        # Negative growth
        if metrics.get('revenue_growth', 0) < 0:
            weaknesses.append("Declining revenue")
        
        # High valuation
        if metrics.get('pe_ratio', 0) > 30:
            weaknesses.append("High P/E ratio (potentially overvalued)")
        
        # Negative free cash flow
        if metrics.get('free_cash_flow', 0) < 0:
            weaknesses.append("Negative free cash flow")
        
        return weaknesses
    
    def compare_to_industry(self, ticker, metrics):
        """
        Compare metrics to industry averages (simplified)
        
        Args:
            ticker (str): Stock ticker
            metrics (dict): Financial metrics
            
        Returns:
            dict: Industry comparison
        """
        try:
            # Industry averages (simplified - would normally fetch from database)
            industry_averages = {
                'Technology': {
                    'pe_ratio': 25,
                    'profit_margins': 0.20,
                    'roe': 0.15,
                    'debt_to_equity': 0.2
                },
                'Healthcare': {
                    'pe_ratio': 20,
                    'profit_margins': 0.15,
                    'roe': 0.12,
                    'debt_to_equity': 0.3
                },
                'Financial Services': {
                    'pe_ratio': 15,
                    'profit_margins': 0.25,
                    'roe': 0.10,
                    'debt_to_equity': 0.8
                },
                'Consumer Discretionary': {
                    'pe_ratio': 18,
                    'profit_margins': 0.08,
                    'roe': 0.12,
                    'debt_to_equity': 0.4
                },
                'Default': {
                    'pe_ratio': 18,
                    'profit_margins': 0.10,
                    'roe': 0.10,
                    'debt_to_equity': 0.4
                }
            }
            
            sector = metrics.get('sector', 'Default')
            industry_avg = industry_averages.get(sector, industry_averages['Default'])
            
            comparison = {}
            
            # Compare key metrics
            for metric, avg_value in industry_avg.items():
                current_value = metrics.get(metric, 0)
                if avg_value > 0:
                    comparison[metric] = {
                        'current': current_value,
                        'industry_avg': avg_value,
                        'vs_industry': (current_value - avg_value) / avg_value * 100
                    }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing to industry for {ticker}: {str(e)}")
            return {}
    
    def calculate_fa_score(self, metrics):
        """
        Calculate fundamental analysis score
        
        Args:
            metrics (dict): Financial metrics
            
        Returns:
            float: FA score (0-100)
        """
        try:
            score = 0
            total_weight = 0
            
            # Valuation scoring (weight: 25)
            pe_ratio = metrics.get('pe_ratio', 0)
            if pe_ratio > 0:
                if pe_ratio < 15:
                    score += 80 * 25  # Undervalued
                elif pe_ratio < 25:
                    score += 60 * 25  # Fair value
                elif pe_ratio < 35:
                    score += 40 * 25  # Slightly overvalued
                else:
                    score += 20 * 25  # Overvalued
                total_weight += 25
            
            # Profitability scoring (weight: 30)
            roe = metrics.get('roe', 0)
            profit_margins = metrics.get('profit_margins', 0)
            
            if roe > 0.15:
                score += 80 * 15
            elif roe > 0.10:
                score += 60 * 15
            elif roe > 0.05:
                score += 40 * 15
            else:
                score += 20 * 15
            total_weight += 15
            
            if profit_margins > 0.15:
                score += 80 * 15
            elif profit_margins > 0.10:
                score += 60 * 15
            elif profit_margins > 0.05:
                score += 40 * 15
            else:
                score += 20 * 15
            total_weight += 15
            
            # Growth scoring (weight: 20)
            revenue_growth = metrics.get('revenue_growth', 0)
            earnings_growth = metrics.get('earnings_growth', 0)
            
            if revenue_growth > 0.15:
                score += 80 * 10
            elif revenue_growth > 0.10:
                score += 60 * 10
            elif revenue_growth > 0.05:
                score += 50 * 10
            elif revenue_growth > 0:
                score += 40 * 10
            else:
                score += 20 * 10
            total_weight += 10
            
            if earnings_growth > 0.15:
                score += 80 * 10
            elif earnings_growth > 0.10:
                score += 60 * 10
            elif earnings_growth > 0.05:
                score += 50 * 10
            elif earnings_growth > 0:
                score += 40 * 10
            else:
                score += 20 * 10
            total_weight += 10
            
            # Financial health scoring (weight: 15)
            debt_to_equity = metrics.get('debt_to_equity', 0)
            current_ratio = metrics.get('current_ratio', 0)
            
            if debt_to_equity < 0.3:
                score += 80 * 8
            elif debt_to_equity < 0.5:
                score += 60 * 8
            elif debt_to_equity < 1.0:
                score += 40 * 8
            else:
                score += 20 * 8
            total_weight += 8
            
            if current_ratio > 2.0:
                score += 80 * 7
            elif current_ratio > 1.5:
                score += 60 * 7
            elif current_ratio > 1.0:
                score += 40 * 7
            else:
                score += 20 * 7
            total_weight += 7
            
            # Dividend scoring (weight: 10)
            dividend_yield = metrics.get('dividend_yield', 0)
            if dividend_yield > 0.05:
                score += 80 * 10
            elif dividend_yield > 0.03:
                score += 60 * 10
            elif dividend_yield > 0.01:
                score += 40 * 10
            else:
                score += 30 * 10
            total_weight += 10
            
            # Calculate final score
            if total_weight > 0:
                final_score = score / total_weight
            else:
                final_score = 50
            
            return max(0, min(100, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating FA score: {str(e)}")
            return 50
    
    def analyze_stock(self, ticker):
        """
        Perform complete fundamental analysis on a stock
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Complete fundamental analysis results
        """
        try:
            # Get financial metrics
            metrics = self.get_financial_metrics(ticker)
            
            if not metrics:
                return None
            
            # Calculate intrinsic value
            intrinsic_value = self.calculate_intrinsic_value(ticker, metrics)
            
            # Analyze financial health
            health_analysis = self.analyze_financial_health(metrics)
            
            # Compare to industry
            industry_comparison = self.compare_to_industry(ticker, metrics)
            
            # Calculate FA score
            fa_score = self.calculate_fa_score(metrics)
            
            results = {
                'ticker': ticker,
                'metrics': metrics,
                'intrinsic_value': intrinsic_value,
                'health_analysis': health_analysis,
                'industry_comparison': industry_comparison,
                'score': fa_score,
                'analysis_date': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing stock {ticker}: {str(e)}")
            return None
