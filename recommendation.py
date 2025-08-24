"""
Recommendation engine for the AI Stock Analysis Bot
Combines technical and fundamental analysis to provide investment recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

class RecommendationEngine:
    def __init__(self, technical_analysis=None, fundamental_analysis=None):
        self.logger = logging.getLogger(__name__)
        self.technical_analysis = technical_analysis
        self.fundamental_analysis = fundamental_analysis
        
    def get_recommendation(self, ticker, ta_results, fa_results=None):
        """
        Generate investment recommendation combining TA and FA
        
        Args:
            ticker (str): Stock ticker symbol
            ta_results (dict): Technical analysis results
            fa_results (dict): Fundamental analysis results
            
        Returns:
            dict: Investment recommendation
        """
        try:
            # Get scores
            ta_score = ta_results.get('score', 50)
            fa_score = fa_results.get('score', 50) if fa_results else 50
            
            # Weight the scores (70% TA, 30% FA for short-term trading)
            ta_weight = 0.7
            fa_weight = 0.3
            
            # Calculate final score
            final_score = (ta_score * ta_weight) + (fa_score * fa_weight)
            
            # Generate recommendation
            action = self.get_action(final_score)
            
            # Generate detailed reasoning
            reasoning = self.generate_reasoning(ticker, ta_results, fa_results, final_score)
            
            # Calculate confidence level
            confidence = self.calculate_confidence(ta_results, fa_results, final_score)
            
            # Generate risk assessment
            risk_assessment = self.assess_risk(ta_results, fa_results)
            
            # Generate price targets
            price_targets = self.calculate_price_targets(ta_results, fa_results)
            
            recommendation = {
                'ticker': ticker,
                'action': action,
                'final_score': final_score,
                'ta_score': ta_score,
                'fa_score': fa_score,
                'reasoning': reasoning,
                'confidence': confidence,
                'risk_assessment': risk_assessment,
                'price_targets': price_targets,
                'timestamp': datetime.now().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'action': 'HOLD',
                'final_score': 50,
                'ta_score': 50,
                'fa_score': 50,
                'reasoning': f"Error in analysis: {str(e)}",
                'confidence': 'Low',
                'risk_assessment': 'High',
                'price_targets': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def get_action(self, score):
        """
        Determine investment action based on score
        
        Args:
            score (float): Combined analysis score
            
        Returns:
            str: Investment action
        """
        if score >= 75:
            return 'STRONG BUY'
        elif score >= 60:
            return 'BUY'
        elif score >= 40:
            return 'HOLD'
        elif score >= 25:
            return 'SELL'
        else:
            return 'STRONG SELL'
    
    def generate_reasoning(self, ticker, ta_results, fa_results, final_score):
        """
        Generate detailed reasoning for the recommendation
        
        Args:
            ticker (str): Stock ticker
            ta_results (dict): Technical analysis results
            fa_results (dict): Fundamental analysis results
            final_score (float): Final recommendation score
            
        Returns:
            str: Detailed reasoning
        """
        try:
            reasoning_parts = []
            
            # Overall assessment
            action = self.get_action(final_score)
            reasoning_parts.append(f"**{ticker} Analysis Summary ({action})**")
            reasoning_parts.append(f"Final Score: {final_score:.1f}/100")
            reasoning_parts.append("")
            
            # Technical analysis reasoning
            ta_score = ta_results.get('score', 50)
            reasoning_parts.append(f"**Technical Analysis (Score: {ta_score:.1f}/100)**")
            
            # RSI analysis
            indicators = ta_results.get('indicators', {})
            if 'RSI' in indicators and not indicators['RSI'].empty:
                current_rsi = indicators['RSI'].iloc[-1]
                if current_rsi < 30:
                    reasoning_parts.append(f"• RSI ({current_rsi:.1f}) indicates oversold conditions - potential buying opportunity")
                elif current_rsi > 70:
                    reasoning_parts.append(f"• RSI ({current_rsi:.1f}) indicates overbought conditions - caution advised")
                else:
                    reasoning_parts.append(f"• RSI ({current_rsi:.1f}) is in neutral territory")
            
            # MACD analysis
            if 'MACD' in indicators and 'MACD_Signal' in indicators:
                if not indicators['MACD'].empty and not indicators['MACD_Signal'].empty:
                    macd = indicators['MACD'].iloc[-1]
                    macd_signal = indicators['MACD_Signal'].iloc[-1]
                    if macd > macd_signal:
                        reasoning_parts.append("• MACD is above signal line - bullish momentum")
                    else:
                        reasoning_parts.append("• MACD is below signal line - bearish momentum")
            
            # Moving averages
            if 'SMA_20' in indicators and 'SMA_50' in indicators:
                if not indicators['SMA_20'].empty and not indicators['SMA_50'].empty:
                    sma_20 = indicators['SMA_20'].iloc[-1]
                    sma_50 = indicators['SMA_50'].iloc[-1]
                    if sma_20 > sma_50:
                        reasoning_parts.append("• Short-term MA above long-term MA - upward trend")
                    else:
                        reasoning_parts.append("• Short-term MA below long-term MA - downward trend")
            
            # Pattern analysis
            patterns = ta_results.get('patterns', [])
            if patterns:
                reasoning_parts.append("• Detected patterns:")
                for pattern in patterns[:3]:  # Show top 3 patterns
                    reasoning_parts.append(f"  - {pattern}")
            
            reasoning_parts.append("")
            
            # Fundamental analysis reasoning
            if fa_results:
                fa_score = fa_results.get('score', 50)
                reasoning_parts.append(f"**Fundamental Analysis (Score: {fa_score:.1f}/100)**")
                
                metrics = fa_results.get('metrics', {})
                
                # Valuation
                pe_ratio = metrics.get('pe_ratio', 0)
                if pe_ratio > 0:
                    if pe_ratio < 15:
                        reasoning_parts.append(f"• P/E ratio ({pe_ratio:.1f}) suggests potential undervaluation")
                    elif pe_ratio > 25:
                        reasoning_parts.append(f"• P/E ratio ({pe_ratio:.1f}) suggests potential overvaluation")
                    else:
                        reasoning_parts.append(f"• P/E ratio ({pe_ratio:.1f}) is within reasonable range")
                
                # Profitability
                roe = metrics.get('roe', 0)
                if roe > 0.15:
                    reasoning_parts.append(f"• Strong ROE ({roe:.1%}) indicates efficient management")
                elif roe < 0.05:
                    reasoning_parts.append(f"• Low ROE ({roe:.1%}) raises profitability concerns")
                
                # Growth
                revenue_growth = metrics.get('revenue_growth', 0)
                if revenue_growth > 0.10:
                    reasoning_parts.append(f"• Strong revenue growth ({revenue_growth:.1%}) shows business expansion")
                elif revenue_growth < 0:
                    reasoning_parts.append(f"• Declining revenue ({revenue_growth:.1%}) is concerning")
                
                # Financial health
                debt_to_equity = metrics.get('debt_to_equity', 0)
                if debt_to_equity < 0.3:
                    reasoning_parts.append(f"• Low debt-to-equity ({debt_to_equity:.1f}) indicates strong balance sheet")
                elif debt_to_equity > 1.0:
                    reasoning_parts.append(f"• High debt-to-equity ({debt_to_equity:.1f}) suggests financial risk")
                
                # Dividends
                dividend_yield = metrics.get('dividend_yield', 0)
                if dividend_yield > 0.03:
                    reasoning_parts.append(f"• Attractive dividend yield ({dividend_yield:.1%}) provides income")
                
                reasoning_parts.append("")
            
            # Risk factors
            reasoning_parts.append("**Key Risk Factors:**")
            risk_factors = self.identify_risk_factors(ta_results, fa_results)
            for risk in risk_factors:
                reasoning_parts.append(f"• {risk}")
            
            reasoning_parts.append("")
            
            # Investment thesis
            reasoning_parts.append("**Investment Thesis:**")
            thesis = self.generate_investment_thesis(ticker, ta_results, fa_results, final_score)
            reasoning_parts.append(thesis)
            
            return "\n".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {str(e)}")
            return f"Analysis completed with score {final_score:.1f}/100. Technical score: {ta_results.get('score', 50):.1f}, Fundamental score: {fa_results.get('score', 50) if fa_results else 50:.1f}"
    
    def calculate_confidence(self, ta_results, fa_results, final_score):
        """
        Calculate confidence level for the recommendation
        
        Args:
            ta_results (dict): Technical analysis results
            fa_results (dict): Fundamental analysis results
            final_score (float): Final recommendation score
            
        Returns:
            str: Confidence level
        """
        try:
            confidence_factors = []
            
            # Score extremes increase confidence
            if final_score > 80 or final_score < 20:
                confidence_factors.append(1)
            elif final_score > 70 or final_score < 30:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # Agreement between TA and FA
            if fa_results:
                ta_score = ta_results.get('score', 50)
                fa_score = fa_results.get('score', 50)
                score_diff = abs(ta_score - fa_score)
                
                if score_diff < 10:
                    confidence_factors.append(1)  # High agreement
                elif score_diff < 20:
                    confidence_factors.append(0.8)  # Moderate agreement
                else:
                    confidence_factors.append(0.5)  # Low agreement
            
            # Pattern detection increases confidence
            patterns = ta_results.get('patterns', [])
            if len(patterns) >= 3:
                confidence_factors.append(0.9)
            elif len(patterns) >= 1:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Calculate average confidence
            avg_confidence = sum(confidence_factors) / len(confidence_factors)
            
            if avg_confidence > 0.8:
                return 'High'
            elif avg_confidence > 0.6:
                return 'Medium'
            else:
                return 'Low'
                
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 'Medium'
    
    def assess_risk(self, ta_results, fa_results):
        """
        Assess investment risk level
        
        Args:
            ta_results (dict): Technical analysis results
            fa_results (dict): Fundamental analysis results
            
        Returns:
            str: Risk level
        """
        try:
            risk_factors = []
            
            # Technical risk factors
            indicators = ta_results.get('indicators', {})
            
            # High volatility
            if 'ATR' in indicators and not indicators['ATR'].empty:
                atr = indicators['ATR'].iloc[-1]
                # This is simplified - would need price context
                risk_factors.append(0.3)
            
            # Overbought/oversold conditions
            if 'RSI' in indicators and not indicators['RSI'].empty:
                rsi = indicators['RSI'].iloc[-1]
                if rsi > 80 or rsi < 20:
                    risk_factors.append(0.8)  # High risk
                elif rsi > 70 or rsi < 30:
                    risk_factors.append(0.6)  # Medium risk
                else:
                    risk_factors.append(0.3)  # Low risk
            
            # Fundamental risk factors
            if fa_results:
                metrics = fa_results.get('metrics', {})
                
                # High debt
                debt_to_equity = metrics.get('debt_to_equity', 0)
                if debt_to_equity > 1.0:
                    risk_factors.append(0.8)
                elif debt_to_equity > 0.5:
                    risk_factors.append(0.5)
                else:
                    risk_factors.append(0.2)
                
                # Low liquidity
                current_ratio = metrics.get('current_ratio', 0)
                if current_ratio < 1.0:
                    risk_factors.append(0.8)
                elif current_ratio < 1.5:
                    risk_factors.append(0.5)
                else:
                    risk_factors.append(0.2)
                
                # High valuation
                pe_ratio = metrics.get('pe_ratio', 0)
                if pe_ratio > 30:
                    risk_factors.append(0.7)
                elif pe_ratio > 20:
                    risk_factors.append(0.4)
                else:
                    risk_factors.append(0.2)
            
            # Calculate average risk
            if risk_factors:
                avg_risk = sum(risk_factors) / len(risk_factors)
                
                if avg_risk > 0.7:
                    return 'High'
                elif avg_risk > 0.4:
                    return 'Medium'
                else:
                    return 'Low'
            else:
                return 'Medium'
                
        except Exception as e:
            self.logger.error(f"Error assessing risk: {str(e)}")
            return 'Medium'
    
    def calculate_price_targets(self, ta_results, fa_results):
        """
        Calculate price targets based on analysis
        
        Args:
            ta_results (dict): Technical analysis results
            fa_results (dict): Fundamental analysis results
            
        Returns:
            dict: Price targets
        """
        try:
            price_targets = {}
            
            # Get current price from TA data
            indicators = ta_results.get('indicators', {})
            current_price = None
            
            # Try to get current price from various sources
            if 'Close' in indicators and not indicators['Close'].empty:
                current_price = indicators['Close'].iloc[-1]
            elif 'SMA_20' in indicators and not indicators['SMA_20'].empty:
                current_price = indicators['SMA_20'].iloc[-1]
            
            if current_price is None or current_price <= 0:
                return {'error': 'No current price available'}
            
            # Simple percentage-based targets
            price_targets['conservative_target'] = current_price * 1.05  # 5% upside
            price_targets['moderate_target'] = current_price * 1.10     # 10% upside
            price_targets['aggressive_target'] = current_price * 1.20   # 20% upside
            price_targets['stop_loss'] = current_price * 0.95          # 5% downside
            
            return price_targets
            
        except Exception as e:
            self.logger.error(f"Error calculating price targets: {str(e)}")
            return {'error': str(e)}
    
    def identify_risk_factors(self, ta_results, fa_results):
        """
        Identify key risk factors
        
        Args:
            ta_results (dict): Technical analysis results
            fa_results (dict): Fundamental analysis results
            
        Returns:
            list: List of risk factors
        """
        risk_factors = []
        
        try:
            # Technical risk factors
            indicators = ta_results.get('indicators', {})
            
            # RSI extremes
            if 'RSI' in indicators and not indicators['RSI'].empty:
                rsi = indicators['RSI'].iloc[-1]
                if rsi > 80:
                    risk_factors.append("Extremely overbought conditions (RSI > 80)")
                elif rsi < 20:
                    risk_factors.append("Extremely oversold conditions (RSI < 20)")
            
            # Volume concerns
            if 'Volume_SMA' in indicators and not indicators['Volume_SMA'].empty:
                # This would need actual volume data to compare
                risk_factors.append("Monitor volume patterns for confirmation")
            
            # Fundamental risk factors
            if fa_results:
                metrics = fa_results.get('metrics', {})
                
                # High debt
                debt_to_equity = metrics.get('debt_to_equity', 0)
                if debt_to_equity > 1.0:
                    risk_factors.append(f"High debt-to-equity ratio ({debt_to_equity:.1f})")
                
                # Poor liquidity
                current_ratio = metrics.get('current_ratio', 0)
                if current_ratio < 1.0:
                    risk_factors.append(f"Poor liquidity (current ratio: {current_ratio:.1f})")
                
                # Negative growth
                revenue_growth = metrics.get('revenue_growth', 0)
                if revenue_growth < -0.05:
                    risk_factors.append(f"Declining revenue ({revenue_growth:.1%})")
                
                # High valuation
                pe_ratio = metrics.get('pe_ratio', 0)
                if pe_ratio > 30:
                    risk_factors.append(f"High P/E ratio ({pe_ratio:.1f}) - valuation risk")
                
                # Low profitability
                roe = metrics.get('roe', 0)
                if roe < 0.05:
                    risk_factors.append(f"Low return on equity ({roe:.1%})")
            
            # General market risks
            risk_factors.append("Market volatility and economic conditions")
            risk_factors.append("Sector-specific risks and competition")
            
            return risk_factors
            
        except Exception as e:
            self.logger.error(f"Error identifying risk factors: {str(e)}")
            return ["Unable to assess risk factors due to data limitations"]
    
    def generate_investment_thesis(self, ticker, ta_results, fa_results, final_score):
        """
        Generate investment thesis
        
        Args:
            ticker (str): Stock ticker
            ta_results (dict): Technical analysis results
            fa_results (dict): Fundamental analysis results
            final_score (float): Final recommendation score
            
        Returns:
            str: Investment thesis
        """
        try:
            action = self.get_action(final_score)
            
            if action in ['STRONG BUY', 'BUY']:
                thesis = f"Based on our comprehensive analysis, {ticker} presents a compelling investment opportunity. "
                
                # Technical strengths
                if ta_results.get('score', 50) > 60:
                    thesis += "The technical analysis reveals strong momentum indicators and favorable chart patterns. "
                
                # Fundamental strengths
                if fa_results and fa_results.get('score', 50) > 60:
                    thesis += "Fundamental analysis shows solid financial health, reasonable valuation, and growth potential. "
                
                thesis += "The combination of technical and fundamental factors supports a positive outlook."
                
            elif action == 'HOLD':
                thesis = f"{ticker} appears to be fairly valued with mixed signals from our analysis. "
                thesis += "While there are some positive indicators, there are also concerns that suggest maintaining current positions rather than increasing exposure. "
                thesis += "Monitor for clearer directional signals before making significant changes."
                
            else:  # SELL or STRONG SELL
                thesis = f"Our analysis suggests caution regarding {ticker}. "
                
                # Technical concerns
                if ta_results.get('score', 50) < 40:
                    thesis += "Technical indicators show weakening momentum and potential downside risk. "
                
                # Fundamental concerns
                if fa_results and fa_results.get('score', 50) < 40:
                    thesis += "Fundamental analysis reveals concerns about valuation, financial health, or growth prospects. "
                
                thesis += "The risk-reward profile suggests considering position reduction or avoidance."
            
            return thesis
            
        except Exception as e:
            self.logger.error(f"Error generating investment thesis: {str(e)}")
            return "Investment thesis could not be generated due to analysis limitations."
    
    def get_sector_recommendations(self, recommendations):
        """
        Analyze recommendations by sector
        
        Args:
            recommendations (list): List of stock recommendations
            
        Returns:
            dict: Sector analysis
        """
        try:
            sector_analysis = {}
            
            for rec in recommendations:
                # This would need sector information from fundamental analysis
                # For now, we'll use a placeholder
                sector = "Technology"  # Would get from FA results
                
                if sector not in sector_analysis:
                    sector_analysis[sector] = {
                        'stocks': [],
                        'avg_score': 0,
                        'actions': {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                    }
                
                sector_analysis[sector]['stocks'].append(rec)
                sector_analysis[sector]['actions'][rec['action']] += 1
            
            # Calculate averages
            for sector, data in sector_analysis.items():
                if data['stocks']:
                    data['avg_score'] = sum(stock['final_score'] for stock in data['stocks']) / len(data['stocks'])
            
            return sector_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing sectors: {str(e)}")
            return {}
