import os
import yfinance as yf
import pandas as pd
from openai import OpenAI
import json
from datetime import datetime, timedelta

class TechnicalAgent:
    """Specialized agent for technical analysis"""
    
    def __init__(self):
        self.name = "Technical Analyst Agent"
    
    def analyze(self, ticker, data):
        """Perform technical analysis"""
        if data is None or len(data) < 20:
            return {"score": 0, "analysis": "Insufficient data for technical analysis"}
        
        latest = data.iloc[-1]
        
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Analysis
        score = 0
        signals = []
        
        # Price vs Moving Averages
        if latest['Close'] > latest['SMA_20']:
            score += 1
            signals.append("Bullish: Price above 20-day MA")
        
        if len(data) >= 50 and latest['Close'] > latest['SMA_50']:
            score += 1
            signals.append("Bullish: Price above 50-day MA")
        
        # RSI
        latest_rsi = rsi.iloc[-1]
        if latest_rsi > 70:
            score -= 0.5
            signals.append(f"Caution: RSI overbought at {latest_rsi:.1f}")
        elif latest_rsi < 30:
            score += 0.5
            signals.append(f"Opportunity: RSI oversold at {latest_rsi:.1f}")
        
        return {
            "score": score,
            "analysis": f"Technical score: {score}/3. " + " | ".join(signals),
            "rsi": latest_rsi,
            "price_vs_sma20": "Above" if latest['Close'] > latest['SMA_20'] else "Below",
            "signals": signals
        }

class FundamentalAgent:
    """Specialized agent for fundamental analysis"""
    
    def __init__(self):
        self.name = "Fundamental Analyst Agent"
    
    def analyze(self, ticker, info):
        """Perform fundamental analysis"""
        if not info:
            return {"score": 0, "analysis": "No fundamental data available"}
        
        score = 0
        metrics = {}
        signals = []
        
        # P/E Ratio
        pe_ratio = info.get('trailingPE')
        if pe_ratio and pe_ratio > 0:
            metrics['PE_Ratio'] = pe_ratio
            if pe_ratio < 15:
                score += 1
                signals.append(f"Good value: P/E ratio {pe_ratio:.1f}")
            elif pe_ratio > 25:
                score -= 0.5
                signals.append(f"Expensive: P/E ratio {pe_ratio:.1f}")
        
        # Profit Margins
        profit_margin = info.get('profitMargins')
        if profit_margin:
            metrics['Profit_Margin'] = profit_margin * 100
            if profit_margin > 0.2:
                score += 1
                signals.append(f"Strong profitability: {profit_margin*100:.1f}% margin")
            elif profit_margin < 0.05:
                score -= 1
                signals.append(f"Weak profitability: {profit_margin*100:.1f}% margin")
        
        # Revenue Growth
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth:
            metrics['Revenue_Growth'] = revenue_growth * 100
            if revenue_growth > 0.15:
                score += 1
                signals.append(f"Strong growth: {revenue_growth*100:.1f}% revenue growth")
            elif revenue_growth < 0:
                score -= 1
                signals.append(f"Declining revenue: {revenue_growth*100:.1f}%")
        
        # Debt to Equity
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity:
            metrics['Debt_to_Equity'] = debt_to_equity
            if debt_to_equity < 50:
                score += 0.5
                signals.append(f"Low debt: D/E ratio {debt_to_equity:.1f}")
            elif debt_to_equity > 100:
                score -= 0.5
                signals.append(f"High debt: D/E ratio {debt_to_equity:.1f}")
        
        return {
            "score": score,
            "analysis": f"Fundamental score: {score}/4. " + " | ".join(signals),
            "metrics": metrics,
            "signals": signals
        }

class SentimentAgent:
    """Specialized agent for sentiment analysis"""
    
    def __init__(self):
        self.name = "Sentiment Analyst Agent"
        self.client = None
        if os.getenv('XAI_API_KEY'):
            self.client = OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=os.getenv('XAI_API_KEY')
            )
    
    def analyze(self, ticker, company_name):
        """Perform sentiment analysis using AI"""
        if not self.client:
            return {"score": 0, "analysis": "Sentiment analysis requires API key"}
        
        try:
            # Generate a sentiment prompt based on current market context
            prompt = f"""Analyze the current market sentiment for {company_name} ({ticker}). 
            Consider recent news, market trends, and industry outlook. 
            Provide a sentiment score from -2 (very negative) to +2 (very positive) and brief reasoning.
            
            Respond in JSON format:
            {{"score": 0.0, "reasoning": "Brief explanation", "outlook": "positive/neutral/negative"}}"""
            
            response = self.client.chat.completions.create(
                model="grok-2-1212",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "score": result.get("score", 0),
                "analysis": f"Sentiment: {result.get('outlook', 'neutral')}. {result.get('reasoning', '')}",
                "outlook": result.get("outlook", "neutral"),
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            return {
                "score": 0,
                "analysis": f"Sentiment analysis unavailable: {str(e)}",
                "outlook": "neutral",
                "reasoning": "Unable to analyze sentiment"
            }

class ManagerAgent:
    """Manager agent that synthesizes all analysis"""
    
    def __init__(self):
        self.name = "Portfolio Manager Agent"
        self.client = None
        if os.getenv('XAI_API_KEY'):
            self.client = OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=os.getenv('XAI_API_KEY')
            )
    
    def synthesize(self, ticker, company_name, technical_result, fundamental_result, sentiment_result):
        """Synthesize all agent results into final recommendation"""
        
        # Calculate weighted score
        tech_weight = 0.4
        fund_weight = 0.4
        sent_weight = 0.2
        
        combined_score = (
            technical_result["score"] * tech_weight +
            fundamental_result["score"] * fund_weight +
            sentiment_result["score"] * sent_weight
        )
        
        # Determine recommendation
        if combined_score >= 1.5:
            recommendation = "STRONG BUY"
        elif combined_score >= 0.5:
            recommendation = "BUY"
        elif combined_score >= -0.5:
            recommendation = "HOLD"
        elif combined_score >= -1.5:
            recommendation = "SELL"
        else:
            recommendation = "STRONG SELL"
        
        # Generate AI summary if available
        ai_summary = ""
        if self.client:
            try:
                summary_prompt = f"""As a portfolio manager, synthesize this multi-agent analysis for {company_name} ({ticker}):

Technical Analysis: {technical_result['analysis']}
Fundamental Analysis: {fundamental_result['analysis']}
Sentiment Analysis: {sentiment_result['analysis']}
Combined Score: {combined_score:.2f}
Recommendation: {recommendation}

Provide a concise 2-3 sentence investment thesis explaining the reasoning behind this recommendation."""

                response = self.client.chat.completions.create(
                    model="grok-2-1212",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=150
                )
                
                ai_summary = response.choices[0].message.content
                
            except Exception as e:
                ai_summary = f"Multi-agent analysis complete. Technical score: {technical_result['score']}, Fundamental score: {fundamental_result['score']}, Sentiment score: {sentiment_result['score']}."
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "recommendation": recommendation,
            "combined_score": combined_score,
            "confidence": min(100, abs(combined_score) * 30 + 50),  # Scale to percentage
            "technical": technical_result,
            "fundamental": fundamental_result,
            "sentiment": sentiment_result,
            "ai_summary": ai_summary or f"Multi-agent consensus: {recommendation} based on combined analysis.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

class MultiAgentSystem:
    """Main multi-agent system orchestrator"""
    
    def __init__(self):
        self.technical_agent = TechnicalAgent()
        self.fundamental_agent = FundamentalAgent()
        self.sentiment_agent = SentimentAgent()
        self.manager_agent = ManagerAgent()
    
    def analyze_stock(self, ticker):
        """Perform comprehensive multi-agent analysis"""
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            data = stock.history(period="3mo")
            info = stock.info
            
            if data.empty:
                return {"error": f"No data available for {ticker}"}
            
            company_name = info.get('longName', ticker)
            
            # Run agents in parallel (conceptually)
            technical_result = self.technical_agent.analyze(ticker, data)
            fundamental_result = self.fundamental_agent.analyze(ticker, info)
            sentiment_result = self.sentiment_agent.analyze(ticker, company_name)
            
            # Manager synthesizes results
            final_analysis = self.manager_agent.synthesize(
                ticker, company_name, technical_result, fundamental_result, sentiment_result
            )
            
            return final_analysis
            
        except Exception as e:
            return {"error": f"Analysis failed for {ticker}: {str(e)}"}
    
    def analyze_portfolio(self, tickers):
        """Analyze multiple stocks as a portfolio"""
        results = []
        for ticker in tickers:
            analysis = self.analyze_stock(ticker)
            if "error" not in analysis:
                results.append(analysis)
        
        # Sort by combined score
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return results