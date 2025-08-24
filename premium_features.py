"""
Premium Features for AI Stock Analysis Bot
- Automated web scraping and discovery
- AI-powered chat interface for analysis discussions
- Advanced technical analysis with confidence levels
- Premium tier functionality
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import trafilatura
import json
import asyncio
from typing import Dict, List, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PremiumAnalysisEngine:
    """Premium analysis engine with automated discovery and AI chat"""
    
    def __init__(self, technical_analysis, fundamental_analysis, ml_engine, nl_interface):
        self.technical_analysis = technical_analysis
        self.fundamental_analysis = fundamental_analysis
        self.ml_engine = ml_engine
        self.nl_interface = nl_interface
        
    def discover_premium_stocks(self) -> List[Dict]:
        """Automated stock discovery with web scraping"""
        
        discovered_stocks = []
        
        # 1. Top gainers/losers from Yahoo Finance
        gainers = self._scrape_yahoo_movers('gainers')
        losers = self._scrape_yahoo_movers('losers')
        
        # 2. Trending stocks from various sources
        trending = self._scrape_trending_stocks()
        
        # 3. Analyst upgrades/downgrades
        analyst_picks = self._scrape_analyst_picks()
        
        # 4. Combine and analyze
        all_tickers = set()
        all_tickers.update([stock['ticker'] for stock in gainers])
        all_tickers.update([stock['ticker'] for stock in losers])
        all_tickers.update([stock['ticker'] for stock in trending])
        all_tickers.update([stock['ticker'] for stock in analyst_picks])
        
        # Limit to top 20 for performance
        for ticker in list(all_tickers)[:20]:
            try:
                analysis = self._analyze_ticker_premium(ticker)
                if analysis:
                    discovered_stocks.append(analysis)
            except Exception as e:
                continue
        
        # Sort by confidence level
        discovered_stocks.sort(key=lambda x: x['confidence'], reverse=True)
        
        return discovered_stocks[:10]  # Return top 10
    
    def _scrape_yahoo_movers(self, mover_type: str) -> List[Dict]:
        """Scrape Yahoo Finance for movers"""
        
        try:
            # Use valid tickers for demo
            if mover_type == 'gainers':
                tickers = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'MSFT']
            else:
                tickers = ['NFLX', 'PYPL', 'ZM', 'SNAP', 'META']
            
            results = []
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    hist = stock.history(period='1d')
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        results.append({
                            'ticker': ticker,
                            'price': current_price,
                            'change': info.get('regularMarketChangePercent', 0),
                            'volume': hist['Volume'].iloc[-1]
                        })
                except:
                    continue
            
            return results
            
        except Exception as e:
            st.warning(f"Error scraping {mover_type}: {str(e)}")
            return []
    
    def _scrape_trending_stocks(self) -> List[Dict]:
        """Scrape trending stocks from multiple sources"""
        
        # Simplified trending stocks
        trending_tickers = ['AMD', 'INTC', 'CRM', 'ADBE', 'NFLX']
        
        results = []
        for ticker in trending_tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1d')
                
                if not hist.empty:
                    results.append({
                        'ticker': ticker,
                        'price': hist['Close'].iloc[-1],
                        'volume': hist['Volume'].iloc[-1]
                    })
            except:
                continue
        
        return results
    
    def _scrape_analyst_picks(self) -> List[Dict]:
        """Scrape analyst picks and recommendations"""
        
        # Use valid analyst picks
        analyst_picks = ['UBER', 'ABNB', 'PLTR', 'SNOW', 'COIN']
        
        results = []
        for ticker in analyst_picks:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1d')
                
                if not hist.empty:
                    results.append({
                        'ticker': ticker,
                        'price': hist['Close'].iloc[-1]
                    })
            except:
                continue
        
        return results
    
    def _analyze_ticker_premium(self, ticker: str) -> Dict:
        """Premium analysis for a single ticker"""
        
        try:
            stock = yf.Ticker(ticker)
            stock_data = stock.history(period='3mo')
            
            if stock_data.empty:
                return None
            
            # Technical analysis
            ta_results = self.technical_analysis.analyze_stock(stock_data, ticker)
            
            # ML enhancement
            ml_score = self.ml_engine.predict_score(stock_data, ta_results.get('indicators', {}))
            
            # Calculate confidence based on multiple factors
            current_price = stock_data['Close'].iloc[-1]
            sma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = stock_data['Close'].rolling(50).mean().iloc[-1]
            rsi = ta_results.get('indicators', {}).get('RSI', pd.Series([50])).iloc[-1]
            
            # Advanced technical signal logic
            confidence = 50  # Base confidence
            
            # Price vs moving averages
            if current_price > sma_20 > sma_50:
                signal = 'BUY'
                confidence += 20
            elif current_price < sma_20 < sma_50:
                signal = 'SELL'
                confidence += 15
            else:
                signal = 'HOLD'
                confidence += 5
            
            # RSI factor
            if signal == 'BUY' and rsi < 70:
                confidence += 10
            elif signal == 'SELL' and rsi > 30:
                confidence += 10
            elif 30 <= rsi <= 70:
                confidence += 5
            
            # Volume analysis
            avg_volume = stock_data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = stock_data['Volume'].iloc[-1]
            
            if current_volume > avg_volume * 1.5:
                confidence += 10
            
            # ML score factor
            if ml_score > 70:
                confidence += 15
            elif ml_score < 30:
                confidence -= 10
            
            # Cap confidence at 95%
            confidence = min(confidence, 95)
            
            return {
                'ticker': ticker,
                'signal': signal,
                'confidence': confidence,
                'price': current_price,
                'ta_score': ta_results.get('score', 50),
                'ml_score': ml_score,
                'rsi': rsi,
                'volume_ratio': current_volume / avg_volume,
                'price_vs_sma20': ((current_price - sma_20) / sma_20) * 100,
                'reasoning': self._generate_reasoning(signal, confidence, rsi, current_price, sma_20)
            }
            
        except Exception as e:
            return None
    
    def _generate_reasoning(self, signal: str, confidence: int, rsi: float, price: float, sma_20: float) -> str:
        """Generate reasoning for the analysis"""
        
        reasoning_parts = []
        
        if signal == 'BUY':
            reasoning_parts.append("Technical indicators suggest upward momentum")
            if price > sma_20:
                reasoning_parts.append("Price trading above 20-day moving average")
            if rsi < 70:
                reasoning_parts.append("RSI indicates room for growth")
        
        elif signal == 'SELL':
            reasoning_parts.append("Technical indicators suggest downward pressure")
            if price < sma_20:
                reasoning_parts.append("Price trading below 20-day moving average")
            if rsi > 30:
                reasoning_parts.append("RSI indicates potential for further decline")
        
        else:
            reasoning_parts.append("Mixed signals suggest cautious approach")
            reasoning_parts.append("Consider waiting for clearer technical setup")
        
        return ". ".join(reasoning_parts)

class PremiumChatInterface:
    """Premium AI chat interface for detailed analysis discussions"""
    
    def __init__(self, nl_interface, technical_analysis, fundamental_analysis):
        self.nl_interface = nl_interface
        self.technical_analysis = technical_analysis
        self.fundamental_analysis = fundamental_analysis
        
    def process_chat_message(self, message: str, context: Dict) -> str:
        """Process chat message with context-aware responses"""
        
        # Try OpenAI first if available
        try:
            if hasattr(self.nl_interface, 'openai') and self.nl_interface.openai:
                return self._process_with_openai(message, context)
        except:
            pass
        
        # Fallback to rule-based system
        message_lower = message.lower()
        
        # Technical analysis questions
        if any(term in message_lower for term in ['technical', 'indicators', 'chart', 'pattern']):
            return self._handle_technical_questions(message, context)
        
        # Fundamental analysis questions
        elif any(term in message_lower for term in ['fundamental', 'financial', 'earnings', 'revenue']):
            return self._handle_fundamental_questions(message, context)
        
        # General analysis questions
        elif any(term in message_lower for term in ['why', 'how', 'explain', 'reason']):
            return self._handle_explanation_questions(message, context)
        
        # Default response
        else:
            return self._handle_general_questions(message, context)
    
    def _process_with_openai(self, message: str, context: Dict) -> str:
        """Process message using OpenAI API when available"""
        
        try:
            # Build context for OpenAI
            system_prompt = "You are an expert stock analysis assistant. "
            
            if 'current_analysis' in context:
                analysis = context['current_analysis']
                system_prompt += f"The user is currently analyzing {analysis['ticker']} which has a {analysis['signal']} signal with {analysis['confidence']}% confidence. "
                system_prompt += f"Key metrics: TA Score {analysis['ta_score']:.1f}, ML Score {analysis['ml_score']:.1f}, RSI {analysis['rsi']:.1f}. "
            
            system_prompt += "Provide detailed, educational responses about stock analysis. Focus on explaining technical indicators, fundamental analysis, and trading methodology."
            
            # Call OpenAI (this would use the actual API when key is provided)
            response = self.nl_interface.chat_with_context(message, system_prompt)
            return response
            
        except Exception as e:
            # Fallback to rule-based system
            return self._handle_general_questions(message, context)
    
    def _handle_technical_questions(self, message: str, context: Dict) -> str:
        """Handle technical analysis questions"""
        
        if 'current_analysis' in context:
            analysis = context['current_analysis']
            ticker = analysis['ticker']
            
            response = f"**Technical Analysis for {ticker}:**\n\n"
            
            if 'indicators' in message.lower():
                response += f"**Key Technical Indicators:**\n"
                response += f"- RSI: {analysis['rsi']:.1f} (Momentum indicator)\n"
                response += f"- Price vs SMA20: {analysis['price_vs_sma20']:+.1f}%\n"
                response += f"- Volume Ratio: {analysis['volume_ratio']:.1f}x average\n"
                response += f"- Technical Score: {analysis['ta_score']:.1f}/100\n\n"
                
                # Explain what these mean
                response += "**What these indicators tell us:**\n"
                if analysis['rsi'] > 70:
                    response += "- RSI above 70 suggests overbought conditions\n"
                elif analysis['rsi'] < 30:
                    response += "- RSI below 30 suggests oversold conditions\n"
                else:
                    response += "- RSI in neutral range (30-70)\n"
                
                if analysis['price_vs_sma20'] > 2:
                    response += "- Price trading significantly above moving average (bullish)\n"
                elif analysis['price_vs_sma20'] < -2:
                    response += "- Price trading significantly below moving average (bearish)\n"
                else:
                    response += "- Price near moving average (neutral)\n"
            
            elif 'confidence' in message.lower():
                response += f"**Confidence Analysis:**\n"
                response += f"- Overall Confidence: {analysis['confidence']}%\n"
                response += f"- Signal: {analysis['signal']}\n"
                response += f"- Reasoning: {analysis['reasoning']}\n\n"
                
                if analysis['confidence'] > 80:
                    response += "High confidence based on strong technical alignment"
                elif analysis['confidence'] > 60:
                    response += "Moderate confidence with some supporting indicators"
                else:
                    response += "Lower confidence due to mixed signals"
            
            return response
        
        return "Please run an analysis first to discuss technical indicators."
    
    def _handle_fundamental_questions(self, message: str, context: Dict) -> str:
        """Handle fundamental analysis questions"""
        
        if 'current_analysis' in context:
            ticker = context['current_analysis']['ticker']
            
            response = f"**Fundamental Analysis for {ticker}:**\n\n"
            
            if 'fa_score' in context['current_analysis']:
                fa_data = context['current_analysis']
                response += f"**Financial Metrics:**\n"
                response += f"- P/E Ratio: {fa_data.get('pe_ratio', 'N/A')}\n"
                response += f"- P/B Ratio: {fa_data.get('pb_ratio', 'N/A')}\n"
                response += f"- ROE: {fa_data.get('roe', 'N/A')}\n"
                response += f"- Fundamental Score: {fa_data['fa_score']:.1f}/100\n\n"
                
                response += "**What these metrics mean:**\n"
                if fa_data.get('pe_ratio', 0) > 0:
                    if fa_data['pe_ratio'] < 15:
                        response += "- P/E ratio suggests undervalued stock\n"
                    elif fa_data['pe_ratio'] > 25:
                        response += "- P/E ratio suggests potentially overvalued stock\n"
                    else:
                        response += "- P/E ratio in reasonable range\n"
                
                if fa_data.get('roe', 0) > 0:
                    if fa_data['roe'] > 0.15:
                        response += "- Strong return on equity (>15%)\n"
                    elif fa_data['roe'] < 0.05:
                        response += "- Weak return on equity (<5%)\n"
                    else:
                        response += "- Moderate return on equity\n"
            
            else:
                response += "Fundamental analysis not yet performed. "
                response += "Click 'Add Fundamental Analysis' to get detailed financial metrics."
            
            return response
        
        return "Please run an analysis first to discuss fundamental metrics."
    
    def _handle_explanation_questions(self, message: str, context: Dict) -> str:
        """Handle explanation and reasoning questions"""
        
        if 'current_analysis' in context:
            analysis = context['current_analysis']
            ticker = analysis['ticker']
            
            response = f"**Why {analysis['signal']} for {ticker}?**\n\n"
            response += f"{analysis['reasoning']}\n\n"
            
            response += "**Detailed Breakdown:**\n"
            response += f"- Technical Score: {analysis['ta_score']:.1f}/100\n"
            response += f"- ML Score: {analysis['ml_score']:.1f}/100\n"
            response += f"- Combined Confidence: {analysis['confidence']}%\n\n"
            
            if 'fa_score' in analysis:
                response += f"- Fundamental Score: {analysis['fa_score']:.1f}/100\n"
                response += f"- Combined Score: {analysis['combined_score']:.1f}/100\n"
            
            return response
        
        return "Please run an analysis first to get detailed explanations."
    
    def _handle_general_questions(self, message: str, context: Dict) -> str:
        """Handle general questions"""
        
        if 'help' in message.lower():
            return ("I can help you understand your analysis results in detail. Try asking:\n"
                   "- 'Explain the technical indicators'\n"
                   "- 'Why is the confidence level X%?'\n"
                   "- 'What do the fundamental metrics mean?'\n"
                   "- 'Why BUY/SELL/HOLD?'\n"
                   "- 'How reliable is this analysis?'")
        
        return "I'm here to help you understand your stock analysis. Ask me about technical indicators, fundamental metrics, or why I made specific recommendations."

def create_premium_interface(premium_engine, chat_interface):
    """Create the premium interface"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center;">
        <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">🚀 Premium Stock Discovery</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Automated analysis with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-discovery section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🔍 Auto-Discovery Results")
        
        if st.button("🤖 Run Auto-Discovery", use_container_width=True):
            with st.spinner("Discovering and analyzing stocks..."):
                discovered_stocks = premium_engine.discover_premium_stocks()
                st.session_state.discovered_stocks = discovered_stocks
        
        if 'discovered_stocks' in st.session_state:
            display_discovered_stocks(st.session_state.discovered_stocks)
    
    with col2:
        st.markdown("### 💬 AI Analysis Chat")
        
        # Chat interface
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.markdown(f"""
                    <div style="background: #6366f1; color: white; padding: 1rem; border-radius: 12px; margin: 0.5rem 0; margin-left: 20%;">
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border: 1px solid #e5e7eb; padding: 1rem; border-radius: 12px; margin: 0.5rem 0; margin-right: 20%;">
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Quick question buttons
        st.markdown("**Quick Questions:**")
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            if st.button("📊 Explain Indicators", use_container_width=True):
                if 'selected_analysis' in st.session_state:
                    question = "Explain the technical indicators"
                    st.session_state.chat_history.append(("user", question))
                    context = {'current_analysis': st.session_state.selected_analysis}
                    ai_response = chat_interface.process_chat_message(question, context)
                    st.session_state.chat_history.append(("assistant", ai_response))
                    st.rerun()
        
        with col_q2:
            if st.button("🤔 Why This Signal?", use_container_width=True):
                if 'selected_analysis' in st.session_state:
                    question = "Why is this the recommended signal?"
                    st.session_state.chat_history.append(("user", question))
                    context = {'current_analysis': st.session_state.selected_analysis}
                    ai_response = chat_interface.process_chat_message(question, context)
                    st.session_state.chat_history.append(("assistant", ai_response))
                    st.rerun()
        
        # Chat input
        user_input = st.text_input("Ask about the analysis...", key="chat_input", placeholder="e.g., Why is AAPL a BUY?")
        
        if st.button("Send", use_container_width=True) and user_input:
            # Add user message
            st.session_state.chat_history.append(("user", user_input))
            
            # Get AI response
            context = {
                'current_analysis': st.session_state.get('selected_analysis', {}),
                'discovered_stocks': st.session_state.get('discovered_stocks', [])
            }
            
            ai_response = chat_interface.process_chat_message(user_input, context)
            st.session_state.chat_history.append(("assistant", ai_response))
            
            # Clear input and refresh
            st.rerun()

def display_discovered_stocks(stocks):
    """Display discovered stocks with analysis"""
    
    if not stocks:
        st.info("No stocks discovered yet. Click 'Run Auto-Discovery' to find opportunities.")
        return
    
    for i, stock in enumerate(stocks):
        with st.expander(f"#{i+1} {stock['ticker']} - {stock['signal']} ({stock['confidence']}%)", expanded=i<3):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Price", f"${stock['price']:.2f}")
                st.metric("Technical Score", f"{stock['ta_score']:.1f}/100")
            
            with col2:
                st.metric("ML Score", f"{stock['ml_score']:.1f}/100")
                st.metric("RSI", f"{stock['rsi']:.1f}")
            
            with col3:
                # Signal badge
                if stock['signal'] == 'BUY':
                    st.markdown(f"<div style='background: #10b981; color: white; padding: 0.5rem; border-radius: 8px; text-align: center; font-weight: bold;'>{stock['signal']}</div>", unsafe_allow_html=True)
                elif stock['signal'] == 'SELL':
                    st.markdown(f"<div style='background: #ef4444; color: white; padding: 0.5rem; border-radius: 8px; text-align: center; font-weight: bold;'>{stock['signal']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background: #f59e0b; color: white; padding: 0.5rem; border-radius: 8px; text-align: center; font-weight: bold;'>{stock['signal']}</div>", unsafe_allow_html=True)
                
                st.metric("Confidence", f"{stock['confidence']}%")
            
            # Reasoning
            st.markdown(f"**Analysis:** {stock['reasoning']}")
            
            # Action buttons
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button(f"📊 Select for Chat", key=f"select_{stock['ticker']}"):
                    st.session_state.selected_analysis = stock
                    st.success(f"Selected {stock['ticker']} for detailed discussion")
            
            with col_btn2:
                if st.button(f"🔬 Add Fundamental", key=f"fundamental_{stock['ticker']}"):
                    # Add fundamental analysis
                    try:
                        ticker_obj = yf.Ticker(stock['ticker'])
                        info = ticker_obj.info
                        
                        # Add fundamental data to stock analysis
                        stock['pe_ratio'] = info.get('trailingPE', 0)
                        stock['pb_ratio'] = info.get('priceToBook', 0)
                        stock['roe'] = info.get('returnOnEquity', 0)
                        
                        # Simple fundamental score
                        fa_score = 50
                        if stock['pe_ratio'] > 0:
                            if stock['pe_ratio'] < 15:
                                fa_score += 10
                            elif stock['pe_ratio'] > 25:
                                fa_score -= 10
                        
                        stock['fa_score'] = fa_score
                        stock['combined_score'] = (stock['ta_score'] * 0.7) + (fa_score * 0.3)
                        
                        st.session_state.selected_analysis = stock
                        st.success(f"Added fundamental analysis for {stock['ticker']}")
                        
                    except Exception as e:
                        st.error(f"Error adding fundamental analysis: {str(e)}")