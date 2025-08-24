"""
Pure Chat Interface with OpenAI Integration
- Full-screen chat interface
- Small recommendations panel
- Real-time stock analysis through conversation
- Fundamental analysis continuation from recommendations
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import os
from realtime_data_manager import data_manager

# Set page config for full width
st.set_page_config(
    page_title="AI Stock Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class GrokStockAnalyst:
    """Grok-powered stock analysis chatbot"""
    
    def __init__(self):
        self.grok_available = self._check_grok_availability()
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'daily_recommendations' not in st.session_state:
            st.session_state.daily_recommendations = []
        if 'analysis_context' not in st.session_state:
            st.session_state.analysis_context = {}
    
    def _check_grok_availability(self) -> bool:
        """Check if Grok/xAI API key is available"""
        try:
            api_key = os.getenv('XAI_API_KEY')
            return api_key is not None and api_key.strip() != ""
        except:
            return False
    
    def process_message(self, message: str) -> str:
        """Process user message with Grok or fallback"""
        
        if self.grok_available:
            return self._process_with_grok(message)
        else:
            return self._process_with_fallback(message)
    
    def _process_with_grok(self, message: str) -> str:
        """Process message using Grok API"""
        
        try:
            from openai import OpenAI
            
            # Create Grok client using xAI endpoint
            client = OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=os.getenv('XAI_API_KEY')
            )
            
            # Build context from chat history and recommendations
            context = self._build_context()
            
            system_prompt = f"""You are an expert stock analysis assistant powered by Grok. You provide:
1. Real-time stock analysis using live market data
2. Technical and fundamental analysis explanations
3. Investment recommendations based on current market conditions
4. Educational content about trading and investing

Current context: {context}

Always provide specific, actionable insights. When analyzing stocks, use current market data and explain your reasoning clearly. Be conversational and insightful."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            # Add recent chat history for context
            recent_history = st.session_state.chat_history[-6:]  # Last 3 exchanges
            for role, content in recent_history:
                messages.insert(-1, {"role": "assistant" if role == "ai" else "user", "content": content})
            
            response = client.chat.completions.create(
                model="grok-2-1212",  # Latest Grok model
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Check if this is a stock analysis request
            if self._is_stock_request(message):
                ticker = self._extract_ticker(message)
                if ticker:
                    market_data = self._get_market_data_context(ticker)
                    ai_response += f"\n\n{market_data}"
            
            return ai_response
            
        except Exception as e:
            return f"Grok API error: {str(e)}. Using fallback analysis."
    
    def _process_with_fallback(self, message: str) -> str:
        """Fallback processing when OpenAI is not available"""
        
        message_lower = message.lower()
        
        # Stock analysis requests
        if any(ticker in message_lower for ticker in ['aapl', 'tsla', 'nvda', 'googl', 'msft', 'amzn', 'meta']):
            ticker = self._extract_ticker(message)
            if ticker:
                return self._analyze_stock_fallback(ticker)
        
        # General investment questions
        if any(term in message_lower for term in ['invest', 'buy', 'sell', 'portfolio', 'market']):
            return self._provide_general_investment_advice(message)
        
        # Technical analysis questions
        if any(term in message_lower for term in ['rsi', 'moving average', 'technical', 'chart']):
            return self._explain_technical_concepts(message)
        
        # Recommendations request
        if any(term in message_lower for term in ['recommend', 'suggestion', 'best stocks']):
            return self._generate_recommendations()
        
        # Default response
        return self._provide_helpful_response(message)
    
    def _analyze_stock_fallback(self, ticker: str) -> str:
        """Provide stock analysis using fallback logic"""
        
        try:
            # Get live data
            quote = data_manager.get_real_time_quote(ticker)
            stock_data = data_manager.get_live_stock_data(ticker, period='3mo')
            
            if 'error' in quote or stock_data is None:
                return f"Unable to fetch current data for {ticker}. Please try again or check the ticker symbol."
            
            # Basic technical analysis
            current_price = quote['current_price']
            change_percent = quote.get('change_percent', 0)
            
            # Calculate simple indicators
            sma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = stock_data['Close'].rolling(50).mean().iloc[-1]
            
            # Generate analysis
            analysis = f"**Analysis for {ticker}**\n\n"
            analysis += f"📊 **Current Price**: ${current_price:.2f}\n"
            
            if change_percent > 0:
                analysis += f"🟢 **Change**: +{change_percent:.1f}% (up today)\n"
            elif change_percent < 0:
                analysis += f"🔴 **Change**: {change_percent:.1f}% (down today)\n"
            else:
                analysis += f"⚪ **Change**: {change_percent:.1f}% (flat)\n"
            
            analysis += f"📈 **20-day SMA**: ${sma_20:.2f}\n"
            analysis += f"📈 **50-day SMA**: ${sma_50:.2f}\n\n"
            
            # Technical signals
            if current_price > sma_20 > sma_50:
                analysis += "**Technical Signal**: 🟢 BULLISH\n"
                analysis += "- Price above both moving averages\n"
                analysis += "- Short-term momentum positive\n"
            elif current_price < sma_20 < sma_50:
                analysis += "**Technical Signal**: 🔴 BEARISH\n"
                analysis += "- Price below both moving averages\n"
                analysis += "- Downward trend indicated\n"
            else:
                analysis += "**Technical Signal**: ⚪ NEUTRAL\n"
                analysis += "- Mixed signals from moving averages\n"
                analysis += "- Wait for clearer direction\n"
            
            analysis += f"\n**Volume**: {quote.get('volume', 'N/A'):,} shares\n"
            analysis += f"**Day Range**: ${quote.get('low', 0):.2f} - ${quote.get('high', 0):.2f}\n"
            
            # Add to analysis context
            st.session_state.analysis_context[ticker] = {
                'price': current_price,
                'change_percent': change_percent,
                'signal': 'BULLISH' if current_price > sma_20 > sma_50 else 'BEARISH' if current_price < sma_20 < sma_50 else 'NEUTRAL',
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing {ticker}: {str(e)}"
    
    def _generate_recommendations(self) -> str:
        """Generate daily stock recommendations"""
        
        try:
            # Get trending stocks
            trending = data_manager.get_trending_stocks()
            
            recommendations = []
            
            for stock in trending[:5]:  # Top 5 trending
                ticker = stock['ticker']
                
                # Quick analysis
                if abs(stock['change_percent']) > 2:  # Significant movement
                    if stock['change_percent'] > 0:
                        signal = "BUY" if stock['change_percent'] > 5 else "WATCH"
                        reason = f"Strong upward momentum (+{stock['change_percent']:.1f}%)"
                    else:
                        signal = "SELL" if stock['change_percent'] < -5 else "WATCH"
                        reason = f"Downward pressure ({stock['change_percent']:.1f}%)"
                else:
                    signal = "HOLD"
                    reason = f"Stable movement ({stock['change_percent']:+.1f}%)"
                
                recommendations.append({
                    'ticker': ticker,
                    'signal': signal,
                    'price': stock['price'],
                    'change_percent': stock['change_percent'],
                    'reason': reason
                })
            
            # Store recommendations
            st.session_state.daily_recommendations = recommendations
            
            # Format response
            response = "**Today's Stock Recommendations**\n\n"
            
            for i, rec in enumerate(recommendations, 1):
                signal_emoji = "🟢" if rec['signal'] == 'BUY' else "🔴" if rec['signal'] == 'SELL' else "🟡" if rec['signal'] == 'WATCH' else "⚪"
                response += f"{i}. **{rec['ticker']}** {signal_emoji} {rec['signal']}\n"
                response += f"   Price: ${rec['price']:.2f} ({rec['change_percent']:+.1f}%)\n"
                response += f"   Reason: {rec['reason']}\n\n"
            
            response += "💡 **Ask me about any stock for detailed fundamental analysis!**\n"
            response += "Example: 'Tell me more about AAPL fundamentals' or 'Should I buy TSLA?'"
            
            return response
            
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def _provide_general_investment_advice(self, message: str) -> str:
        """Provide general investment guidance"""
        
        advice = "**Investment Guidance**\n\n"
        
        if 'portfolio' in message.lower():
            advice += "**Portfolio Management Tips:**\n"
            advice += "- Diversify across sectors and asset classes\n"
            advice += "- Set clear risk tolerance levels\n"
            advice += "- Regular rebalancing (quarterly or semi-annually)\n"
            advice += "- Keep 3-6 months emergency fund before investing\n"
        elif 'buy' in message.lower():
            advice += "**Before Buying Stocks:**\n"
            advice += "- Research the company's financials\n"
            advice += "- Check recent news and analyst ratings\n"
            advice += "- Consider the overall market conditions\n"
            advice += "- Have a clear exit strategy\n"
        elif 'sell' in message.lower():
            advice += "**Selling Decisions:**\n"
            advice += "- Take profits when targets are reached\n"
            advice += "- Cut losses at predetermined levels\n"
            advice += "- Consider tax implications\n"
            advice += "- Don't let emotions drive decisions\n"
        else:
            advice += "**General Investment Principles:**\n"
            advice += "- Time in market > timing the market\n"
            advice += "- Dollar-cost averaging for regular investments\n"
            advice += "- Stay informed but avoid overtrading\n"
            advice += "- Focus on long-term wealth building\n"
        
        advice += "\n💬 **Ask me specific questions about any stock or investment strategy!**"
        
        return advice
    
    def _explain_technical_concepts(self, message: str) -> str:
        """Explain technical analysis concepts"""
        
        message_lower = message.lower()
        
        if 'rsi' in message_lower:
            return """**RSI (Relative Strength Index)**

📊 **What it is**: Measures momentum on a 0-100 scale

📈 **How to read it**:
- Above 70: Potentially overbought (consider selling)
- Below 30: Potentially oversold (consider buying)
- 50: Neutral momentum

🎯 **Trading signals**:
- RSI divergence from price can signal reversals
- Works best in ranging markets
- Combine with other indicators for confirmation"""

        elif 'moving average' in message_lower or 'sma' in message_lower:
            return """**Moving Averages**

📊 **Simple Moving Average (SMA)**: Average price over N periods

📈 **Common periods**:
- 20-day: Short-term trend
- 50-day: Medium-term trend  
- 200-day: Long-term trend

🎯 **Trading signals**:
- Price above SMA = Bullish trend
- Price below SMA = Bearish trend
- Golden Cross: 50 SMA crosses above 200 SMA (bullish)
- Death Cross: 50 SMA crosses below 200 SMA (bearish)"""

        else:
            return """**Technical Analysis Basics**

📊 **Key Indicators**:
- Moving Averages: Trend direction
- RSI: Momentum and overbought/oversold
- Volume: Confirms price movements
- Support/Resistance: Key price levels

🎯 **How to use**:
- Combine multiple indicators
- Look for confirmation signals
- Consider timeframes (short vs long-term)
- Practice risk management

💡 **Ask me about specific indicators for detailed explanations!**"""
    
    def _provide_helpful_response(self, message: str) -> str:
        """Provide helpful default response"""
        
        return """**I'm your Grok-powered Stock Analysis Assistant!**

**What I can help with**:
- Real-time stock analysis (e.g., "analyze AAPL")
- Daily stock recommendations
- Technical indicator explanations
- Investment strategy guidance
- Fundamental analysis discussions

**Try asking**:
- "Give me today's recommendations"
- "Analyze Tesla stock"
- "Explain RSI indicator"
- "Should I buy Microsoft?"
- "Tell me about NVDA fundamentals"

**I use live market data to provide current analysis and insights!**"""
    
    def _build_context(self) -> str:
        """Build context string for Grok"""
        
        context_parts = []
        
        # Add recent recommendations
        if st.session_state.daily_recommendations:
            context_parts.append(f"Today's recommendations: {len(st.session_state.daily_recommendations)} stocks analyzed")
        
        # Add recent analysis
        if st.session_state.analysis_context:
            recent_tickers = list(st.session_state.analysis_context.keys())[-3:]
            context_parts.append(f"Recently analyzed: {', '.join(recent_tickers)}")
        
        # Add market status
        market_status = data_manager.get_market_status()
        if market_status.get('is_open'):
            context_parts.append("Market is currently open")
        else:
            context_parts.append("Market is currently closed")
        
        return ". ".join(context_parts) if context_parts else "Starting new analysis session"
    
    def _is_stock_request(self, message: str) -> bool:
        """Check if message is requesting stock analysis"""
        keywords = ['analyze', 'analysis', 'stock', 'price', 'buy', 'sell', 'fundamentals']
        return any(keyword in message.lower() for keyword in keywords)
    
    def _extract_ticker(self, message: str) -> str:
        """Extract ticker symbol from message"""
        common_tickers = ['AAPL', 'TSLA', 'NVDA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX', 'CRM', 'AMD']
        message_upper = message.upper()
        
        for ticker in common_tickers:
            if ticker in message_upper:
                return ticker
        
        # Look for potential ticker patterns (3-5 uppercase letters)
        import re
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        matches = re.findall(ticker_pattern, message.upper())
        
        for match in matches:
            if data_manager.validate_ticker(match):
                return match
        
        return None
    
    def _get_market_data_context(self, ticker: str) -> str:
        """Get current market data context for a ticker"""
        
        try:
            quote = data_manager.get_real_time_quote(ticker)
            freshness = data_manager.get_data_freshness(ticker)
            
            if 'error' not in quote:
                context = f"\n**Live Market Data for {ticker}:**\n"
                context += f"Current Price: ${quote['current_price']:.2f}\n"
                context += f"Change: {quote['change']:+.2f} ({quote['change_percent']:+.1f}%)\n"
                context += f"Volume: {quote['volume']:,}\n"
                context += f"Data freshness: {freshness.get('minutes_old', 0)} minutes old"
                return context
            
        except Exception as e:
            pass
        
        return ""

def create_chat_interface(db, data_retrieval, technical_analysis, fundamental_analysis, 
                         recommendation_engine, ml_engine, nl_interface, scheduler, user_tier):
    """Create pure chat interface with Grok integration"""
    
    # Initialize chatbot
    chatbot = GrokStockAnalyst()
    
    # Clean, modern CSS for chat interface
    st.markdown("""
    <style>
    /* Hide Streamlit elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stDecoration {display:none;}
    
    /* Main container */
    .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    
    /* Chat messages styling */
    .chat-message {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 18px;
        max-width: 75%;
        word-wrap: break-word;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    
    .user-message {
        background: #007bff;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .ai-message {
        background: #f1f3f4;
        color: #333;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border: 1px solid #e8eaed;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 2px solid #e8eaed;
        padding: 12px 20px;
        font-size: 1rem;
        transition: border-color 0.2s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        background: #007bff;
        color: white;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
        transition: background 0.2s;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        background: #0056b3;
    }
    
    /* Quick action buttons */
    .quick-btn {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        color: #495057;
        border-radius: 16px;
        padding: 8px 16px;
        margin: 4px;
        font-size: 0.85rem;
        transition: all 0.2s;
    }
    
    .quick-btn:hover {
        background: #e9ecef;
        border-color: #adb5bd;
    }
    
    /* Recommendations panel */
    .rec-panel {
        background: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .rec-item {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 3px solid #007bff;
        transition: background 0.2s;
    }
    
    .rec-item:hover {
        background: #e9ecef;
    }
    
    /* Status indicator */
    .status-indicator {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 6px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        z-index: 1000;
    }
    
    .status-connected {
        background: #d1f2eb;
        color: #0e5429;
        border: 1px solid #7dcea0;
    }
    
    .status-fallback {
        background: #fdf2e9;
        color: #b7472a;
        border: 1px solid #f1948a;
    }
    
    /* Chat container */
    .chat-container {
        height: 65vh;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 12px;
        background: #ffffff;
    }
    
    /* Hide labels */
    .stTextInput label {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Status indicator
    if chatbot.grok_available:
        st.markdown('<div class="status-indicator status-connected">Grok Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-indicator status-fallback">Offline Mode</div>', unsafe_allow_html=True)
    
    # Layout: 75% chat, 25% recommendations
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    {message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message ai-message">
                    {message}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        col_input, col_send = st.columns([5, 1])
        
        with col_input:
            user_input = st.text_input(
                "chat_input",
                placeholder="Ask about any stock or investment topic...",
                key="chat_input",
                label_visibility="collapsed"
            )
        
        with col_send:
            send_button = st.button("💬", use_container_width=True)
        
        # Quick action buttons
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        
        with col_q1:
            if st.button("Recommendations", use_container_width=True):
                user_input = "Give me today's stock recommendations"
                send_button = True
        
        with col_q2:
            if st.button("AAPL Analysis", use_container_width=True):
                user_input = "Analyze Apple stock"
                send_button = True
        
        with col_q3:
            if st.button("Market Status", use_container_width=True):
                user_input = "What's the current market status?"
                send_button = True
        
        with col_q4:
            if st.button("Learn RSI", use_container_width=True):
                user_input = "Explain RSI indicator"
                send_button = True
        
        # Process message
        if send_button and user_input:
            # Add user message
            st.session_state.chat_history.append(("user", user_input))
            
            # Get AI response
            with st.spinner("Analyzing..."):
                ai_response = chatbot.process_message(user_input)
                st.session_state.chat_history.append(("ai", ai_response))
            
            st.rerun()
    
    with col2:
        # Compact recommendations panel
        st.markdown('<div class="rec-panel">', unsafe_allow_html=True)
        
        if st.button("↻ Daily Picks", use_container_width=True):
            with st.spinner("Getting picks..."):
                response = chatbot._generate_recommendations()
                st.session_state.chat_history.append(("ai", response))
            st.rerun()
        
        # Display current recommendations
        if st.session_state.daily_recommendations:
            for rec in st.session_state.daily_recommendations[:4]:  # Show top 4
                signal_color = "#28a745" if rec['signal'] == 'BUY' else "#dc3545" if rec['signal'] == 'SELL' else "#ffc107"
                signal_dot = "●"
                
                if st.button(f"{rec['ticker']} {signal_dot}", key=f"rec_{rec['ticker']}", use_container_width=True):
                    question = f"Tell me about {rec['ticker']} - should I {rec['signal'].lower()}?"
                    st.session_state.chat_history.append(("user", question))
                    with st.spinner("Analyzing..."):
                        ai_response = chatbot.process_message(question)
                        st.session_state.chat_history.append(("ai", ai_response))
                    st.rerun()
                
                st.markdown(f"""
                <div style="color: {signal_color}; font-size: 0.8rem; margin-bottom: 8px; text-align: center;">
                    {rec['signal']} • ${rec['price']:.2f} • {rec['change_percent']:+.1f}%
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("Click **Daily Picks** for recommendations")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent analysis (compact)
        if st.session_state.analysis_context:
            st.markdown("**Recent**")
            for ticker, data in list(st.session_state.analysis_context.items())[-2:]:
                signal_emoji = "🟢" if data['signal'] == 'BULLISH' else "🔴" if data['signal'] == 'BEARISH' else "⚪"
                
                if st.button(f"{ticker} {signal_emoji}", key=f"recent_{ticker}", use_container_width=True):
                    question = f"More about {ticker} fundamentals"
                    st.session_state.chat_history.append(("user", question))
                    with st.spinner("Analyzing..."):
                        ai_response = chatbot.process_message(question)
                        st.session_state.chat_history.append(("ai", ai_response))
                    st.rerun()
        
        # Clear option
        if len(st.session_state.chat_history) > 5:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.analysis_context = {}
                st.rerun()

# Export the main function
def create_enhanced_interface(db, data_retrieval, technical_analysis, fundamental_analysis, 
                             recommendation_engine, ml_engine, nl_interface, scheduler, user_tier):
    """Main interface function for compatibility"""
    return create_chat_interface(db, data_retrieval, technical_analysis, fundamental_analysis, 
                                recommendation_engine, ml_engine, nl_interface, scheduler, user_tier)