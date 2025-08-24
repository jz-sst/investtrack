"""
Enhanced Full-Screen UI with Persistent AI Chatbot
- AI tracks all user interactions and decisions
- Full-screen utilization with better layouts
- Configurable analysis rules and parameters
- Free and Premium tier functionality
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from realtime_data_manager import data_manager
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import uuid

# Set page config for full width
st.set_page_config(
    page_title="AI Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InteractionTracker:
    """Track all user interactions for AI chatbot context"""
    
    def __init__(self):
        if 'interaction_history' not in st.session_state:
            st.session_state.interaction_history = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())[:8]
    
    def log_interaction(self, action_type: str, details: Dict):
        """Log user interaction"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'session_id': st.session_state.session_id,
            'action_type': action_type,
            'details': details
        }
        st.session_state.interaction_history.append(interaction)
        
        # Keep only last 100 interactions to manage memory
        if len(st.session_state.interaction_history) > 100:
            st.session_state.interaction_history = st.session_state.interaction_history[-100:]
    
    def get_context_summary(self) -> str:
        """Get summary of recent interactions for AI context"""
        if not st.session_state.interaction_history:
            return "No previous interactions in this session."
        
        recent_interactions = st.session_state.interaction_history[-10:]
        
        summary = f"Session {st.session_state.session_id} - Recent Activity:\n"
        for interaction in recent_interactions:
            time_str = datetime.fromisoformat(interaction['timestamp']).strftime("%H:%M")
            summary += f"{time_str}: {interaction['action_type']} - {interaction['details'].get('summary', 'Action performed')}\n"
        
        return summary

class AnalysisConfigManager:
    """Manage technical and fundamental analysis configuration"""
    
    def __init__(self):
        self.init_default_configs()
    
    def init_default_configs(self):
        """Initialize default analysis configurations"""
        
        if 'technical_config' not in st.session_state:
            st.session_state.technical_config = {
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'sma_short': 20,
                'sma_long': 50,
                'volume_multiplier': 1.5,
                'confidence_threshold': 60,
                'signal_weights': {
                    'price_vs_sma': 0.3,
                    'rsi_position': 0.25,
                    'volume_analysis': 0.2,
                    'momentum': 0.25
                }
            }
        
        if 'fundamental_config' not in st.session_state:
            st.session_state.fundamental_config = {
                'pe_ratio_max': 25,
                'pe_ratio_min': 5,
                'roe_minimum': 0.15,
                'debt_to_equity_max': 2.0,
                'current_ratio_min': 1.0,
                'signal_weights': {
                    'valuation_metrics': 0.4,
                    'profitability': 0.3,
                    'financial_health': 0.3
                }
            }
    
    def update_technical_config(self, config: Dict):
        """Update technical analysis configuration"""
        st.session_state.technical_config.update(config)
    
    def update_fundamental_config(self, config: Dict):
        """Update fundamental analysis configuration"""
        st.session_state.fundamental_config.update(config)

class EnhancedChatbot:
    """Enhanced AI chatbot with full interaction tracking"""
    
    def __init__(self, tracker: InteractionTracker, technical_analysis, fundamental_analysis):
        self.tracker = tracker
        self.technical_analysis = technical_analysis
        self.fundamental_analysis = fundamental_analysis
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def process_message(self, message: str, user_tier: str = 'free') -> str:
        """Process user message with full context awareness"""
        
        # Log the chat interaction
        self.tracker.log_interaction('chat_message', {
            'message': message,
            'tier': user_tier,
            'summary': f"User asked: {message[:50]}..."
        })
        
        # Get interaction context
        context = self.tracker.get_context_summary()
        
        # Process based on user tier and message content
        if user_tier == 'free':
            return self._process_free_tier_message(message, context)
        else:
            return self._process_premium_message(message, context)
    
    def _process_free_tier_message(self, message: str, context: str) -> str:
        """Process message for free tier users"""
        
        message_lower = message.lower()
        
        # Stock analysis requests
        if any(ticker in message_lower for ticker in ['aapl', 'tsla', 'nvda', 'googl', 'msft', 'amzn']):
            ticker = self._extract_ticker(message)
            if ticker:
                return self._provide_stock_analysis(ticker, context)
        
        # Technical analysis questions
        if any(term in message_lower for term in ['technical', 'rsi', 'moving average', 'indicators']):
            return self._explain_technical_analysis(message, context)
        
        # Fundamental analysis questions  
        if any(term in message_lower for term in ['fundamental', 'pe ratio', 'earnings', 'revenue']):
            return self._explain_fundamental_analysis(message, context)
        
        # Configuration questions
        if any(term in message_lower for term in ['configure', 'settings', 'parameters']):
            return self._explain_configuration(context)
        
        # Context-aware general response
        return self._generate_contextual_response(message, context)
    
    def _process_premium_message(self, message: str, context: str) -> str:
        """Process message for premium tier users"""
        
        message_lower = message.lower()
        
        # Auto-recommendations
        if any(term in message_lower for term in ['recommend', 'suggest', 'best stocks', 'opportunities']):
            return self._provide_recommendations(context)
        
        # Advanced analysis requests
        if any(term in message_lower for term in ['deep analysis', 'comprehensive', 'detailed']):
            return self._provide_advanced_analysis(message, context)
        
        # Fall back to free tier processing for basic queries
        return self._process_free_tier_message(message, context)
    
    def _provide_stock_analysis(self, ticker: str, context: str) -> str:
        """Provide stock analysis based on current configuration"""
        
        try:
            # Get live stock data using data manager
            stock_data = data_manager.get_live_stock_data(ticker, period='3mo')
            quote = data_manager.get_real_time_quote(ticker)
            
            if stock_data is None or stock_data.empty:
                return f"Unable to fetch live data for {ticker}. Please check the ticker symbol or try again."
            
            # Apply technical analysis with current config
            config = st.session_state.technical_config
            current_price = quote.get('current_price', stock_data['Close'].iloc[-1])
            
            # Get data freshness info
            freshness = data_manager.get_data_freshness(ticker)
            
            # Calculate indicators
            sma_short = stock_data['Close'].rolling(config['sma_short']).mean().iloc[-1]
            sma_long = stock_data['Close'].rolling(config['sma_long']).mean().iloc[-1]
            rsi = self._calculate_rsi(stock_data['Close'], config['rsi_period']).iloc[-1]
            
            # Generate signal
            signal, confidence = self._generate_signal(current_price, sma_short, sma_long, rsi, config)
            
            # Log analysis interaction
            self.tracker.log_interaction('stock_analysis', {
                'ticker': ticker,
                'signal': signal,
                'confidence': confidence,
                'summary': f"Analyzed {ticker}: {signal} signal with {confidence}% confidence"
            })
            
            # Create response
            response = f"**Live Analysis for {ticker}**\n\n"
            response += f"📊 **Current Price**: ${current_price:.2f}\n"
            
            if 'change_percent' in quote:
                change_indicator = "🟢" if quote['change_percent'] > 0 else "🔴" if quote['change_percent'] < 0 else "⚪"
                response += f"{change_indicator} **Change**: {quote['change']:+.2f} ({quote['change_percent']:+.1f}%)\n"
            
            response += f"🎯 **Signal**: {signal}\n"
            response += f"📈 **Confidence**: {confidence}%\n"
            response += f"🕐 **Data Age**: {freshness.get('minutes_old', 0)} minutes\n\n"
            response += f"**Technical Indicators:**\n"
            response += f"- RSI ({config['rsi_period']} period): {rsi:.1f}\n"
            response += f"- SMA {config['sma_short']}: ${sma_short:.2f}\n"
            response += f"- SMA {config['sma_long']}: ${sma_long:.2f}\n\n"
            
            # Add reasoning based on context
            response += f"**Analysis Context:**\n"
            response += f"Based on your session activity, this analysis uses your configured parameters. "
            
            if confidence > 70:
                response += "High confidence signal suggests strong technical alignment."
            elif confidence > 50:
                response += "Moderate confidence indicates mixed signals."
            else:
                response += "Low confidence suggests waiting for clearer setup."
            
            return response
            
        except Exception as e:
            return f"Error analyzing {ticker}: {str(e)}"
    
    def _provide_recommendations(self, context: str) -> str:
        """Provide stock recommendations for premium users"""
        
        try:
            # Premium recommendation logic
            recommended_tickers = ['NVDA', 'AAPL', 'TSLA', 'GOOGL', 'MSFT']
            recommendations = []
            
            for ticker in recommended_tickers:
                try:
                    stock = yf.Ticker(ticker)
                    stock_data = stock.history(period='1mo')
                    
                    if not stock_data.empty:
                        current_price = stock_data['Close'].iloc[-1]
                        sma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
                        
                        # Quick signal generation
                        if current_price > sma_20 * 1.02:
                            signal = 'BUY'
                            confidence = 75
                        elif current_price < sma_20 * 0.98:
                            signal = 'SELL'
                            confidence = 70
                        else:
                            signal = 'HOLD'
                            confidence = 50
                        
                        recommendations.append({
                            'ticker': ticker,
                            'signal': signal,
                            'confidence': confidence,
                            'price': current_price
                        })
                except:
                    continue
            
            # Sort by confidence
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Log recommendation interaction
            self.tracker.log_interaction('recommendations_generated', {
                'count': len(recommendations),
                'top_pick': recommendations[0]['ticker'] if recommendations else 'None',
                'summary': f"Generated {len(recommendations)} recommendations"
            })
            
            # Create response
            response = "**Premium Stock Recommendations**\n\n"
            response += f"Based on your interaction history and current market analysis:\n\n"
            
            for i, rec in enumerate(recommendations[:5], 1):
                response += f"{i}. **{rec['ticker']}** - {rec['signal']} ({rec['confidence']}%)\n"
                response += f"   Price: ${rec['price']:.2f}\n\n"
            
            response += f"**Context from your session:**\n{context[-200:]}..."
            
            return response
            
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _generate_signal(self, price: float, sma_short: float, sma_long: float, rsi: float, config: Dict) -> tuple:
        """Generate trading signal based on configuration"""
        
        score = 0
        weights = config['signal_weights']
        
        # Price vs moving averages
        if price > sma_short > sma_long:
            score += weights['price_vs_sma'] * 100
        elif price < sma_short < sma_long:
            score -= weights['price_vs_sma'] * 100
        
        # RSI analysis
        if rsi < config['rsi_oversold']:
            score += weights['rsi_position'] * 80
        elif rsi > config['rsi_overbought']:
            score -= weights['rsi_position'] * 80
        
        # Normalize score to 0-100
        confidence = max(0, min(100, 50 + score))
        
        if confidence > 70:
            signal = 'BUY'
        elif confidence < 30:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return signal, int(confidence)
    
    def _extract_ticker(self, message: str) -> str:
        """Extract ticker symbol from message"""
        common_tickers = ['AAPL', 'TSLA', 'NVDA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX']
        message_upper = message.upper()
        
        for ticker in common_tickers:
            if ticker in message_upper:
                return ticker
        return None
    
    def _explain_technical_analysis(self, message: str, context: str) -> str:
        """Explain technical analysis concepts"""
        config = st.session_state.technical_config
        
        response = "**Technical Analysis Configuration**\n\n"
        response += f"Your current settings:\n"
        response += f"- RSI Period: {config['rsi_period']} days\n"
        response += f"- RSI Overbought: {config['rsi_overbought']}\n"
        response += f"- RSI Oversold: {config['rsi_oversold']}\n"
        response += f"- Short-term SMA: {config['sma_short']} days\n"
        response += f"- Long-term SMA: {config['sma_long']} days\n\n"
        
        response += "**How it works:**\n"
        response += "- RSI measures momentum (0-100 scale)\n"
        response += "- Moving averages show trend direction\n"
        response += "- Volume analysis confirms price movements\n"
        response += "- Combined signals generate buy/sell/hold recommendations\n\n"
        
        response += f"Based on your session: {context[-100:]}..."
        
        return response
    
    def _explain_fundamental_analysis(self, message: str, context: str) -> str:
        """Explain fundamental analysis concepts"""
        config = st.session_state.fundamental_config
        
        response = "**Fundamental Analysis Configuration**\n\n"
        response += f"Your current settings:\n"
        response += f"- Maximum P/E Ratio: {config['pe_ratio_max']}\n"
        response += f"- Minimum ROE: {config['roe_minimum']:.1%}\n"
        response += f"- Max Debt-to-Equity: {config['debt_to_equity_max']}\n\n"
        
        response += "**Key Metrics:**\n"
        response += "- P/E Ratio: Price relative to earnings\n"
        response += "- ROE: Return on equity efficiency\n"
        response += "- Debt-to-Equity: Financial leverage\n"
        response += "- Current Ratio: Short-term liquidity\n\n"
        
        response += f"Context: {context[-100:]}..."
        
        return response
    
    def _explain_configuration(self, context: str) -> str:
        """Explain configuration options"""
        
        response = "**Analysis Configuration**\n\n"
        response += "You can customize both technical and fundamental analysis parameters.\n\n"
        response += "**Technical Settings:**\n"
        response += "- Adjust RSI periods and thresholds\n"
        response += "- Modify moving average lengths\n"
        response += "- Set signal weight preferences\n\n"
        response += "**Fundamental Settings:**\n"
        response += "- Define valuation limits\n"
        response += "- Set profitability requirements\n"
        response += "- Configure financial health metrics\n\n"
        response += "Use the Configuration panel to adjust these settings."
        
        return response
    
    def _generate_contextual_response(self, message: str, context: str) -> str:
        """Generate contextual response based on interaction history"""
        
        response = "I'm tracking your session and ready to help with stock analysis.\n\n"
        
        if 'stock_analysis' in context:
            response += "I see you've been analyzing stocks. "
        if 'recommendations' in context:
            response += "You've requested recommendations. "
        if 'configuration' in context:
            response += "You've been adjusting settings. "
        
        response += "\n**What I can help with:**\n"
        response += "- Stock analysis (e.g., 'analyze AAPL')\n"
        response += "- Technical indicator explanations\n"
        response += "- Fundamental analysis guidance\n"
        response += "- Configuration assistance\n"
        
        if len(st.session_state.interaction_history) > 5:
            response += f"\n**Session Summary:**\nYou've made {len(st.session_state.interaction_history)} interactions. "
            response += "I'm learning your preferences to provide better assistance."
        
        return response

def create_configuration_panel():
    """Create configuration panel for analysis parameters"""
    
    st.sidebar.header("📊 Analysis Configuration")
    
    config_manager = AnalysisConfigManager()
    
    # Technical Analysis Configuration
    with st.sidebar.expander("⚙️ Technical Analysis"):
        st.subheader("Technical Parameters")
        
        rsi_period = st.slider("RSI Period", 10, 30, st.session_state.technical_config['rsi_period'])
        rsi_overbought = st.slider("RSI Overbought", 65, 80, st.session_state.technical_config['rsi_overbought'])
        rsi_oversold = st.slider("RSI Oversold", 20, 35, st.session_state.technical_config['rsi_oversold'])
        sma_short = st.slider("Short-term SMA", 10, 30, st.session_state.technical_config['sma_short'])
        sma_long = st.slider("Long-term SMA", 40, 100, st.session_state.technical_config['sma_long'])
        
        if st.button("Update Technical Config"):
            config_manager.update_technical_config({
                'rsi_period': rsi_period,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'sma_short': sma_short,
                'sma_long': sma_long
            })
            st.success("Technical configuration updated!")
    
    # Fundamental Analysis Configuration
    with st.sidebar.expander("💰 Fundamental Analysis"):
        st.subheader("Fundamental Parameters")
        
        pe_max = st.slider("Max P/E Ratio", 15, 40, st.session_state.fundamental_config['pe_ratio_max'])
        roe_min = st.slider("Min ROE (%)", 5, 25, int(st.session_state.fundamental_config['roe_minimum'] * 100))
        debt_max = st.slider("Max Debt-to-Equity", 0.5, 3.0, st.session_state.fundamental_config['debt_to_equity_max'])
        
        if st.button("Update Fundamental Config"):
            config_manager.update_fundamental_config({
                'pe_ratio_max': pe_max,
                'roe_minimum': roe_min / 100,
                'debt_to_equity_max': debt_max
            })
            st.success("Fundamental configuration updated!")

def create_enhanced_interface(db, data_retrieval, technical_analysis, fundamental_analysis, 
                            recommendation_engine, ml_engine, nl_interface, scheduler, user_tier):
    """Create enhanced full-screen interface with persistent AI chatbot"""
    
    # Initialize components
    tracker = InteractionTracker()
    chatbot = EnhancedChatbot(tracker, technical_analysis, fundamental_analysis)
    
    # Configuration panel
    create_configuration_panel()
    
    # User tier selection
    st.sidebar.header("🎯 Service Tier")
    tier_option = st.sidebar.selectbox("Select Tier", ["Free", "Premium"], 
                                      index=0 if user_tier == 'free' else 1)
    
    if tier_option != user_tier:
        user_tier = tier_option.lower()
        tracker.log_interaction('tier_change', {
            'new_tier': user_tier,
            'summary': f"Switched to {user_tier} tier"
        })
    
    # Main interface
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .chat-section {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        height: 80vh;
        overflow-y: auto;
    }
    .interaction-panel {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AI Stock Analysis Assistant</h1>
        <p>Tier: {user_tier.title()} | Session: {st.session_state.session_id}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout - 60% chat, 40% interactions
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="chat-section">', unsafe_allow_html=True)
        st.subheader("💬 AI Assistant Chat")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.markdown(f"""
                    <div style="background: #007bff; color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-left: 20%;">
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #ddd; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-right: 20%;">
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Ask anything about stocks or analysis...", 
                                  placeholder=f"e.g., 'analyze AAPL' or 'explain RSI' ({'premium recommendations available' if user_tier == 'premium' else 'free analysis available'})")
        
        col_send, col_clear = st.columns([3, 1])
        
        with col_send:
            if st.button("Send Message", use_container_width=True) and user_input:
                # Add user message
                st.session_state.chat_history.append(("user", user_input))
                
                # Get AI response
                ai_response = chatbot.process_message(user_input, user_tier)
                st.session_state.chat_history.append(("assistant", ai_response))
                
                st.rerun()
        
        with col_clear:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                tracker.log_interaction('chat_cleared', {'summary': 'Chat history cleared'})
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="interaction-panel">', unsafe_allow_html=True)
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        # Stock input
        ticker_input = st.text_input("Stock Ticker", placeholder="e.g., AAPL")
        
        if st.button("📊 Analyze Stock", use_container_width=True) and ticker_input:
            tracker.log_interaction('quick_analysis', {
                'ticker': ticker_input,
                'summary': f"Quick analysis requested for {ticker_input}"
            })
            
            # Add to chat
            message = f"analyze {ticker_input}"
            st.session_state.chat_history.append(("user", f"Quick analysis: {ticker_input}"))
            ai_response = chatbot.process_message(message, user_tier)
            st.session_state.chat_history.append(("assistant", ai_response))
            st.rerun()
        
        # Premium features
        if user_tier == 'premium':
            st.markdown("---")
            st.subheader("🚀 Premium Features")
            
            if st.button("🎯 Get Recommendations", use_container_width=True):
                tracker.log_interaction('premium_recommendations', {
                    'summary': 'Premium recommendations requested'
                })
                
                st.session_state.chat_history.append(("user", "Get me stock recommendations"))
                ai_response = chatbot.process_message("recommend stocks", user_tier)
                st.session_state.chat_history.append(("assistant", ai_response))
                st.rerun()
            
            if st.button("🔍 Advanced Analysis", use_container_width=True):
                tracker.log_interaction('advanced_analysis', {
                    'summary': 'Advanced analysis requested'
                })
                
                st.session_state.chat_history.append(("user", "Perform advanced market analysis"))
                ai_response = chatbot.process_message("deep analysis market", user_tier)
                st.session_state.chat_history.append(("assistant", ai_response))
                st.rerun()
        
        # Session Information
        st.markdown("---")
        st.subheader("📈 Session Info")
        
        st.metric("Interactions", len(st.session_state.interaction_history))
        st.metric("Chat Messages", len(st.session_state.chat_history))
        
        # Recent activity
        if st.session_state.interaction_history:
            st.subheader("🕒 Recent Activity")
            recent = st.session_state.interaction_history[-5:]
            for interaction in reversed(recent):
                time_str = datetime.fromisoformat(interaction['timestamp']).strftime("%H:%M")
                st.text(f"{time_str}: {interaction['details'].get('summary', 'Action')}")
        
        st.markdown('</div>', unsafe_allow_html=True)