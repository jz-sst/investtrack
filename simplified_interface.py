import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import yfinance as yf

# Configure Streamlit
st.set_page_config(
    page_title="AI Stock Analysis Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class StockAnalyzer:
    """Simplified stock analyzer with tiered features"""
    
    def __init__(self):
        self.user_tier = self._get_user_tier()
    
    def _get_user_tier(self):
        """Determine user tier - simplified for demo"""
        return st.session_state.get('user_tier', 'free')
    
    def get_trending_stocks(self):
        """Get trending stocks from various sources"""
        # Popular tech stocks and market movers
        trending_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'SPY', 'QQQ', 'AMD', 'CRM', 'UBER', 'SHOP', 'SNOW', 'PLTR'
        ]
        
        recommendations = []
        
        for ticker in trending_tickers[:8]:  # Limit to 8 stocks
            try:
                data, info = self.get_stock_data(ticker, period="1mo")
                if data is not None and len(data) > 10:  # Need at least 10 days for basic analysis
                    data = self.calculate_technical_indicators(data)
                    analysis = self.analyze_technical_signals(data, ticker)
                    
                    # Add fundamental score for premium users
                    fundamental_score = 0
                    if self.user_tier in ['premium', 'premium_plus']:
                        fund_analysis = self.get_fundamental_data(info)
                        if fund_analysis:
                            fundamental_score = fund_analysis['score']
                    
                    # Combined score
                    combined_score = analysis['score'] + (fundamental_score * 0.5)
                    
                    recommendations.append({
                        'ticker': ticker,
                        'signal': analysis['signal'],
                        'score': combined_score,
                        'price': analysis['current_price'],
                        'rsi': analysis.get('rsi'),
                        'company_name': info.get('longName', ticker) if info else ticker
                    })
                    
            except Exception as e:
                continue
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:5]
    
    def get_stock_data(self, ticker, period="3mo"):
        """Get stock data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            info = stock.info
            return data, info
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None, None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators with explanations"""
        if data is None or len(data) < 20:
            return None
        
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI Calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        return data
    
    def analyze_technical_signals(self, data, ticker):
        """Analyze technical signals with detailed explanations"""
        if data is None or len(data) < 10:
            return {"signal": "INSUFFICIENT_DATA", "explanation": "Not enough data for analysis", "score": 0, "current_price": 0, "rsi": None, "explanations": []}
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = []
        explanations = []
        score = 0
        
        # Price vs Moving Averages
        if latest['Close'] > latest['SMA_20']:
            signals.append("BULLISH: Price above 20-day MA")
            explanations.append(f"Current price ${latest['Close']:.2f} is above the 20-day moving average ${latest['SMA_20']:.2f}, indicating short-term upward momentum.")
            score += 1
        else:
            signals.append("BEARISH: Price below 20-day MA")
            explanations.append(f"Current price ${latest['Close']:.2f} is below the 20-day moving average ${latest['SMA_20']:.2f}, suggesting short-term weakness.")
            score -= 1
        
        if len(data) >= 50 and not pd.isna(latest['SMA_50']):
            if latest['Close'] > latest['SMA_50']:
                signals.append("BULLISH: Price above 50-day MA")
                explanations.append(f"Price ${latest['Close']:.2f} is above the 50-day moving average ${latest['SMA_50']:.2f}, showing medium-term strength.")
                score += 1
            else:
                signals.append("BEARISH: Price below 50-day MA")
                explanations.append(f"Price ${latest['Close']:.2f} is below the 50-day moving average ${latest['SMA_50']:.2f}, indicating medium-term weakness.")
                score -= 1
        
        # RSI Analysis
        if not pd.isna(latest['RSI']):
            if latest['RSI'] > 70:
                signals.append("OVERBOUGHT: RSI > 70")
                explanations.append(f"RSI is {latest['RSI']:.1f}, above 70, suggesting the stock may be overbought and due for a pullback.")
                score -= 0.5
            elif latest['RSI'] < 30:
                signals.append("OVERSOLD: RSI < 30")
                explanations.append(f"RSI is {latest['RSI']:.1f}, below 30, indicating the stock may be oversold and due for a bounce.")
                score += 0.5
            else:
                signals.append("NEUTRAL: RSI in normal range")
                explanations.append(f"RSI is {latest['RSI']:.1f}, in the normal range (30-70), showing balanced momentum.")
        
        # MACD Analysis
        if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                signals.append("BULLISH: MACD bullish crossover")
                explanations.append(f"MACD line ({latest['MACD']:.3f}) crossed above signal line ({latest['MACD_Signal']:.3f}), generating a buy signal.")
                score += 1
            elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                signals.append("BEARISH: MACD bearish crossover")
                explanations.append(f"MACD line ({latest['MACD']:.3f}) crossed below signal line ({latest['MACD_Signal']:.3f}), generating a sell signal.")
                score -= 1
        
        # Bollinger Bands
        if not pd.isna(latest['BB_Upper']) and not pd.isna(latest['BB_Lower']):
            if latest['Close'] > latest['BB_Upper']:
                signals.append("OVERBOUGHT: Price above upper Bollinger Band")
                explanations.append(f"Price ${latest['Close']:.2f} is above the upper Bollinger Band ${latest['BB_Upper']:.2f}, suggesting potential overbought conditions.")
                score -= 0.5
            elif latest['Close'] < latest['BB_Lower']:
                signals.append("OVERSOLD: Price below lower Bollinger Band")
                explanations.append(f"Price ${latest['Close']:.2f} is below the lower Bollinger Band ${latest['BB_Lower']:.2f}, indicating potential oversold conditions.")
                score += 0.5
        
        # Determine overall signal
        if score >= 2:
            overall_signal = "STRONG BUY"
        elif score >= 1:
            overall_signal = "BUY"
        elif score >= -1:
            overall_signal = "HOLD"
        elif score >= -2:
            overall_signal = "SELL"
        else:
            overall_signal = "STRONG SELL"
        
        return {
            "signal": overall_signal,
            "score": score,
            "signals": signals,
            "explanations": explanations,
            "current_price": latest['Close'],
            "rsi": latest['RSI'] if not pd.isna(latest['RSI']) else None,
            "macd": latest['MACD'] if not pd.isna(latest['MACD']) else None
        }
    
    def get_fundamental_data(self, info):
        """Get fundamental analysis data for premium users"""
        if not info:
            return None
        
        fundamentals = {}
        explanations = []
        score = 0
        
        # P/E Ratio
        pe_ratio = info.get('trailingPE')
        if pe_ratio:
            fundamentals['PE_Ratio'] = pe_ratio
            if pe_ratio < 15:
                explanations.append(f"P/E ratio of {pe_ratio:.1f} is relatively low, suggesting the stock may be undervalued.")
                score += 1
            elif pe_ratio > 25:
                explanations.append(f"P/E ratio of {pe_ratio:.1f} is high, which could indicate overvaluation or high growth expectations.")
                score -= 0.5
            else:
                explanations.append(f"P/E ratio of {pe_ratio:.1f} is in a reasonable range.")
        
        # Debt to Equity
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity:
            fundamentals['Debt_to_Equity'] = debt_to_equity
            if debt_to_equity < 30:
                explanations.append(f"Debt-to-equity ratio of {debt_to_equity:.1f}% is low, indicating strong financial stability.")
                score += 0.5
            elif debt_to_equity > 60:
                explanations.append(f"Debt-to-equity ratio of {debt_to_equity:.1f}% is high, suggesting higher financial risk.")
                score -= 0.5
        
        # ROE
        roe = info.get('returnOnEquity')
        if roe:
            fundamentals['ROE'] = roe * 100
            if roe > 0.15:
                explanations.append(f"Return on Equity of {roe*100:.1f}% is strong, showing efficient use of shareholder funds.")
                score += 1
            elif roe < 0.08:
                explanations.append(f"Return on Equity of {roe*100:.1f}% is low, indicating potential inefficiency.")
                score -= 0.5
        
        # Revenue Growth
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth:
            fundamentals['Revenue_Growth'] = revenue_growth * 100
            if revenue_growth > 0.1:
                explanations.append(f"Revenue growth of {revenue_growth*100:.1f}% shows strong business expansion.")
                score += 1
            elif revenue_growth < 0:
                explanations.append(f"Revenue declined by {abs(revenue_growth)*100:.1f}%, which is concerning.")
                score -= 1
        
        return {
            'fundamentals': fundamentals,
            'explanations': explanations,
            'score': score
        }
    
    def create_grok_chatbot(self):
        """Create Grok-powered chatbot for Premium Plus users"""
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        st.markdown("### 🤖 Grok AI Assistant")
        st.markdown("Ask questions about stocks, market analysis, or investment strategies.")
        
        # Chat display
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Grok:** {message['content']}")
        
        # Chat input
        user_input = st.text_input("Ask Grok about stocks or markets:", key="chat_input")
        
        if st.button("Send") and user_input:
            # Add user message
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            # Get Grok response
            try:
                from openai import OpenAI
                
                client = OpenAI(
                    base_url="https://api.x.ai/v1",
                    api_key=os.getenv('XAI_API_KEY')
                )
                
                system_prompt = """You are Grok, an expert financial AI assistant. You provide:
                - Stock analysis and investment advice
                - Market insights and trend analysis
                - Educational content about trading and investing
                - Real-time market commentary
                
                Be conversational, insightful, and helpful. Provide specific, actionable advice."""
                
                response = client.chat.completions.create(
                    model="grok-2-1212",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
                
            except Exception as e:
                error_msg = "Grok is currently unavailable. Please check your API key or try again later."
                st.session_state.chat_history.append({'role': 'assistant', 'content': error_msg})
            
            st.rerun()
    
    def create_technical_chart(self, data, ticker):
        """Create technical analysis chart"""
        if data is None or len(data) < 20:
            return None
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f'{ticker} Price & Moving Averages', 'RSI', 'MACD'),
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price and Moving Averages
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_20'],
            name='20-day MA', line=dict(color='orange', width=2)
        ), row=1, col=1)
        
        if 'SMA_50' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA_50'],
                name='50-day MA', line=dict(color='red', width=2)
            ), row=1, col=1)
        
        # Bollinger Bands
        if 'BB_Upper' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Upper'],
                name='BB Upper', line=dict(color='gray', dash='dash')
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Lower'],
                name='BB Lower', line=dict(color='gray', dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
            ), row=1, col=1)
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['RSI'],
                name='RSI', line=dict(color='purple')
            ), row=2, col=1)
            # RSI reference lines (simplified for Plotly compatibility)
            fig.add_hline(y=70, line_dash="dash", line_color="red")
            fig.add_hline(y=30, line_dash="dash", line_color="green")
        
        # MACD
        if 'MACD' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MACD'],
                name='MACD', line=dict(color='blue')
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MACD_Signal'],
                name='Signal', line=dict(color='red')
            ), row=3, col=1)
        
        fig.update_layout(
            title=f'{ticker} Technical Analysis',
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig

def create_main_interface():
    """Create the main analysis interface"""
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    # Initialize session state
    if 'show_detailed_analysis' not in st.session_state:
        st.session_state.show_detailed_analysis = False
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header { 
        text-align: center; 
        color: #1f77b4; 
        margin-bottom: 2rem;
    }
    .tier-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .free-tier { background: linear-gradient(45deg, #28a745, #20c997); color: white; }
    .premium-tier { background: linear-gradient(45deg, #ffc107, #fd7e14); color: black; }
    .analysis-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .signal-buy { color: #28a745; font-weight: bold; font-size: 1.2em; }
    .signal-sell { color: #dc3545; font-weight: bold; font-size: 1.2em; }
    .signal-hold { color: #ffc107; font-weight: bold; font-size: 1.2em; }
    .explanation-box {
        background: #fff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #17a2b8;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📈 AI Stock Analysis Bot</h1>', unsafe_allow_html=True)
    
    # Tier selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tier = st.selectbox(
            "Select your subscription tier:",
            ["free", "premium", "premium_plus"],
            format_func=lambda x: {
                "free": "Free Tier (Technical Analysis + Auto Recommendations)",
                "premium": "Premium Tier (Technical + Fundamental + Auto Recommendations)", 
                "premium_plus": "Premium Plus (Full Analysis + AI Chatbot + Auto Recommendations)"
            }[x]
        )
        st.session_state.user_tier = tier
    
    # Tier indicator
    if tier == "free":
        st.markdown('<div class="tier-indicator free-tier">🆓 FREE TIER - Technical Analysis + Auto Recommendations</div>', unsafe_allow_html=True)
    elif tier == "premium":
        st.markdown('<div class="tier-indicator premium-tier">⭐ PREMIUM TIER - Full Analysis + Auto Recommendations</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="tier-indicator premium-tier">💎 PREMIUM PLUS - Full Analysis + AI Chatbot + Auto Recommendations</div>', unsafe_allow_html=True)
    
    # Auto Recommendations Section
    st.markdown("## 🎯 Today's AI-Generated Recommendations")
    
    with st.spinner("Analyzing trending stocks..."):
        recommendations = analyzer.get_trending_stocks()
    

    
    if recommendations:
        cols = st.columns(5)
        for i, rec in enumerate(recommendations):
            with cols[i]:
                signal_color = {"STRONG BUY": "🟢", "BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "STRONG SELL": "🔴"}.get(rec['signal'], "⚪")
                st.markdown(f"""
                <div class="analysis-card" style="padding: 1rem; margin: 0.5rem 0;">
                    <h4>{rec['ticker']}</h4>
                    <p><strong>{rec['company_name'][:20]}...</strong></p>
                    <p>${rec['price']:.2f}</p>
                    <p>{signal_color} {rec['signal']}</p>
                    <p>Score: {rec['score']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No trending stocks found at the moment. Market may be closed or data temporarily unavailable.")
    
    st.markdown("---")
    
    # Advanced Features Section
    if tier == "premium_plus":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("## 🤖 Multi-Agent Analysis System")
            
            # Multi-Agent Analysis Button
            if st.button("🧠 Run Multi-Agent Analysis", type="secondary", use_container_width=True):
                from multi_agent_system import MultiAgentSystem
                
                with st.spinner("Running multi-agent analysis on trending stocks..."):
                    mas = MultiAgentSystem()
                    trending_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
                    
                    for ticker in trending_tickers[:3]:  # Analyze top 3
                        st.markdown(f"### 🔍 Multi-Agent Analysis: {ticker}")
                        
                        analysis = mas.analyze_stock(ticker)
                        
                        if "error" not in analysis:
                            # Display recommendation
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Recommendation", analysis['recommendation'])
                            with col_b:
                                st.metric("Confidence", f"{analysis['confidence']:.0f}%")
                            with col_c:
                                st.metric("Combined Score", f"{analysis['combined_score']:.2f}")
                            
                            # AI Summary
                            st.markdown("#### 🎯 Investment Thesis")
                            st.info(analysis['ai_summary'])
                            
                            # Agent Details
                            with st.expander("📊 Detailed Agent Analysis"):
                                col_tech, col_fund, col_sent = st.columns(3)
                                
                                with col_tech:
                                    st.markdown("**Technical Agent**")
                                    st.write(f"Score: {analysis['technical']['score']:.1f}")
                                    st.write(analysis['technical']['analysis'])
                                
                                with col_fund:
                                    st.markdown("**Fundamental Agent**")
                                    st.write(f"Score: {analysis['fundamental']['score']:.1f}")
                                    st.write(analysis['fundamental']['analysis'])
                                
                                with col_sent:
                                    st.markdown("**Sentiment Agent**")
                                    st.write(f"Score: {analysis['sentiment']['score']:.1f}")
                                    st.write(analysis['sentiment']['analysis'])
                            
                            st.markdown("---")
                        else:
                            st.error(f"Could not analyze {ticker}: {analysis['error']}")
        
        with col2:
            st.markdown("### 🤖 Grok AI Assistant")
            analyzer.create_grok_chatbot()
    else:
        st.markdown("## 🤖 Multi-Agent Analysis System")
        st.info("💎 **Upgrade to Premium Plus** to unlock the advanced multi-agent analysis system with specialized AI agents for technical, fundamental, and sentiment analysis!")
    
    # Market Trend Analysis Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔥 Get Market Trend Recommendations", type="primary", use_container_width=True):
            st.session_state.show_detailed_analysis = True
    
    # Show detailed analysis if button was clicked
    if st.session_state.get('show_detailed_analysis', False):
        with st.spinner("Analyzing current market trends and finding best opportunities..."):
            # Get top 3 recommendations for detailed analysis
            detailed_recommendations = analyzer.get_trending_stocks()[:3]
            
            for i, rec in enumerate(detailed_recommendations):
                st.markdown(f"## 📊 {i+1}. {rec['ticker']} - {rec['company_name']} Analysis")
                
                # Get detailed data for this stock
                data, info = analyzer.get_stock_data(rec['ticker'], period="3mo")
                
                if data is not None:
                    # Calculate technical indicators
                    data = analyzer.calculate_technical_indicators(data)
                    
                    # Create chart
                    chart = analyzer.create_technical_chart(data, rec['ticker'])
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Technical signals analysis
                    tech_analysis = analyzer.analyze_technical_signals(data, rec['ticker'])
                    
                    # Display technical analysis results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f'<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown("### Technical Signal")
                        signal_class = f"signal-{tech_analysis['signal'].lower().replace(' ', '-').replace('strong-', '')}"
                        st.markdown(f'<div class="{signal_class}">{tech_analysis["signal"]}</div>', unsafe_allow_html=True)
                        st.markdown(f"**Score:** {tech_analysis['score']:.1f}")
                        st.markdown(f"**Current Price:** ${tech_analysis['current_price']:.2f}")
                        if tech_analysis['rsi']:
                            st.markdown(f"**RSI:** {tech_analysis['rsi']:.1f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### How We Reached This Conclusion")
                        for j, explanation in enumerate(tech_analysis['explanations']):
                            st.markdown(f'<div class="explanation-box">', unsafe_allow_html=True)
                            st.markdown(f"**{j+1}.** {explanation}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Fundamental Analysis (Premium and Premium Plus)
                    if tier in ["premium", "premium_plus"]:
                        st.markdown("### 💰 Fundamental Analysis")
                        
                        fund_analysis = analyzer.get_fundamental_data(info)
                        
                        if fund_analysis and fund_analysis['fundamentals']:
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.markdown(f'<div class="analysis-card">', unsafe_allow_html=True)
                                st.markdown("#### Fundamental Metrics")
                                for key, value in fund_analysis['fundamentals'].items():
                                    if 'Growth' in key or 'ROE' in key:
                                        st.markdown(f"**{key.replace('_', ' ')}:** {value:.1f}%")
                                    else:
                                        st.markdown(f"**{key.replace('_', ' ')}:** {value:.1f}")
                                st.markdown(f"**Fundamental Score:** {fund_analysis['score']:.1f}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("#### Fundamental Analysis Explanation")
                                for j, explanation in enumerate(fund_analysis['explanations']):
                                    st.markdown(f'<div class="explanation-box">', unsafe_allow_html=True)
                                    st.markdown(f"**{j+1}.** {explanation}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("Limited fundamental data available for this stock.")
                    
                    st.markdown("---")
                
                else:
                    st.error(f"Could not fetch detailed data for {rec['ticker']}.")
        
        # Reset button to analyze again
        if st.button("🔄 Analyze New Market Trends", use_container_width=True):
            st.session_state.show_detailed_analysis = False
            st.rerun()

if __name__ == "__main__":
    create_main_interface()