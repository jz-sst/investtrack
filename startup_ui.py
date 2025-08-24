"""
Modern Startup-Style UI for AI Stock Analysis Bot
Trendy colors, modern fonts, and clean design for younger generation
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

# Modern startup color palette
COLORS = {
    'primary': '#6366f1',      # Indigo
    'secondary': '#8b5cf6',    # Violet  
    'accent': '#06b6d4',       # Cyan
    'success': '#10b981',      # Emerald
    'warning': '#f59e0b',      # Amber
    'error': '#ef4444',        # Red
    'dark': '#1f2937',         # Dark gray
    'light': '#f8fafc',        # Light gray
    'gradient1': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'gradient2': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    'gradient3': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'gradient4': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
}

def get_modern_css():
    """Get modern CSS styling for the app"""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {{
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }}
    
    .main-container {{
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }}
    
    .hero-section {{
        background: {COLORS['gradient1']};
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    }}
    
    .hero-title {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #fff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .hero-subtitle {{
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
    }}
    
    .card {{
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }}
    
    .card-header {{
        font-size: 1.25rem;
        font-weight: 600;
        color: {COLORS['dark']};
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .metric-card {{
        background: {COLORS['gradient3']};
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        font-size: 0.875rem;
        opacity: 0.9;
        font-weight: 500;
    }}
    
    .status-badge {{
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }}
    
    .status-buy {{
        background: {COLORS['success']};
        color: white;
    }}
    
    .status-hold {{
        background: {COLORS['warning']};
        color: white;
    }}
    
    .status-sell {{
        background: {COLORS['error']};
        color: white;
    }}
    
    .chat-container {{
        background: {COLORS['light']};
        border-radius: 16px;
        padding: 1.5rem;
        height: 500px;
        overflow-y: auto;
        border: 1px solid #e5e7eb;
    }}
    
    .chat-message {{
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 12px;
        max-width: 80%;
    }}
    
    .chat-user {{
        background: {COLORS['primary']};
        color: white;
        margin-left: auto;
    }}
    
    .chat-ai {{
        background: white;
        border: 1px solid #e5e7eb;
        color: {COLORS['dark']};
    }}
    
    .action-button {{
        background: {COLORS['gradient2']};
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    
    .action-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }}
    
    .technical-analysis-card {{
        background: {COLORS['gradient4']};
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    }}
    
    .analysis-result {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid {COLORS['primary']};
    }}
    
    .stock-ticker {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COLORS['primary']};
        margin-bottom: 0.5rem;
    }}
    
    .price-display {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['dark']};
        margin-bottom: 1rem;
    }}
    
    .indicator-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }}
    
    .indicator-item {{
        background: {COLORS['light']};
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }}
    
    .recommendation-panel {{
        background: {COLORS['gradient1']};
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }}
    
    .recommendation-action {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }}
    
    .confidence-bar {{
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }}
    
    .confidence-fill {{
        height: 100%;
        background: white;
        transition: width 0.3s ease;
    }}
    </style>
    """

def create_modern_interface(db, data_retrieval, technical_analysis, fundamental_analysis, 
                           recommendation_engine, ml_engine, nl_interface, scheduler, user_tier):
    """Create modern startup-style interface with premium features"""
    
    # Import premium features
    from premium_features import PremiumAnalysisEngine, PremiumChatInterface, create_premium_interface
    
    # Check if user wants premium features
    if st.sidebar.button("🚀 Premium Mode"):
        st.session_state.premium_mode = True
    
    if st.sidebar.button("📊 Standard Mode"):
        st.session_state.premium_mode = False
    
    # Initialize premium mode if not set
    if 'premium_mode' not in st.session_state:
        st.session_state.premium_mode = False
    
    # Show premium interface if enabled
    if st.session_state.premium_mode:
        premium_engine = PremiumAnalysisEngine(technical_analysis, fundamental_analysis, ml_engine, nl_interface)
        chat_interface = PremiumChatInterface(nl_interface, technical_analysis, fundamental_analysis)
        create_premium_interface(premium_engine, chat_interface)
        return
    
    st.markdown(get_modern_css(), unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🚀 AI Stock Analysis</div>
        <div class="hero-subtitle">Technical-first approach to smart investing</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Technical Analysis Section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📊 Technical Analysis Engine</div>', unsafe_allow_html=True)
        
        # Stock input
        ticker = st.text_input("Enter Stock Ticker", placeholder="e.g., AAPL, TSLA, NVDA", key="ticker_input")
        
        # Analysis buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("⚡ Quick Analysis", use_container_width=True):
                if ticker:
                    run_technical_analysis(ticker, technical_analysis, ml_engine, 'quick')
        
        with col_btn2:
            if st.button("🔍 Deep Analysis", use_container_width=True):
                if ticker:
                    run_technical_analysis(ticker, technical_analysis, ml_engine, 'deep')
        
        with col_btn3:
            if st.button("🌟 Discovery", use_container_width=True):
                run_market_discovery(technical_analysis, ml_engine)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display Analysis Results
        display_technical_results()
        
    with col2:
        # Market Status
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🌐 Market Status</div>', unsafe_allow_html=True)
        display_market_status_modern()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Stats
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📈 Today\'s Signals</div>', unsafe_allow_html=True)
        display_daily_signals()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fundamental Analysis Option
        if 'current_analysis' in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">🔬 Go Deeper?</div>', unsafe_allow_html=True)
            
            if st.button("🧮 Add Fundamental Analysis", use_container_width=True):
                run_fundamental_analysis(st.session_state.current_analysis['ticker'], 
                                       fundamental_analysis, recommendation_engine)
            
            st.markdown('</div>', unsafe_allow_html=True)

def run_technical_analysis(ticker, technical_analysis, ml_engine, analysis_type):
    """Run technical analysis on a stock"""
    
    try:
        with st.spinner(f"Analyzing {ticker}..."):
            # Get stock data
            stock = yf.Ticker(ticker)
            stock_data = stock.history(period='6mo' if analysis_type == 'deep' else '3mo')
            
            if stock_data.empty:
                st.error(f"No data found for {ticker}")
                return
            
            # Technical analysis
            ta_results = technical_analysis.analyze_stock(stock_data, ticker)
            
            # ML enhancement
            ml_score = ml_engine.predict_score(stock_data, ta_results.get('indicators', {}))
            
            # Generate recommendation
            current_price = stock_data['Close'].iloc[-1]
            sma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
            rsi = ta_results.get('indicators', {}).get('RSI', pd.Series([50])).iloc[-1]
            
            # Technical signal logic
            if current_price > sma_20 * 1.02 and rsi < 70:
                signal = 'BUY'
                confidence = 85
            elif current_price < sma_20 * 0.98 and rsi > 30:
                signal = 'SELL'
                confidence = 80
            else:
                signal = 'HOLD'
                confidence = 60
            
            # Store results
            st.session_state.current_analysis = {
                'ticker': ticker,
                'signal': signal,
                'confidence': confidence,
                'price': current_price,
                'ta_score': ta_results.get('score', 50),
                'ml_score': ml_score,
                'rsi': rsi,
                'sma_20': sma_20,
                'stock_data': stock_data
            }
            
            st.success(f"Technical analysis complete for {ticker}")
            
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {str(e)}")

def run_market_discovery(technical_analysis, ml_engine):
    """Run market discovery analysis"""
    
    try:
        with st.spinner("Discovering market opportunities..."):
            discovery_stocks = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX']
            results = []
            
            for ticker in discovery_stocks:
                try:
                    stock = yf.Ticker(ticker)
                    stock_data = stock.history(period='1mo')
                    
                    if not stock_data.empty:
                        current_price = stock_data['Close'].iloc[-1]
                        sma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
                        
                        # Quick technical signal
                        if current_price > sma_20 * 1.02:
                            signal = 'BUY'
                        elif current_price < sma_20 * 0.98:
                            signal = 'SELL'
                        else:
                            signal = 'HOLD'
                        
                        results.append({
                            'ticker': ticker,
                            'signal': signal,
                            'price': current_price,
                            'change': ((current_price - sma_20) / sma_20) * 100
                        })
                except:
                    continue
            
            st.session_state.discovery_results = results
            st.success("Market discovery complete!")
            
    except Exception as e:
        st.error(f"Error in market discovery: {str(e)}")

def display_technical_results():
    """Display technical analysis results"""
    
    if 'current_analysis' in st.session_state:
        analysis = st.session_state.current_analysis
        
        st.markdown(f"""
        <div class="analysis-result">
            <div class="stock-ticker">{analysis['ticker']}</div>
            <div class="price-display">${analysis['price']:.2f}</div>
            
            <div class="recommendation-panel">
                <div class="recommendation-action">{analysis['signal']}</div>
                <div>Technical Confidence: {analysis['confidence']}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {analysis['confidence']}%"></div>
                </div>
            </div>
            
            <div class="indicator-grid">
                <div class="indicator-item">
                    <div><strong>TA Score</strong></div>
                    <div>{analysis['ta_score']:.1f}/100</div>
                </div>
                <div class="indicator-item">
                    <div><strong>ML Score</strong></div>
                    <div>{analysis['ml_score']:.1f}/100</div>
                </div>
                <div class="indicator-item">
                    <div><strong>RSI</strong></div>
                    <div>{analysis['rsi']:.1f}</div>
                </div>
                <div class="indicator-item">
                    <div><strong>vs SMA20</strong></div>
                    <div>{((analysis['price'] - analysis['sma_20']) / analysis['sma_20'] * 100):+.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chart
        fig = go.Figure()
        
        stock_data = analysis['stock_data']
        
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            mode='lines',
            name='Price',
            line=dict(color=COLORS['primary'], width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'].rolling(20).mean(),
            mode='lines',
            name='SMA 20',
            line=dict(color=COLORS['accent'], width=2)
        ))
        
        fig.update_layout(
            title=f"{analysis['ticker']} Technical Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_market_status_modern():
    """Display modern market status"""
    
    try:
        spy = yf.Ticker('SPY')
        spy_data = spy.history(period='1d')
        
        if not spy_data.empty:
            current_price = spy_data['Close'].iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${current_price:.2f}</div>
                <div class="metric-label">S&P 500</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="status-badge status-buy">Markets Active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-hold">Markets Closed</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.markdown('<div class="status-badge status-sell">Connection Error</div>', unsafe_allow_html=True)

def display_daily_signals():
    """Display daily trading signals"""
    
    if 'discovery_results' in st.session_state:
        results = st.session_state.discovery_results
        
        buy_count = sum(1 for r in results if r['signal'] == 'BUY')
        hold_count = sum(1 for r in results if r['signal'] == 'HOLD')
        sell_count = sum(1 for r in results if r['signal'] == 'SELL')
        
        st.markdown(f"""
        <div class="indicator-grid">
            <div class="indicator-item">
                <div class="status-badge status-buy">{buy_count} BUY</div>
            </div>
            <div class="indicator-item">
                <div class="status-badge status-hold">{hold_count} HOLD</div>
            </div>
            <div class="indicator-item">
                <div class="status-badge status-sell">{sell_count} SELL</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show top signals
        st.markdown("**Top Signals:**")
        for result in results[:3]:
            st.markdown(f"- {result['ticker']}: {result['signal']} (${result['price']:.2f})")
    
    else:
        st.info("Run discovery to see signals")

def run_fundamental_analysis(ticker, fundamental_analysis, recommendation_engine):
    """Run fundamental analysis after technical analysis"""
    
    try:
        with st.spinner(f"Adding fundamental analysis for {ticker}..."):
            # Get fundamental data
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Basic fundamental metrics
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            roe = info.get('returnOnEquity', 0)
            
            # Simple fundamental score
            fa_score = 50  # Base score
            
            if pe_ratio > 0:
                if pe_ratio < 15:
                    fa_score += 10
                elif pe_ratio > 25:
                    fa_score -= 10
            
            if pb_ratio > 0:
                if pb_ratio < 1.5:
                    fa_score += 10
                elif pb_ratio > 3:
                    fa_score -= 10
            
            if roe > 0:
                if roe > 0.15:
                    fa_score += 15
                elif roe < 0.05:
                    fa_score -= 15
            
            # Update analysis with fundamental data
            if 'current_analysis' in st.session_state:
                st.session_state.current_analysis['fa_score'] = fa_score
                st.session_state.current_analysis['pe_ratio'] = pe_ratio
                st.session_state.current_analysis['pb_ratio'] = pb_ratio
                st.session_state.current_analysis['roe'] = roe
                
                # Combined recommendation
                ta_score = st.session_state.current_analysis['ta_score']
                combined_score = (ta_score * 0.7) + (fa_score * 0.3)
                
                st.session_state.current_analysis['combined_score'] = combined_score
                
                st.success("Fundamental analysis added!")
            
    except Exception as e:
        st.error(f"Error in fundamental analysis: {str(e)}")