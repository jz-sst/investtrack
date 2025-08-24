
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from typing import Dict, List, Any

# Import existing modules
from data_retrieval import DataRetrieval
from technical_analysis import TechnicalAnalysis
from fundamental_analysis import FundamentalAnalysis
from recommendation import RecommendationEngine
from ml_engine import MLEngine
from database import Database

# Set page config
st.set_page_config(
    page_title="InvestTrack Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InvestTrackPro:
    def __init__(self):
        self.init_session_state()
        self.init_components()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Dashboard'
        if 'user_portfolio' not in st.session_state:
            st.session_state.user_portfolio = self.get_sample_portfolio()
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ['AAPL', 'NVDA', 'MSFT']
        if 'alerts' not in st.session_state:
            st.session_state.alerts = self.get_sample_alerts()
    
    def init_components(self):
        """Initialize analysis components"""
        self.db = Database()
        self.data_retrieval = DataRetrieval(self.db)
        self.technical_analysis = TechnicalAnalysis()
        self.fundamental_analysis = FundamentalAnalysis()
        self.ml_engine = MLEngine()
        self.recommendation_engine = RecommendationEngine(
            self.technical_analysis, 
            self.fundamental_analysis
        )
    
    def get_sample_portfolio(self):
        """Sample portfolio data"""
        return {
            'AAPL': {'qty': 220, 'avg_cost': 165.00, 'current_price': 198.44},
            'NVDA': {'qty': 60, 'avg_cost': 720.00, 'current_price': 890.20},
            'MSFT': {'qty': 140, 'avg_cost': 350.00, 'current_price': 421.10},
            'XLE': {'qty': 180, 'avg_cost': 86.00, 'current_price': 85.210}
        }
    
    def get_sample_alerts(self):
        """Sample alerts data"""
        return [
            {
                'alert': 'ROI below 5%',
                'project': 'Logistics Hub',
                'triggered': '2h ago',
                'severity': 'High',
                'status': 'Open'
            },
            {
                'alert': 'IRR under 10%',
                'project': 'EV Charging Lot 4',
                'triggered': '1d ago',
                'severity': 'Medium',
                'status': 'Acknowledged'
            },
            {
                'alert': 'Cashflow variance > 15%',
                'project': 'Solar Farm A',
                'triggered': '3d ago',
                'severity': 'Critical',
                'status': 'Resolved'
            }
        ]
    
    def get_custom_css(self):
        """Custom CSS for InvestTrack Pro theme"""
        return """
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Root variables */
        :root {
            --primary-bg: #0f1419;
            --secondary-bg: #1a1f2e;
            --card-bg: #243447;
            --accent-color: #00d4aa;
            --text-primary: #ffffff;
            --text-secondary: #8892b0;
            --border-color: #2d3748;
            --success-color: #48bb78;
            --warning-color: #ed8936;
            --error-color: #f56565;
        }
        
        /* Global styles */
        .stApp {
            background-color: var(--primary-bg);
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--secondary-bg);
            border-right: 1px solid var(--border-color);
        }
        
        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }
        
        /* Custom card styling */
        .invest-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .invest-card-header {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        /* Metrics styling */
        .metric-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }
        
        .metric-label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
        }
        
        .metric-change {
            font-size: 14px;
            font-weight: 500;
            margin-top: 4px;
        }
        
        .metric-positive { color: var(--success-color); }
        .metric-negative { color: var(--error-color); }
        
        /* Navigation */
        .nav-item {
            padding: 12px 16px;
            margin: 4px 0;
            border-radius: 8px;
            cursor: pointer;
            color: var(--text-secondary);
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .nav-item:hover {
            background-color: var(--card-bg);
            color: var(--text-primary);
        }
        
        .nav-item.active {
            background-color: var(--accent-color);
            color: var(--primary-bg);
        }
        
        /* Buttons */
        .accent-button {
            background: var(--accent-color);
            color: var(--primary-bg);
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
        }
        
        /* Tables */
        .invest-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }
        
        .invest-table th {
            background: var(--secondary-bg);
            color: var(--text-secondary);
            padding: 12px;
            text-align: left;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .invest-table td {
            padding: 12px;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-color);
        }
        
        /* Status badges */
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-open { background: rgba(245, 101, 101, 0.1); color: var(--error-color); }
        .status-acknowledged { background: rgba(237, 137, 54, 0.1); color: var(--warning-color); }
        .status-resolved { background: rgba(72, 187, 120, 0.1); color: var(--success-color); }
        
        /* Charts */
        .plotly-graph-div {
            background: transparent !important;
        }
        </style>
        """
    
    def render_sidebar(self):
        """Render the sidebar navigation"""
        with st.sidebar:
            st.markdown("""
            <div style="padding: 20px 0; border-bottom: 1px solid var(--border-color); margin-bottom: 20px;">
                <h2 style="color: var(--text-primary); margin: 0; font-weight: 700;">InvestTrack Pro</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation menu
            pages = [
                ("Dashboard", "🏠"),
                ("Portfolio", "💼"),
                ("Stock Analysis", "📊"),
                ("Market Scanner", "🔍"),
                ("Opportunities", "💡"),
                ("Projects", "📁"),
                ("Alerts", "🚨"),
                ("Settings", "⚙️")
            ]
            
            for page, icon in pages:
                active_class = "active" if st.session_state.current_page == page else ""
                if st.button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
            
            # User profile at bottom
            st.markdown("""
            <div style="position: fixed; bottom: 20px; padding: 12px; background: var(--card-bg); border-radius: 8px; border: 1px solid var(--border-color);">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="width: 32px; height: 32px; background: var(--accent-color); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: var(--primary-bg); font-weight: 600;">AM</div>
                    <div>
                        <div style="color: var(--text-primary); font-weight: 500; font-size: 14px;">Alex Morgan</div>
                        <div style="color: var(--text-secondary); font-size: 12px;">Premium</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.markdown('<div class="invest-card-header">Dashboard</div>', unsafe_allow_html=True)
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate portfolio metrics
        total_value = sum(holding['qty'] * holding['current_price'] for holding in st.session_state.user_portfolio.values())
        total_cost = sum(holding['qty'] * holding['avg_cost'] for holding in st.session_state.user_portfolio.values())
        unrealized_pnl = total_value - total_cost
        daily_pnl = 4320  # Sample data
        ytd_return = 12.4  # Sample data
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Portfolio Value</div>
                <div class="metric-value">${total_value:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily P/L</div>
                <div class="metric-value metric-positive">+${daily_pnl:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">YTD Return</div>
                <div class="metric-value metric-positive">+{ytd_return}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">Moderate</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Main content row
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Performance chart
            st.markdown('<div class="invest-card">', unsafe_allow_html=True)
            st.markdown('<div class="invest-card-header">Performance Overview</div>', unsafe_allow_html=True)
            
            # Generate sample performance data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            performance = np.cumsum(np.random.normal(0.001, 0.02, len(dates))) + 1
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=performance * 100,
                mode='lines',
                name='Portfolio',
                line=dict(color='#00d4aa', width=2)
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(gridcolor='#2d3748'),
                yaxis=dict(gridcolor='#2d3748')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent news
            st.markdown('<div class="invest-card">', unsafe_allow_html=True)
            st.markdown('<div class="invest-card-header">Recent News</div>', unsafe_allow_html=True)
            
            news_items = [
                {"title": "S&P 500 rises as tech leads gains, investors eye Powell speech", "source": "Reuters", "time": "5m ago"},
                {"title": "NASDAQ posts third straight day of gains on chip rally", "source": "Bloomberg", "time": "12m ago"},
                {"title": "Energy stocks slip as oil retreats from weekly highs", "source": "WSJ", "time": "23m ago"}
            ]
            
            for item in news_items:
                st.markdown(f"""
                <div style="padding: 12px 0; border-bottom: 1px solid var(--border-color);">
                    <div style="color: var(--text-primary); font-weight: 500; margin-bottom: 4px;">{item['title']}</div>
                    <div style="color: var(--text-secondary); font-size: 12px;">{item['source']} • {item['time']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            # Key indices
            st.markdown('<div class="invest-card">', unsafe_allow_html=True)
            st.markdown('<div class="invest-card-header">Key Indices</div>', unsafe_allow_html=True)
            
            indices = [
                {"name": "S&P 500", "value": "5,235.12", "change": "+0.8%"},
                {"name": "NASDAQ", "value": "16,120.45", "change": "+1.2%"},
                {"name": "Dow Jones", "value": "39,210.33", "change": "+0.6%"}
            ]
            
            for idx in indices:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 8px 0;">
                    <span style="color: var(--text-primary);">{idx['name']}</span>
                    <div>
                        <span style="color: var(--text-primary);">{idx['value']}</span>
                        <span style="color: var(--success-color); margin-left: 8px;">{idx['change']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Alerts
            st.markdown('<div class="invest-card">', unsafe_allow_html=True)
            st.markdown('<div class="invest-card-header">Alerts</div>', unsafe_allow_html=True)
            
            alert_items = [
                {"text": "AAPL price alert", "condition": "> $200", "status": "Armed"},
                {"text": "Portfolio drop", "condition": "< -5% daily", "status": "Armed"},
                {"text": "NVDA RSI", "condition": "RSI < 30", "status": "Paused"}
            ]
            
            for alert in alert_items:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0;">
                    <div>
                        <div style="color: var(--text-primary); font-size: 14px;">{alert['text']}</div>
                        <div style="color: var(--text-secondary); font-size: 12px;">{alert['condition']}</div>
                    </div>
                    <span class="status-badge status-open">{alert['status']}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Watchlist
            self.render_watchlist_widget()
    
    def render_watchlist_widget(self):
        """Render watchlist widget"""
        st.markdown('<div class="invest-card">', unsafe_allow_html=True)
        st.markdown('<div class="invest-card-header">Watchlist</div>', unsafe_allow_html=True)
        
        # Sample watchlist data
        watchlist_data = [
            {"ticker": "AAPL", "price": "$198.44", "change": "+2.1%", "signal": "Buy"},
            {"ticker": "NVDA", "price": "$1,012.33", "change": "+0.8%", "signal": "Hold"},
            {"ticker": "MSFT", "price": "$414.22", "change": "+1.5%", "signal": "Buy"}
        ]
        
        for item in watchlist_data:
            change_color = "var(--success-color)" if "+" in item['change'] else "var(--error-color)"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border-color);">
                <div>
                    <span style="color: var(--text-primary); font-weight: 500;">{item['ticker']}</span>
                    <span style="color: var(--text-secondary); margin-left: 12px;">{item['price']}</span>
                </div>
                <div style="text-align: right;">
                    <div style="color: {change_color}; font-size: 12px;">{item['change']}</div>
                    <div style="color: var(--accent-color); font-size: 12px;">{item['signal']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_portfolio(self):
        """Render portfolio management page"""
        st.markdown('<div class="invest-card-header">Portfolio</div>', unsafe_allow_html=True)
        
        # Portfolio summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_value = sum(holding['qty'] * holding['current_price'] for holding in st.session_state.user_portfolio.values())
        total_cost = sum(holding['qty'] * holding['avg_cost'] for holding in st.session_state.user_portfolio.values())
        unrealized_pnl = total_value - total_cost
        cash = 12500  # Sample cash value
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Value</div>
                <div class="metric-value">${total_value:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pnl_color = "metric-positive" if unrealized_pnl > 0 else "metric-negative"
            pnl_sign = "+" if unrealized_pnl > 0 else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Unrealized P/L</div>
                <div class="metric-value {pnl_color}">{pnl_sign}${unrealized_pnl:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cash</div>
                <div class="metric-value">${cash:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">Moderate</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Holdings table
        st.markdown('<div class="invest-card">', unsafe_allow_html=True)
        st.markdown('<div class="invest-card-header">Holdings</div>', unsafe_allow_html=True)
        
        # Create holdings DataFrame
        holdings_data = []
        for ticker, data in st.session_state.user_portfolio.items():
            market_value = data['qty'] * data['current_price']
            pnl = market_value - (data['qty'] * data['avg_cost'])
            pnl_pct = (pnl / (data['qty'] * data['avg_cost'])) * 100
            
            holdings_data.append({
                'Asset': ticker,
                'Qty': data['qty'],
                'Avg Cost': f"${data['avg_cost']:.2f}",
                'Market Value': f"${market_value:,.0f}",
                'P/L': f"${pnl:,.0f}",
                'P/L %': f"{pnl_pct:+.1f}%"
            })
        
        holdings_df = pd.DataFrame(holdings_data)
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance chart and allocation
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown('<div class="invest-card">', unsafe_allow_html=True)
            st.markdown('<div class="invest-card-header">Performance</div>', unsafe_allow_html=True)
            
            # Sample performance data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
            portfolio_values = np.cumsum(np.random.normal(200, 1000, len(dates))) + total_value - 50000
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=portfolio_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00d4aa', width=2),
                fill='tonexty'
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(gridcolor='#2d3748'),
                yaxis=dict(gridcolor='#2d3748')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            st.markdown('<div class="invest-card">', unsafe_allow_html=True)
            st.markdown('<div class="invest-card-header">Allocation</div>', unsafe_allow_html=True)
            
            # Calculate allocation
            total_portfolio_value = sum(holding['qty'] * holding['current_price'] for holding in st.session_state.user_portfolio.values())
            allocations = {ticker: (data['qty'] * data['current_price'] / total_portfolio_value) * 100 
                         for ticker, data in st.session_state.user_portfolio.items()}
            
            for ticker, allocation in allocations.items():
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 8px 0;">
                    <span style="color: var(--text-primary);">{ticker}</span>
                    <span style="color: var(--text-secondary);">{allocation:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_alerts(self):
        """Render alerts page"""
        st.markdown('<div class="invest-card-header">Alerts Center</div>', unsafe_allow_html=True)
        
        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Open Alerts</div>
                <div class="metric-value">18</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Acknowledged</div>
                <div class="metric-value">12</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Resolved</div>
                <div class="metric-value">34</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Avg Time to Close</div>
                <div class="metric-value">1.8d</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Alerts table
        st.markdown('<div class="invest-card">', unsafe_allow_html=True)
        st.markdown('<div class="invest-card-header">Recent Alerts</div>', unsafe_allow_html=True)
        
        alerts_df = pd.DataFrame(st.session_state.alerts)
        
        # Display alerts with custom styling
        for _, alert in alerts_df.iterrows():
            status_class = f"status-{alert['status'].lower()}"
            severity_color = {
                'High': 'var(--error-color)',
                'Medium': 'var(--warning-color)', 
                'Critical': 'var(--error-color)'
            }.get(alert['severity'], 'var(--text-secondary)')
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid var(--border-color);">
                <div style="flex: 1;">
                    <div style="color: var(--text-primary); font-weight: 500;">{alert['alert']}</div>
                    <div style="color: var(--text-secondary); font-size: 12px;">{alert['project']} • {alert['triggered']}</div>
                </div>
                <div style="display: flex; gap: 12px; align-items: center;">
                    <span style="color: {severity_color}; font-size: 12px;">{alert['severity']}</span>
                    <span class="status-badge {status_class}">{alert['status']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_market_scanner(self):
        """Render market scanner page"""
        st.markdown('<div class="invest-card-header">Market Scanner</div>', unsafe_allow_html=True)
        
        col_filters, col_results = st.columns([1, 2])
        
        with col_filters:
            st.markdown('<div class="invest-card">', unsafe_allow_html=True)
            st.markdown('<div class="invest-card-header">Filters</div>', unsafe_allow_html=True)
            
            # Filter controls
            exchange = st.selectbox("Exchange", ["All", "NYSE", "NASDAQ"], key="scanner_exchange")
            sector = st.selectbox("Sector", ["Any", "Technology", "Healthcare", "Finance"], key="scanner_sector")
            market_cap = st.selectbox("Market Cap", ["> $10B", "$1B - $10B", "< $1B"], key="scanner_mcap")
            
            st.markdown("**Technical**")
            rsi_filter = st.checkbox("RSI < 30", key="rsi_filter")
            price_filter = st.checkbox("Price vs 200MA: Above", key="price_filter")
            macd_filter = st.checkbox("MACD: Bullish", key="macd_filter")
            
            st.markdown("**Fundamentals**")
            pe_filter = st.checkbox("P/E: 5 - 40", key="pe_filter")
            revenue_filter = st.checkbox("Revenue Growth: > 5%", key="revenue_filter")
            
            if st.button("Apply Filters", type="primary"):
                st.success("Filters applied! Scanning 5,000+ stocks...")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_results:
            st.markdown('<div class="invest-card">', unsafe_allow_html=True)
            st.markdown('<div class="invest-card-header">Results</div>', unsafe_allow_html=True)
            
            # Sample scanner results
            scanner_results = [
                {"Ticker": "AAPL", "Company": "Apple Inc.", "Price": "$198.42", "% Chg": "+1.2%", "RSI": 62, "Trend": "↗"},
                {"Ticker": "MSFT", "Company": "Microsoft", "Price": "$421.10", "% Chg": "+0.8%", "RSI": 58, "Trend": "↗"},
                {"Ticker": "NVDA", "Company": "NVIDIA", "Price": "$890.20", "% Chg": "+2.1%", "RSI": 65, "Trend": "↗"}
            ]
            
            results_df = pd.DataFrame(scanner_results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        # Apply custom CSS
        st.markdown(self.get_custom_css(), unsafe_allow_html=True)
        
        # Render sidebar
        self.render_sidebar()
        
        # Route to appropriate page
        if st.session_state.current_page == "Dashboard":
            self.render_dashboard()
        elif st.session_state.current_page == "Portfolio":
            self.render_portfolio()
        elif st.session_state.current_page == "Alerts":
            self.render_alerts()
        elif st.session_state.current_page == "Market Scanner":
            self.render_market_scanner()
        elif st.session_state.current_page == "Stock Analysis":
            st.markdown("## 📊 Stock Analysis")
            st.info("Stock Analysis page - Enhanced charting and technical analysis coming soon!")
        elif st.session_state.current_page == "Opportunities":
            st.markdown("## 💡 Investment Opportunities")
            st.info("Investment Opportunities page - AI-powered opportunity discovery coming soon!")
        elif st.session_state.current_page == "Projects":
            st.markdown("## 📁 Capital Projects")
            st.info("Capital Projects page - Project management and ROI tracking coming soon!")
        elif st.session_state.current_page == "Settings":
            st.markdown("## ⚙️ Settings")
            st.info("Settings page - Profile and preferences management coming soon!")

if __name__ == "__main__":
    app = InvestTrackPro()
    app.run()
