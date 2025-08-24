"""
Modern UI components for the AI Stock Analysis Bot
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

def display_interactive_chat(nl_interface, db, data_retrieval, technical_analysis, 
                            fundamental_analysis, recommendation_engine, ml_engine, user_tier):
    """Interactive chat interface with action execution"""
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat display area
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 10px; border-radius: 10px; margin: 10px 0; text-align: right;">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #f1f8e9; padding: 10px; border-radius: 10px; margin: 10px 0;">
                    <strong>AI Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Ask me anything about stocks:", key="chat_input_main", placeholder="e.g., 'What is the update for today?' or 'Analyze AAPL'")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Send", type="primary", use_container_width=True):
            if user_input:
                process_chat_command(user_input, db, data_retrieval, technical_analysis, 
                                   fundamental_analysis, recommendation_engine, ml_engine)
    
    with col2:
        if st.button("Clear Chat", type="secondary", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

def process_chat_command(user_input, db, data_retrieval, technical_analysis, 
                        fundamental_analysis, recommendation_engine, ml_engine):
    """Process chat commands and execute actions"""
    
    # Add user message
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
    
    # Process the command
    user_input_lower = user_input.lower()
    
    if 'update for today' in user_input_lower or 'daily update' in user_input_lower:
        # Run daily discovery
        response = execute_daily_discovery(db, data_retrieval, technical_analysis, 
                                         fundamental_analysis, recommendation_engine, ml_engine)
        
    elif 'analyze' in user_input_lower:
        # Extract ticker if mentioned
        words = user_input.split()
        ticker = None
        for word in words:
            if word.upper() in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'AMD', 'NFLX']:
                ticker = word.upper()
                break
        
        if ticker:
            response = execute_stock_analysis(ticker, db, data_retrieval, technical_analysis, 
                                            fundamental_analysis, recommendation_engine, ml_engine)
        else:
            response = "Please specify a stock ticker to analyze (e.g., 'Analyze AAPL')"
    
    elif 'trending' in user_input_lower or 'hot stocks' in user_input_lower:
        response = execute_trending_analysis(db, data_retrieval, technical_analysis, 
                                           fundamental_analysis, recommendation_engine, ml_engine)
    
    elif 'market sentiment' in user_input_lower or 'market overview' in user_input_lower:
        response = execute_market_analysis(db, data_retrieval, technical_analysis, 
                                         fundamental_analysis, recommendation_engine, ml_engine)
    
    else:
        response = generate_smart_response(user_input)
    
    # Add AI response
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
    st.rerun()

def execute_daily_discovery(db, data_retrieval, technical_analysis, 
                           fundamental_analysis, recommendation_engine, ml_engine):
    """Execute daily stock discovery"""
    
    try:
        # Discovery categories
        discovery_stocks = {
            'Growth Stocks': ['NVDA', 'TSLA', 'SHOP', 'ROKU', 'SQ'],
            'Value Stocks': ['BRK.B', 'JPM', 'JNJ', 'PG', 'KO'],
            'Tech Stocks': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
            'Dividend Stocks': ['T', 'VZ', 'XOM', 'CVX', 'MO']
        }
        
        results = {}
        total_buy_signals = 0
        
        for category, tickers in discovery_stocks.items():
            category_results = []
            for ticker in tickers[:2]:  # Analyze top 2 per category
                try:
                    stock = yf.Ticker(ticker)
                    stock_data = stock.history(period='3mo')
                    
                    if not stock_data.empty:
                        # Quick analysis
                        ta_results = technical_analysis.analyze_stock(stock_data, ticker)
                        ml_score = ml_engine.predict_score(stock_data, ta_results.get('indicators', {}))
                        combined_score = (ta_results['score'] + ml_score) / 2
                        
                        recommendation = recommendation_engine.get_recommendation(ticker, ta_results, None)
                        
                        if recommendation['action'] == 'BUY':
                            total_buy_signals += 1
                        
                        category_results.append({
                            'ticker': ticker,
                            'score': combined_score,
                            'action': recommendation['action'],
                            'price': stock_data['Close'].iloc[-1]
                        })
                except:
                    continue
            
            results[category] = category_results
        
        # Store results for display
        st.session_state.daily_discovery = results
        
        # Generate response
        response = f"**Daily Market Discovery Update - {datetime.now().strftime('%Y-%m-%d')}**\n\n"
        response += f"🔍 **Discovery Summary:** Analyzed {sum(len(r) for r in results.values())} stocks across 4 categories\n"
        response += f"📈 **Buy Signals:** {total_buy_signals} stocks showing BUY recommendation\n\n"
        
        for category, stocks in results.items():
            if stocks:
                response += f"**{category}:**\n"
                for stock in stocks:
                    response += f"- {stock['ticker']}: {stock['action']} (Score: {stock['score']:.1f}, Price: ${stock['price']:.2f})\n"
                response += "\n"
        
        return response
        
    except Exception as e:
        return f"Error running daily discovery: {str(e)}"

def execute_stock_analysis(ticker, db, data_retrieval, technical_analysis, 
                          fundamental_analysis, recommendation_engine, ml_engine):
    """Execute analysis for a specific stock"""
    
    try:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period='6mo')
        
        if stock_data.empty:
            return f"No data available for {ticker}"
        
        # Technical analysis
        ta_results = technical_analysis.analyze_stock(stock_data, ticker)
        
        # ML enhancement
        ml_score = ml_engine.predict_score(stock_data, ta_results.get('indicators', {}))
        combined_score = (ta_results['score'] + ml_score) / 2
        
        # Get recommendation
        recommendation = recommendation_engine.get_recommendation(ticker, ta_results, None)
        
        # Store results
        st.session_state.analysis_results = {
            'ticker': ticker,
            'ta_results': ta_results,
            'recommendation': recommendation,
            'stock_data': stock_data,
            'current_price': stock_data['Close'].iloc[-1]
        }
        
        # Generate response
        response = f"**Analysis for {ticker}**\n\n"
        response += f"📊 **Current Price:** ${stock_data['Close'].iloc[-1]:.2f}\n"
        response += f"🎯 **Recommendation:** {recommendation['action']}\n"
        response += f"📈 **Technical Score:** {ta_results['score']:.1f}/100\n"
        response += f"🤖 **ML Score:** {ml_score:.1f}/100\n"
        response += f"⚖️ **Combined Score:** {combined_score:.1f}/100\n"
        response += f"🔒 **Confidence:** {recommendation['confidence']}\n\n"
        response += f"**Key Insights:** {recommendation['reasoning'][:200]}..."
        
        return response
        
    except Exception as e:
        return f"Error analyzing {ticker}: {str(e)}"

def execute_trending_analysis(db, data_retrieval, technical_analysis, 
                             fundamental_analysis, recommendation_engine, ml_engine):
    """Execute trending stocks analysis"""
    
    try:
        trending_stocks = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'MSFT']
        results = []
        
        for ticker in trending_stocks:
            try:
                stock = yf.Ticker(ticker)
                stock_data = stock.history(period='1mo')
                
                if not stock_data.empty:
                    current_price = stock_data['Close'].iloc[-1]
                    prev_price = stock_data['Close'].iloc[-2]
                    change = ((current_price - prev_price) / prev_price) * 100
                    
                    # Quick TA
                    ta_results = technical_analysis.analyze_stock(stock_data, ticker)
                    
                    results.append({
                        'ticker': ticker,
                        'price': current_price,
                        'change': change,
                        'score': ta_results['score']
                    })
            except:
                continue
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        response = "**Trending Stocks Analysis**\n\n"
        for i, stock in enumerate(results, 1):
            response += f"{i}. **{stock['ticker']}**: ${stock['price']:.2f} "
            response += f"({stock['change']:+.1f}%) - Score: {stock['score']:.1f}/100\n"
        
        return response
        
    except Exception as e:
        return f"Error analyzing trending stocks: {str(e)}"

def execute_market_analysis(db, data_retrieval, technical_analysis, 
                           fundamental_analysis, recommendation_engine, ml_engine):
    """Execute market sentiment analysis"""
    
    try:
        market_etfs = ['SPY', 'QQQ', 'IWM', 'DIA']
        results = []
        
        for etf in market_etfs:
            try:
                stock = yf.Ticker(etf)
                stock_data = stock.history(period='1mo')
                
                if not stock_data.empty:
                    current_price = stock_data['Close'].iloc[-1]
                    sma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
                    trend = "Bullish" if current_price > sma_20 else "Bearish"
                    
                    results.append({
                        'etf': etf,
                        'price': current_price,
                        'trend': trend
                    })
            except:
                continue
        
        bullish_count = sum(1 for r in results if r['trend'] == 'Bullish')
        sentiment = "Bullish" if bullish_count > len(results)/2 else "Bearish"
        
        response = f"**Market Sentiment Analysis**\n\n"
        response += f"📊 **Overall Sentiment:** {sentiment}\n\n"
        
        for result in results:
            response += f"- **{result['etf']}**: ${result['price']:.2f} - {result['trend']}\n"
        
        return response
        
    except Exception as e:
        return f"Error analyzing market sentiment: {str(e)}"

def generate_smart_response(user_input):
    """Generate smart response for general queries"""
    
    responses = {
        'help': "I can help you with:\n- Daily market updates\n- Stock analysis\n- Trending stocks\n- Market sentiment\n- Technical indicators\n\nTry asking: 'What is the update for today?' or 'Analyze AAPL'",
        'features': "Available features:\n- Real-time stock analysis\n- Daily auto-discovery\n- ML-enhanced predictions\n- Technical indicators\n- Market sentiment tracking\n- Portfolio insights",
        'default': "I'm your AI stock analysis assistant. I can analyze stocks, provide market updates, and help with investment decisions. Ask me about specific stocks or say 'What is the update for today?' for the latest market insights."
    }
    
    user_input_lower = user_input.lower()
    
    if 'help' in user_input_lower:
        return responses['help']
    elif 'features' in user_input_lower or 'what can you do' in user_input_lower:
        return responses['features']
    else:
        return responses['default']

def display_daily_market_summary(db, data_retrieval, technical_analysis, 
                                fundamental_analysis, recommendation_engine, ml_engine):
    """Display daily market summary"""
    
    # Check if we have daily discovery results
    if 'daily_discovery' in st.session_state:
        results = st.session_state.daily_discovery
        
        st.markdown("#### 📊 Today's Discoveries")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_stocks = sum(len(stocks) for stocks in results.values())
        buy_signals = sum(1 for stocks in results.values() 
                         for stock in stocks if stock['action'] == 'BUY')
        
        with col1:
            st.metric("Stocks Analyzed", total_stocks)
        with col2:
            st.metric("Buy Signals", buy_signals)
        with col3:
            st.metric("Categories", len(results))
        with col4:
            st.metric("Success Rate", f"{(buy_signals/total_stocks*100):.1f}%" if total_stocks > 0 else "0%")
        
        # Display results by category
        for category, stocks in results.items():
            if stocks:
                st.markdown(f"**{category}:**")
                for stock in stocks:
                    action_color = "🟢" if stock['action'] == 'BUY' else "🟡" if stock['action'] == 'HOLD' else "🔴"
                    st.markdown(f"{action_color} {stock['ticker']}: {stock['action']} (Score: {stock['score']:.1f})")
    
    else:
        st.info("💡 Ask me 'What is the update for today?' to get the latest market discoveries!")

def display_quick_actions_panel(db, data_retrieval, technical_analysis, 
                               fundamental_analysis, recommendation_engine, ml_engine):
    """Display quick actions panel"""
    
    if st.button("📊 Daily Update", use_container_width=True):
        response = execute_daily_discovery(db, data_retrieval, technical_analysis, 
                                         fundamental_analysis, recommendation_engine, ml_engine)
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.rerun()
    
    if st.button("🔥 Trending Analysis", use_container_width=True):
        response = execute_trending_analysis(db, data_retrieval, technical_analysis, 
                                           fundamental_analysis, recommendation_engine, ml_engine)
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.rerun()
    
    if st.button("🌐 Market Sentiment", use_container_width=True):
        response = execute_market_analysis(db, data_retrieval, technical_analysis, 
                                         fundamental_analysis, recommendation_engine, ml_engine)
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.rerun()

def display_market_status():
    """Display current market status"""
    
    try:
        # Get current market data
        spy = yf.Ticker('SPY')
        spy_data = spy.history(period='1d')
        
        if not spy_data.empty:
            current_price = spy_data['Close'].iloc[-1]
            st.metric("S&P 500 (SPY)", f"${current_price:.2f}")
            st.success("✅ Markets Active")
        else:
            st.warning("⚠️ Market data unavailable")
            
    except Exception as e:
        st.error("❌ Market connection error")

def display_premium_features_compact():
    """Display premium features in compact format"""
    
    features = [
        "🤖 ML Enhancement",
        "📊 Advanced TA",
        "📰 News Analysis",
        "🔍 Pattern Detection",
        "💬 Unlimited Chat",
        "⚡ Real-time Data"
    ]
    
    for feature in features:
        st.markdown(f"✅ {feature}")

def display_modern_analysis_results(results):
    """Display analysis results in modern format"""
    
    if not results:
        return
    
    ticker = results.get('ticker', 'Unknown')
    recommendation = results.get('recommendation', {})
    
    # Header
    st.markdown(f"### 📈 {ticker} Analysis Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Action", recommendation.get('action', 'HOLD'))
    with col2:
        st.metric("Score", f"{recommendation.get('final_score', 0):.1f}/100")
    with col3:
        st.metric("Price", f"${results.get('current_price', 0):.2f}")
    with col4:
        st.metric("Confidence", recommendation.get('confidence', 'Medium'))
    
    # Chart
    if 'stock_data' in results:
        stock_data = results['stock_data']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f"{ticker} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Reasoning
    reasoning = recommendation.get('reasoning', '')
    if reasoning:
        st.markdown("#### 📝 Analysis Summary")
        st.markdown(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning)