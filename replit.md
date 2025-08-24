# AI Stock Analysis Bot - Replit.md

## Overview

This is a comprehensive AI-driven stock analysis application that autonomously discovers, analyzes, and recommends stocks using advanced machine learning, natural language processing, and automated web scraping. The system features proactive stock discovery, tiered service levels, and conversational AI capabilities for discussing analysis results.

## User Preferences

Preferred communication style: Simple, everyday language.
User Interface Preferences: Clean, minimal design without large text blocks or excessive headers. User-friendly buttons and neat layout prioritized.
UI Design: Simplified tier-based interface with clear explanations of analysis methods.
Analysis Flow: Technical analysis for free tier with detailed explanations, fundamental analysis for premium tier with reasoning.
Tier System: Free tier provides technical analysis with explanations, Premium tier adds fundamental analysis with detailed reasoning.

## System Architecture

### Core Architecture
- **Modular Design**: Separate modules for different analysis types (technical, fundamental, recommendation)
- **Data Layer**: SQLite database for caching stock data and analysis results
- **Service Layer**: Individual analysis services that can be composed together
- **Presentation Layer**: Streamlit web UI and CLI for different use cases

### Technology Stack
- **Backend**: Python with yfinance for data retrieval
- **Technical Analysis**: pandas-ta for indicators and pattern detection
- **Database**: SQLite for local caching
- **Frontend**: Streamlit for interactive web interface
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas and NumPy for efficient data manipulation

## Key Components

### 1. Data Retrieval (`data_retrieval.py`)
- **Purpose**: Fetch stock data from yfinance API with intelligent caching
- **Features**: 
  - Automatic caching to reduce API calls
  - Support for multiple time periods
  - Data validation and cleaning
  - Force refresh capability

### 2. Technical Analysis (`technical_analysis.py`)
- **Purpose**: Analyze price patterns and technical indicators
- **Features**:
  - Moving averages (SMA, EMA)
  - Momentum indicators (RSI, MACD)
  - Support/resistance detection
  - Trend analysis
  - Pattern recognition scoring

### 3. Fundamental Analysis (`fundamental_analysis.py`)
- **Purpose**: Evaluate company financial health and metrics
- **Features**:
  - Financial ratios (P/E, ROE, ROA, etc.)
  - Valuation metrics
  - Profitability analysis
  - Growth metrics scoring

### 4. Recommendation Engine (`recommendation.py`)
- **Purpose**: Combine technical and fundamental analysis for investment recommendations
- **Features**:
  - Weighted scoring system (70% TA, 30% FA)
  - Risk assessment
  - Confidence calculations
  - Detailed reasoning generation

### 5. Database Layer (`database.py`)
- **Purpose**: Handle data persistence and caching
- **Features**:
  - SQLite-based caching system
  - Separate tables for stock data, company info, and analysis results
  - Automatic cache expiration
  - Data integrity management

### 6. Machine Learning Engine (`ml_engine.py`)
- **Purpose**: Implement machine learning for pattern recognition and score prediction
- **Features**:
  - Random Forest classifier for pattern detection
  - Gradient Boosting regressor for score prediction
  - Continuous learning from analysis outcomes
  - Feature engineering from technical indicators
  - Model persistence and loading

### 7. Web Scraper (`web_scraper.py`)
- **Purpose**: Autonomous stock discovery and information gathering
- **Features**:
  - Trending stock discovery from multiple sources
  - News article scraping and sentiment analysis
  - Analyst ratings and recommendations
  - Insider trading activity monitoring
  - SEC filings and regulatory updates

### 8. Scheduler (`scheduler.py`)
- **Purpose**: Automated daily analysis and discovery
- **Features**:
  - Daily stock discovery at market open
  - Comprehensive analysis of discovered stocks
  - Weekly ML model updates
  - Hourly trending stock updates
  - Manual job triggering for testing

### 9. Natural Language Interface (`natural_language_interface.py`)
- **Purpose**: Conversational AI for discussing analysis results
- **Features**:
  - OpenAI-powered natural language processing
  - Query classification and context understanding
  - Tier-based response filtering
  - Conversation history management
  - Suggested questions generation

### 10. Service Tier Manager (`service_tiers.py`)
- **Purpose**: Manage different subscription levels and feature access
- **Features**:
  - Three-tier service model (Free, Premium, Extra Premium)
  - Daily usage limits and tracking
  - Feature filtering based on tier
  - Upgrade suggestions and comparisons
  - Usage analytics and reporting

### 11. Utilities (`utils.py`)
- **Purpose**: Common helper functions
- **Features**:
  - Currency and percentage formatting
  - Ticker validation
  - Data type conversions

## Data Flow

1. **User Input**: User selects stocks and analysis parameters
2. **Data Retrieval**: System fetches live stock data directly from yfinance API
3. **Technical Analysis**: Price data analyzed for patterns and indicators
4. **Fundamental Analysis**: Company metrics evaluated (if enabled)
5. **ML Enhancement**: Machine learning models improve analysis accuracy
6. **Recommendation**: Combined analysis produces scored recommendations
7. **Presentation**: Results displayed via Streamlit UI with real-time data

## External Dependencies

### Primary APIs
- **yfinance**: Main data source for stock prices and company information
- **Alpha Vantage**: Mentioned as potential fallback (not yet implemented)
- **Financial Modeling Prep**: Alternative data source (not yet implemented)

### Python Libraries
- **pandas**: Data manipulation and analysis
- **pandas-ta**: Technical analysis indicators
- **plotly**: Interactive charting
- **streamlit**: Web interface
- **sqlite3**: Database operations
- **scipy**: Scientific computing for pattern detection
- **numpy**: Numerical operations

## Deployment Strategy

### Current Setup
- **Platform**: Designed for Replit deployment
- **Database**: SQLite file-based storage (suitable for single-instance deployment)
- **Caching**: Local file-based caching system
- **Scaling**: Designed for moderate workloads (hundreds of stocks)

### Interfaces
- **Web Interface**: Streamlit app (`app.py`) for interactive analysis
- **CLI Interface**: Command-line tool (`main.py`) for batch processing
- **Modular Design**: Easy to extend with additional analysis modules

### Performance Considerations
- **Caching Strategy**: Aggressive caching to minimize API calls
- **Batch Processing**: CLI supports analyzing multiple stocks efficiently
- **Database Optimization**: Indexed tables for fast lookups
- **Memory Management**: Efficient pandas operations for large datasets

### Recent Enhancements (January 2025)
- **✅ Machine Learning Engine**: Implemented Random Forest and Gradient Boosting models
- **✅ Automated Discovery**: Daily stock discovery from multiple sources
- **✅ Natural Language Interface**: OpenAI-powered conversational AI
- **✅ Service Tiers**: Three-tier subscription model (Free, Premium, Extra Premium)
- **✅ Web Scraping**: Automated news and analyst data collection
- **✅ Scheduler**: Background automation for daily analysis
- **✅ ML-Enhanced Analysis**: Machine learning improves technical analysis accuracy
- **✅ PostgreSQL Database**: Upgraded from SQLite to PostgreSQL with comprehensive schema
- **✅ User Authentication**: Login system with user management and service tier tracking
- **✅ Live Data Integration**: Real-time market data analysis with yfinance API
- **✅ Comprehensive Technical Analysis**: Full technical indicators with live market data
- **✅ Extra Premium Access**: User upgraded to highest tier with unlimited features
- **✅ Enhanced UI/UX**: Redesigned interface with AI chat on left, configuration on right
- **✅ Premium Features Activated**: All premium features unlocked and accessible
- **✅ Improved User Experience**: Modern styling with gradient backgrounds and cards
- **✅ Direct Data Integration**: Bypassed database caching for real-time analysis
- **✅ Fixed Chat Interface**: Modern bubble styling with proper input handling
- **✅ Working Auto Discovery**: All discovery features operational with live data
- **✅ Complete UI Redesign**: Modern interactive interface with AI chat assistant
- **✅ Interactive AI Assistant**: Can execute commands like "What is the update for today?"
- **✅ Fixed Technical Issues**: Resolved deprecated pandas methods and price calculation errors
- **✅ Verified API Status**: yfinance API fully operational with live market data
- **✅ Signal Generation Working**: System produces BUY/HOLD/SELL signals correctly
- **✅ Startup-Style UI**: Modern interface with trendy colors and fonts for younger generation
- **✅ Technical-First Approach**: Analysis starts with technical indicators, option to add fundamentals
- **✅ Fixed LaTeX Errors**: Replaced bullet points causing rendering issues
- **✅ Error Handling**: Graceful database connection handling and offline mode
- **✅ Premium Features**: Automated web scraping, AI chat interface, and advanced analysis
- **✅ Premium Discovery**: Auto-discovery of stocks with confidence levels and reasoning
- **✅ AI Chat Interface**: Context-aware chat for discussing analysis results and methods
- **✅ Enhanced Full-Screen UI**: Complete interface redesign with persistent AI chatbot tracking all interactions
- **✅ Real-Time Data Integration**: Live market data with minute-by-minute updates from free APIs
- **✅ Configurable Analysis Rules**: User-customizable technical and fundamental analysis parameters
- **✅ Free/Premium Tier System**: Free tier provides analysis discussion, Premium adds automated recommendations
- **✅ Interaction Tracking**: AI chatbot maintains complete session history and context awareness
- **✅ Pure Chat Interface**: Full-screen chat interface connected to OpenAI with small recommendations panel
- **✅ Conversational Analysis**: All stock analysis and fundamental discussions happen through natural conversation
- **✅ Daily Recommendations Integration**: Small panel shows daily picks that can be discussed in chat for deeper analysis
- **✅ Simplified Tier-Based Interface**: Free tier provides technical analysis with explanations, Premium adds fundamental analysis
- **✅ Educational Analysis Approach**: Detailed explanations of how technical and fundamental conclusions are reached
- **✅ Clean Non-Chat Interface**: Moved away from complex chat interface to simple analysis-focused design
- **✅ Market Trend Button**: Removed manual stock input, replaced with automated market trend recommendations button
- **✅ Focus on Automation**: System now purely focuses on AI-generated recommendations based on current market trends
- **✅ Working Recommendations**: Fixed data requirements, system now successfully generates 8 trending stock recommendations
- **✅ Portfolio Optimization Ideas**: Received comprehensive enhancement suggestions for weekly portfolio optimization, event-driven alerts, multi-agent systems, sentiment analysis, backtesting tools, and fundamental screening
- **✅ Multi-Agent System**: Implemented specialized AI agents (Technical, Fundamental, Sentiment, Manager) for comprehensive stock analysis
- **✅ AI-Powered Synthesis**: Manager agent uses Grok AI to create investment thesis from multi-agent analysis
- **✅ Premium Plus Feature**: Multi-agent system exclusive to highest tier alongside chatbot functionality

### Future Enhancements
- **Options Analysis**: Advanced derivatives and options strategies
- **Backtesting Engine**: Historical performance validation
- **Portfolio Optimization**: AI-driven portfolio construction
- **Risk Management**: Advanced risk modeling and alerts