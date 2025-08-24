"""
Database models for the AI Stock Analysis Bot
PostgreSQL database models using SQLAlchemy
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import logging

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    # Fallback to SQLite for development
    DATABASE_URL = 'sqlite:///stock_analysis.db'

# Create database engine
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()

class User(Base):
    """User model for service tier management"""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    service_tier = Column(String(20), default='free')  # free, premium, extra_premium
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    usage_records = relationship("UsageRecord", back_populates="user")
    analysis_results = relationship("AnalysisResult", back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")

class Stock(Base):
    """Stock information model"""
    __tablename__ = 'stocks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(10), unique=True, index=True)
    company_name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    price = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    stock_data = relationship("StockData", back_populates="stock")
    analysis_results = relationship("AnalysisResult", back_populates="stock")
    news_articles = relationship("NewsArticle", back_populates="stock")
    analyst_ratings = relationship("AnalystRating", back_populates="stock")

class StockData(Base):
    """Historical stock price data"""
    __tablename__ = 'stock_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stock_id = Column(UUID(as_uuid=True), ForeignKey('stocks.id'))
    period = Column(String(20))  # 1d, 1wk, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y
    data = Column(JSON)  # OHLCV data as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="stock_data")

class AnalysisResult(Base):
    """Stock analysis results"""
    __tablename__ = 'analysis_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    stock_id = Column(UUID(as_uuid=True), ForeignKey('stocks.id'))
    analysis_type = Column(String(50))  # technical, fundamental, combined
    period = Column(String(20))
    
    # Analysis scores
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    final_score = Column(Float)
    
    # Recommendation
    action = Column(String(10))  # BUY, SELL, HOLD
    confidence = Column(String(10))  # Low, Medium, High
    risk_level = Column(String(10))  # Low, Medium, High
    
    # Detailed results
    technical_indicators = Column(JSON)
    fundamental_metrics = Column(JSON)
    patterns_detected = Column(JSON)
    reasoning = Column(Text)
    
    # ML enhancement
    ml_enhanced = Column(Boolean, default=False)
    ml_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="analysis_results")
    stock = relationship("Stock", back_populates="analysis_results")

class UsageRecord(Base):
    """User usage tracking"""
    __tablename__ = 'usage_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    resource_type = Column(String(50))  # stocks, nl_queries, alerts, api_calls
    usage_count = Column(Integer, default=1)
    date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="usage_records")

class NewsArticle(Base):
    """Stock news articles"""
    __tablename__ = 'news_articles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stock_id = Column(UUID(as_uuid=True), ForeignKey('stocks.id'))
    title = Column(String(500))
    content = Column(Text)
    url = Column(String(1000))
    source = Column(String(100))
    sentiment = Column(String(20))  # positive, negative, neutral
    sentiment_score = Column(Float)
    published_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="news_articles")

class AnalystRating(Base):
    """Analyst ratings and price targets"""
    __tablename__ = 'analyst_ratings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stock_id = Column(UUID(as_uuid=True), ForeignKey('stocks.id'))
    analyst_firm = Column(String(100))
    rating = Column(String(20))  # Strong Buy, Buy, Hold, Sell, Strong Sell
    price_target = Column(Float)
    previous_rating = Column(String(20))
    rating_change = Column(String(20))  # upgrade, downgrade, maintain
    published_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="analyst_ratings")

class ChatHistory(Base):
    """Natural language chat history"""
    __tablename__ = 'chat_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    query = Column(Text)
    response = Column(Text)
    query_type = Column(String(50))
    confidence = Column(String(10))
    ai_powered = Column(Boolean, default=False)
    context = Column(JSON)  # Analysis context
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="chat_history")

class DiscoveryResult(Base):
    """Daily stock discovery results"""
    __tablename__ = 'discovery_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    discovery_date = Column(DateTime, default=datetime.utcnow)
    source = Column(String(100))  # yahoo, finviz, reddit, etc.
    trending_stocks = Column(JSON)  # List of tickers
    total_discovered = Column(Integer)
    total_analyzed = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class MLModel(Base):
    """Machine learning model tracking"""
    __tablename__ = 'ml_models'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100))
    model_type = Column(String(50))  # pattern_classifier, score_predictor
    version = Column(String(20))
    accuracy = Column(Float)
    training_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    model_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

class TradingSignal(Base):
    """Trading signals generated by the system"""
    __tablename__ = 'trading_signals'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stock_id = Column(UUID(as_uuid=True), ForeignKey('stocks.id'))
    signal_type = Column(String(20))  # BUY, SELL, HOLD
    strength = Column(Float)  # 0-1 scale
    trigger_price = Column(Float)
    target_price = Column(Float)
    stop_loss = Column(Float)
    valid_until = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables created successfully")
    except Exception as e:
        logging.error(f"Error creating database tables: {str(e)}")
        raise

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database with default data"""
    create_tables()
    
    # Create demo user if not exists
    db = SessionLocal()
    try:
        demo_user = db.query(User).filter(User.username == 'demo_user').first()
        if not demo_user:
            demo_user = User(
                username='demo_user',
                email='demo@example.com',
                service_tier='free'
            )
            db.add(demo_user)
            db.commit()
            logging.info("Demo user created")
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_database()