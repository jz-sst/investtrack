"""
Database service for AI Stock Analysis Bot
High-level database operations using PostgreSQL models
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import json
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_

from database_models import (
    SessionLocal, User, Stock, StockData, AnalysisResult, 
    UsageRecord, NewsArticle, AnalystRating, ChatHistory,
    DiscoveryResult, MLModel, TradingSignal, init_database
)

class DatabaseService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        try:
            init_database()
            self.logger.info("Database service initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
    
    def get_db(self) -> Session:
        """Get database session"""
        return SessionLocal()
    
    # User management
    def get_or_create_user(self, username: str, email: str = None, service_tier: str = 'free') -> User:
        """Get existing user or create new one"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                user = User(
                    username=username,
                    email=email or f"{username}@example.com",
                    service_tier=service_tier
                )
                db.add(user)
                db.commit()
                db.refresh(user)
            return user
        except Exception as e:
            self.logger.error(f"Error getting/creating user: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def get_user_tier(self, username: str) -> str:
        """Get user's service tier"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.username == username).first()
            return user.service_tier if user else 'free'
        except Exception as e:
            self.logger.error(f"Error getting user tier: {str(e)}")
            return 'free'
        finally:
            db.close()
    
    def update_user_tier(self, username: str, new_tier: str) -> bool:
        """Update user's service tier"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.username == username).first()
            if user:
                user.service_tier = new_tier
                user.updated_at = datetime.utcnow()
                db.commit()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating user tier: {str(e)}")
            db.rollback()
            return False
        finally:
            db.close()
    
    # Stock management
    def get_or_create_stock(self, ticker: str, company_name: str = None) -> Stock:
        """Get existing stock or create new one"""
        db = self.get_db()
        try:
            stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
            if not stock:
                stock = Stock(
                    ticker=ticker.upper(),
                    company_name=company_name or ticker.upper()
                )
                db.add(stock)
                db.commit()
                db.refresh(stock)
            return stock
        except Exception as e:
            self.logger.error(f"Error getting/creating stock: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def update_stock_info(self, ticker: str, info: Dict) -> bool:
        """Update stock information"""
        db = self.get_db()
        try:
            stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
            if stock:
                stock.company_name = info.get('longName', stock.company_name)
                stock.sector = info.get('sector')
                stock.industry = info.get('industry')
                stock.market_cap = info.get('marketCap')
                stock.price = info.get('currentPrice')
                stock.volume = info.get('volume')
                stock.updated_at = datetime.utcnow()
                db.commit()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating stock info: {str(e)}")
            db.rollback()
            return False
        finally:
            db.close()
    
    # Stock data caching
    def cache_stock_data(self, ticker: str, period: str, data: pd.DataFrame) -> bool:
        """Cache stock data"""
        db = self.get_db()
        try:
            stock = self.get_or_create_stock(ticker)
            
            # Check if data exists
            stock_data = db.query(StockData).filter(
                and_(StockData.stock_id == stock.id, StockData.period == period)
            ).first()
            
            data_json = data.to_dict('records') if not data.empty else []
            
            if stock_data:
                stock_data.data = data_json
                stock_data.updated_at = datetime.utcnow()
            else:
                stock_data = StockData(
                    stock_id=stock.id,
                    period=period,
                    data=data_json
                )
                db.add(stock_data)
            
            db.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error caching stock data: {str(e)}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_cached_stock_data(self, ticker: str, period: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """Get cached stock data"""
        db = self.get_db()
        try:
            stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
            if not stock:
                return None
            
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            stock_data = db.query(StockData).filter(
                and_(
                    StockData.stock_id == stock.id,
                    StockData.period == period,
                    StockData.updated_at > cutoff_time
                )
            ).first()
            
            if stock_data and stock_data.data:
                return pd.DataFrame(stock_data.data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting cached stock data: {str(e)}")
            return None
        finally:
            db.close()
    
    # Analysis results
    def save_analysis_result(self, username: str, ticker: str, analysis_data: Dict) -> bool:
        """Save analysis result"""
        db = self.get_db()
        try:
            user = self.get_or_create_user(username)
            stock = self.get_or_create_stock(ticker)
            
            analysis_result = AnalysisResult(
                user_id=user.id,
                stock_id=stock.id,
                analysis_type=analysis_data.get('analysis_type', 'combined'),
                period=analysis_data.get('period', '1y'),
                technical_score=analysis_data.get('ta_results', {}).get('score', 0),
                fundamental_score=analysis_data.get('fa_results', {}).get('score', 0),
                final_score=analysis_data.get('recommendation', {}).get('final_score', 0),
                action=analysis_data.get('recommendation', {}).get('action', 'HOLD'),
                confidence=analysis_data.get('recommendation', {}).get('confidence', 'Medium'),
                risk_level=analysis_data.get('recommendation', {}).get('risk_assessment', 'Medium'),
                technical_indicators=analysis_data.get('ta_results', {}).get('indicators', {}),
                fundamental_metrics=analysis_data.get('fa_results', {}).get('metrics', {}),
                patterns_detected=analysis_data.get('ta_results', {}).get('patterns', []),
                reasoning=analysis_data.get('recommendation', {}).get('reasoning', ''),
                ml_enhanced=analysis_data.get('ml_enhanced', False),
                ml_score=analysis_data.get('ml_score', 0)
            )
            
            db.add(analysis_result)
            db.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error saving analysis result: {str(e)}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_recent_analysis_results(self, username: str, limit: int = 10) -> List[Dict]:
        """Get recent analysis results for user"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                return []
            
            results = db.query(AnalysisResult, Stock).join(Stock).filter(
                AnalysisResult.user_id == user.id
            ).order_by(desc(AnalysisResult.created_at)).limit(limit).all()
            
            analysis_list = []
            for result, stock in results:
                analysis_list.append({
                    'ticker': stock.ticker,
                    'company_name': stock.company_name,
                    'final_score': result.final_score,
                    'action': result.action,
                    'confidence': result.confidence,
                    'risk_level': result.risk_level,
                    'reasoning': result.reasoning,
                    'ml_enhanced': result.ml_enhanced,
                    'created_at': result.created_at.isoformat()
                })
            
            return analysis_list
        except Exception as e:
            self.logger.error(f"Error getting recent analysis results: {str(e)}")
            return []
        finally:
            db.close()
    
    # Usage tracking
    def record_usage(self, username: str, resource_type: str, usage_count: int = 1) -> bool:
        """Record user usage"""
        db = self.get_db()
        try:
            user = self.get_or_create_user(username)
            
            # Check if there's already a record for today
            today = datetime.utcnow().date()
            existing_record = db.query(UsageRecord).filter(
                and_(
                    UsageRecord.user_id == user.id,
                    UsageRecord.resource_type == resource_type,
                    func.date(UsageRecord.date) == today
                )
            ).first()
            
            if existing_record:
                existing_record.usage_count += usage_count
            else:
                usage_record = UsageRecord(
                    user_id=user.id,
                    resource_type=resource_type,
                    usage_count=usage_count
                )
                db.add(usage_record)
            
            db.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error recording usage: {str(e)}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_daily_usage(self, username: str) -> Dict:
        """Get daily usage statistics"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                return {}
            
            today = datetime.utcnow().date()
            records = db.query(UsageRecord).filter(
                and_(
                    UsageRecord.user_id == user.id,
                    func.date(UsageRecord.date) == today
                )
            ).all()
            
            usage = {}
            for record in records:
                usage[record.resource_type] = record.usage_count
            
            return usage
        except Exception as e:
            self.logger.error(f"Error getting daily usage: {str(e)}")
            return {}
        finally:
            db.close()
    
    # Chat history
    def save_chat_message(self, username: str, query: str, response: str, 
                         query_type: str = 'general', ai_powered: bool = False,
                         context: Dict = None) -> bool:
        """Save chat message"""
        db = self.get_db()
        try:
            user = self.get_or_create_user(username)
            
            chat_record = ChatHistory(
                user_id=user.id,
                query=query,
                response=response,
                query_type=query_type,
                ai_powered=ai_powered,
                context=context or {}
            )
            
            db.add(chat_record)
            db.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error saving chat message: {str(e)}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_chat_history(self, username: str, limit: int = 50) -> List[Dict]:
        """Get chat history for user"""
        db = self.get_db()
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                return []
            
            chats = db.query(ChatHistory).filter(
                ChatHistory.user_id == user.id
            ).order_by(desc(ChatHistory.created_at)).limit(limit).all()
            
            chat_list = []
            for chat in chats:
                chat_list.append({
                    'query': chat.query,
                    'response': chat.response,
                    'query_type': chat.query_type,
                    'ai_powered': chat.ai_powered,
                    'timestamp': chat.created_at.isoformat()
                })
            
            return chat_list
        except Exception as e:
            self.logger.error(f"Error getting chat history: {str(e)}")
            return []
        finally:
            db.close()
    
    # Discovery results
    def save_discovery_result(self, source: str, trending_stocks: List[str]) -> bool:
        """Save daily discovery results"""
        db = self.get_db()
        try:
            discovery = DiscoveryResult(
                source=source,
                trending_stocks=trending_stocks,
                total_discovered=len(trending_stocks)
            )
            
            db.add(discovery)
            db.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error saving discovery result: {str(e)}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_latest_discovery_results(self, days: int = 7) -> List[Dict]:
        """Get latest discovery results"""
        db = self.get_db()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            results = db.query(DiscoveryResult).filter(
                DiscoveryResult.discovery_date > cutoff_date
            ).order_by(desc(DiscoveryResult.discovery_date)).all()
            
            discovery_list = []
            for result in results:
                discovery_list.append({
                    'source': result.source,
                    'trending_stocks': result.trending_stocks,
                    'total_discovered': result.total_discovered,
                    'discovery_date': result.discovery_date.isoformat()
                })
            
            return discovery_list
        except Exception as e:
            self.logger.error(f"Error getting discovery results: {str(e)}")
            return []
        finally:
            db.close()
    
    # News articles
    def save_news_article(self, ticker: str, article_data: Dict) -> bool:
        """Save news article"""
        db = self.get_db()
        try:
            stock = self.get_or_create_stock(ticker)
            
            article = NewsArticle(
                stock_id=stock.id,
                title=article_data.get('title', ''),
                content=article_data.get('summary', ''),
                url=article_data.get('link', ''),
                source=article_data.get('source', ''),
                published_at=datetime.fromisoformat(article_data.get('published', datetime.utcnow().isoformat()))
            )
            
            db.add(article)
            db.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error saving news article: {str(e)}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def get_recent_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get recent news for stock"""
        db = self.get_db()
        try:
            stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
            if not stock:
                return []
            
            articles = db.query(NewsArticle).filter(
                NewsArticle.stock_id == stock.id
            ).order_by(desc(NewsArticle.published_at)).limit(limit).all()
            
            news_list = []
            for article in articles:
                news_list.append({
                    'title': article.title,
                    'content': article.content,
                    'url': article.url,
                    'source': article.source,
                    'published_at': article.published_at.isoformat()
                })
            
            return news_list
        except Exception as e:
            self.logger.error(f"Error getting recent news: {str(e)}")
            return []
        finally:
            db.close()
    
    # Health check
    def health_check(self) -> Dict:
        """Database health check"""
        db = self.get_db()
        try:
            # Simple query to check connection
            user_count = db.query(User).count()
            stock_count = db.query(Stock).count()
            analysis_count = db.query(AnalysisResult).count()
            
            return {
                'status': 'healthy',
                'users': user_count,
                'stocks': stock_count,
                'analyses': analysis_count,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Database health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
        finally:
            db.close()

# Global database service instance
db_service = DatabaseService()