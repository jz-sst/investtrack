"""
Scheduler for AI Stock Analysis Bot
Handles daily scraping, analysis, and ML model updates
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import json
import os

from web_scraper import WebScraper
from data_retrieval import DataRetrieval
from technical_analysis import TechnicalAnalysis
from fundamental_analysis import FundamentalAnalysis
from recommendation import RecommendationEngine
from ml_engine import MLEngine
from database import Database

class StockAnalysisScheduler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scheduler = BackgroundScheduler()
        self.is_running = False
        
        # Initialize components
        self.db = Database()
        self.scraper = WebScraper()
        self.data_retrieval = DataRetrieval(self.db)
        self.technical_analysis = TechnicalAnalysis()
        self.fundamental_analysis = FundamentalAnalysis()
        self.recommendation_engine = RecommendationEngine()
        self.ml_engine = MLEngine()
        
        # Load ML models
        self.ml_engine.load_models()
        
        # Results storage
        self.daily_results = []
        self.analysis_cache = {}
        
    def start_scheduler(self):
        """Start the background scheduler"""
        try:
            if not self.is_running:
                # Schedule daily tasks
                self.scheduler.add_job(
                    self.daily_stock_discovery,
                    'cron',
                    hour=9,  # 9 AM
                    minute=0,
                    id='daily_discovery'
                )
                
                self.scheduler.add_job(
                    self.daily_analysis_run,
                    'cron',
                    hour=10,  # 10 AM
                    minute=0,
                    id='daily_analysis'
                )
                
                self.scheduler.add_job(
                    self.weekly_ml_update,
                    'cron',
                    day_of_week='sun',
                    hour=2,  # 2 AM Sunday
                    minute=0,
                    id='weekly_ml_update'
                )
                
                self.scheduler.add_job(
                    self.hourly_trending_update,
                    'cron',
                    minute=0,  # Every hour
                    id='hourly_trending'
                )
                
                self.scheduler.start()
                self.is_running = True
                
                self.logger.info("Stock analysis scheduler started")
                
        except Exception as e:
            self.logger.error(f"Error starting scheduler: {str(e)}")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            self.logger.info("Stock analysis scheduler stopped")
    
    def daily_stock_discovery(self):
        """Daily task to discover trending stocks"""
        try:
            self.logger.info("Starting daily stock discovery...")
            
            # Discover trending stocks
            trending_stocks = self.scraper.discover_trending_stocks()
            
            # Store in database
            discovery_data = {
                'date': datetime.now().isoformat(),
                'trending_stocks': trending_stocks,
                'discovery_method': 'automated_scraping'
            }
            
            # Save to cache
            self.analysis_cache['daily_discovery'] = discovery_data
            
            self.logger.info(f"Discovered {len(trending_stocks)} trending stocks")
            
        except Exception as e:
            self.logger.error(f"Error in daily stock discovery: {str(e)}")
    
    def daily_analysis_run(self):
        """Daily comprehensive analysis of discovered stocks"""
        try:
            self.logger.info("Starting daily analysis run...")
            
            # Get discovered stocks
            discovery_data = self.analysis_cache.get('daily_discovery', {})
            trending_stocks = discovery_data.get('trending_stocks', [])
            
            if not trending_stocks:
                self.logger.warning("No trending stocks found for analysis")
                return
            
            daily_results = []
            
            for ticker in trending_stocks[:50]:  # Limit to top 50
                try:
                    # Comprehensive analysis
                    result = self.analyze_single_stock(ticker)
                    if result:
                        daily_results.append(result)
                        
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {ticker}: {str(e)}")
                    continue
            
            # Sort by recommendation score
            daily_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            # Store results
            self.daily_results = daily_results
            
            # Update analysis cache
            self.analysis_cache['daily_analysis'] = {
                'date': datetime.now().isoformat(),
                'total_analyzed': len(daily_results),
                'top_picks': daily_results[:10]
            }
            
            self.logger.info(f"Daily analysis completed. Analyzed {len(daily_results)} stocks")
            
        except Exception as e:
            self.logger.error(f"Error in daily analysis run: {str(e)}")
    
    def analyze_single_stock(self, ticker):
        """Analyze a single stock comprehensively"""
        try:
            # Get stock data
            stock_data = self.data_retrieval.get_stock_data(ticker, '1y')
            if stock_data.empty:
                return None
            
            # Technical analysis
            ta_results = self.technical_analysis.analyze_stock(stock_data, ticker)
            
            # Fundamental analysis
            fa_results = self.fundamental_analysis.analyze_stock(ticker)
            
            # ML enhancement
            ml_score = self.ml_engine.predict_score(stock_data, ta_results.get('indicators', {}))
            
            # Combine traditional and ML scores
            combined_ta_score = (ta_results['score'] + ml_score) / 2
            ta_results['score'] = combined_ta_score
            
            # Web scraping data
            scraped_data = self.scraper.comprehensive_stock_scrape(ticker)
            
            # Get recommendation
            recommendation = self.recommendation_engine.get_recommendation(
                ticker, ta_results, fa_results
            )
            
            # Enhanced recommendation with scraped data
            recommendation = self.enhance_recommendation_with_scraped_data(
                recommendation, scraped_data
            )
            
            # Compile comprehensive result
            result = {
                'ticker': ticker,
                'analysis_date': datetime.now().isoformat(),
                'stock_data': stock_data.to_dict(),
                'ta_results': ta_results,
                'fa_results': fa_results,
                'scraped_data': scraped_data,
                'recommendation': recommendation,
                'final_score': recommendation['final_score'],
                'ml_enhanced': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {ticker}: {str(e)}")
            return None
    
    def enhance_recommendation_with_scraped_data(self, recommendation, scraped_data):
        """Enhance recommendation with scraped data"""
        try:
            # Add news sentiment
            news_items = scraped_data.get('news', [])
            if news_items:
                recommendation['news_summary'] = f"Recent news coverage: {len(news_items)} articles"
            
            # Add analyst ratings
            analyst_ratings = scraped_data.get('analyst_ratings', {})
            if analyst_ratings:
                recommendation['analyst_consensus'] = analyst_ratings.get('average_rating', {})
            
            # Add insider trading sentiment
            insider_trading = scraped_data.get('insider_trading', {})
            if insider_trading:
                net_buying = insider_trading.get('net_buying', 0)
                net_selling = insider_trading.get('net_selling', 0)
                
                if net_buying > net_selling:
                    recommendation['insider_sentiment'] = "Positive (Net Buying)"
                elif net_selling > net_buying:
                    recommendation['insider_sentiment'] = "Negative (Net Selling)"
                else:
                    recommendation['insider_sentiment'] = "Neutral"
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error enhancing recommendation: {str(e)}")
            return recommendation
    
    def weekly_ml_update(self):
        """Weekly ML model update"""
        try:
            self.logger.info("Starting weekly ML model update...")
            
            # Collect recent analysis data for training
            training_data = self.collect_training_data()
            
            if len(training_data) > 10:
                # Retrain models
                self.ml_engine.train_pattern_classifier(training_data)
                self.ml_engine.train_score_predictor(training_data)
                
                self.logger.info("ML models updated successfully")
            else:
                self.logger.warning("Not enough training data for ML update")
                
        except Exception as e:
            self.logger.error(f"Error in weekly ML update: {str(e)}")
    
    def collect_training_data(self):
        """Collect training data from recent analyses"""
        try:
            # This would collect actual performance data
            # For now, returning sample data structure
            training_data = []
            
            # In production, you'd query the database for recent analyses
            # and their actual performance outcomes
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error collecting training data: {str(e)}")
            return []
    
    def hourly_trending_update(self):
        """Update trending stocks every hour"""
        try:
            # Quick update of trending stocks
            trending_stocks = self.scraper.discover_trending_stocks()
            
            # Update cache
            self.analysis_cache['hourly_trending'] = {
                'timestamp': datetime.now().isoformat(),
                'trending_stocks': trending_stocks[:20]
            }
            
            self.logger.info(f"Updated trending stocks: {len(trending_stocks)} found")
            
        except Exception as e:
            self.logger.error(f"Error in hourly trending update: {str(e)}")
    
    def get_daily_recommendations(self, tier='free'):
        """
        Get daily recommendations based on service tier
        
        Args:
            tier (str): Service tier - 'free', 'premium', or 'extra_premium'
            
        Returns:
            list: Filtered recommendations based on tier
        """
        try:
            if not self.daily_results:
                return []
            
            if tier == 'free':
                # Free tier: Top 3 recommendations, basic info
                return self.filter_recommendations_for_free(self.daily_results[:3])
            
            elif tier == 'premium':
                # Premium tier: Top 10 recommendations, detailed analysis
                return self.filter_recommendations_for_premium(self.daily_results[:10])
            
            elif tier == 'extra_premium':
                # Extra premium: All recommendations, full analysis + ML insights
                return self.filter_recommendations_for_extra_premium(self.daily_results)
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting daily recommendations: {str(e)}")
            return []
    
    def filter_recommendations_for_free(self, recommendations):
        """Filter recommendations for free tier"""
        filtered = []
        for rec in recommendations:
            filtered_rec = {
                'ticker': rec['ticker'],
                'action': rec['recommendation']['action'],
                'final_score': rec['recommendation']['final_score'],
                'reasoning': rec['recommendation']['reasoning'][:500] + "..." if len(rec['recommendation']['reasoning']) > 500 else rec['recommendation']['reasoning']
            }
            filtered.append(filtered_rec)
        return filtered
    
    def filter_recommendations_for_premium(self, recommendations):
        """Filter recommendations for premium tier"""
        filtered = []
        for rec in recommendations:
            filtered_rec = {
                'ticker': rec['ticker'],
                'recommendation': rec['recommendation'],
                'ta_score': rec['ta_results']['score'],
                'fa_score': rec['fa_results']['score'] if rec['fa_results'] else None,
                'patterns': rec['ta_results']['patterns'],
                'news_summary': rec['recommendation'].get('news_summary', 'No recent news'),
                'analyst_consensus': rec['recommendation'].get('analyst_consensus', {})
            }
            filtered.append(filtered_rec)
        return filtered
    
    def filter_recommendations_for_extra_premium(self, recommendations):
        """Filter recommendations for extra premium tier"""
        # Return full analysis for extra premium
        return recommendations
    
    def get_scheduler_status(self):
        """Get current scheduler status"""
        return {
            'is_running': self.is_running,
            'jobs': [job.id for job in self.scheduler.get_jobs()],
            'next_run_times': {job.id: job.next_run_time for job in self.scheduler.get_jobs()},
            'last_discovery': self.analysis_cache.get('daily_discovery', {}).get('date'),
            'last_analysis': self.analysis_cache.get('daily_analysis', {}).get('date'),
            'total_analyzed_today': len(self.daily_results)
        }
    
    def manual_trigger(self, job_type):
        """Manually trigger a scheduled job"""
        try:
            if job_type == 'discovery':
                self.daily_stock_discovery()
            elif job_type == 'analysis':
                self.daily_analysis_run()
            elif job_type == 'ml_update':
                self.weekly_ml_update()
            elif job_type == 'trending':
                self.hourly_trending_update()
            
            self.logger.info(f"Manually triggered {job_type} job")
            
        except Exception as e:
            self.logger.error(f"Error manually triggering {job_type}: {str(e)}")

# Global scheduler instance
scheduler_instance = None

def get_scheduler():
    """Get global scheduler instance"""
    global scheduler_instance
    if scheduler_instance is None:
        scheduler_instance = StockAnalysisScheduler()
    return scheduler_instance