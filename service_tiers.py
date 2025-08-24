"""
Service tier management for AI Stock Analysis Bot
Handles different subscription levels and feature access
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

class ServiceTierManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tier_configs = self.load_tier_configurations()
        
    def load_tier_configurations(self) -> Dict:
        """Load service tier configurations"""
        return {
            'free': {
                'name': 'Free',
                'price': 0,
                'daily_stock_limit': 3,
                'features': {
                    'basic_analysis': True,
                    'technical_indicators': ['RSI', 'MACD', 'SMA'],
                    'fundamental_metrics': ['pe_ratio', 'market_cap'],
                    'chart_access': True,
                    'recommendation_detail': 'basic',
                    'historical_data': '1_month',
                    'alerts': 0,
                    'nl_queries_per_day': 5,
                    'pattern_detection': 'basic',
                    'news_articles': 1,
                    'export_data': False,
                    'api_access': False,
                    'priority_support': False
                },
                'limitations': {
                    'analysis_depth': 'basic',
                    'real_time_data': False,
                    'ml_insights': False,
                    'portfolio_tracking': False,
                    'custom_alerts': False
                }
            },
            
            'premium': {
                'name': 'Premium',
                'price': 29.99,
                'daily_stock_limit': 25,
                'features': {
                    'basic_analysis': True,
                    'technical_indicators': ['RSI', 'MACD', 'SMA', 'EMA', 'Bollinger_Bands', 'Stochastic', 'ATR'],
                    'fundamental_metrics': ['pe_ratio', 'market_cap', 'roe', 'debt_to_equity', 'profit_margins', 'revenue_growth'],
                    'chart_access': True,
                    'recommendation_detail': 'detailed',
                    'historical_data': '2_years',
                    'alerts': 10,
                    'nl_queries_per_day': 50,
                    'pattern_detection': 'advanced',
                    'news_articles': 10,
                    'export_data': True,
                    'api_access': 'limited',
                    'priority_support': True,
                    'real_time_data': True,
                    'ml_insights': 'basic',
                    'portfolio_tracking': True,
                    'custom_alerts': True,
                    'sector_analysis': True,
                    'peer_comparison': True
                },
                'limitations': {
                    'analysis_depth': 'comprehensive',
                    'advanced_ml': False,
                    'institutional_data': False,
                    'backtesting': 'limited'
                }
            },
            
            'extra_premium': {
                'name': 'Extra Premium',
                'price': 99.99,
                'daily_stock_limit': 100,
                'features': {
                    'basic_analysis': True,
                    'technical_indicators': 'all',
                    'fundamental_metrics': 'all',
                    'chart_access': True,
                    'recommendation_detail': 'comprehensive',
                    'historical_data': '10_years',
                    'alerts': 100,
                    'nl_queries_per_day': 500,
                    'pattern_detection': 'ai_powered',
                    'news_articles': 50,
                    'export_data': True,
                    'api_access': 'full',
                    'priority_support': True,
                    'real_time_data': True,
                    'ml_insights': 'advanced',
                    'portfolio_tracking': True,
                    'custom_alerts': True,
                    'sector_analysis': True,
                    'peer_comparison': True,
                    'advanced_ml': True,
                    'institutional_data': True,
                    'backtesting': 'full',
                    'options_analysis': True,
                    'earnings_prediction': True,
                    'risk_modeling': True,
                    'sentiment_analysis': True,
                    'insider_trading_data': True,
                    'analyst_consensus': True,
                    'custom_screeners': True
                },
                'limitations': {}
            }
        }
    
    def get_user_tier(self, user_id: str) -> str:
        """
        Get user's current tier
        
        Args:
            user_id (str): User identifier
            
        Returns:
            str: User's tier level
        """
        # This would normally check a database
        # For now, defaulting to free tier
        return 'free'
    
    def check_feature_access(self, user_id: str, feature: str) -> bool:
        """
        Check if user has access to a specific feature
        
        Args:
            user_id (str): User identifier
            feature (str): Feature name
            
        Returns:
            bool: True if user has access
        """
        try:
            user_tier = self.get_user_tier(user_id)
            tier_config = self.tier_configs.get(user_tier, self.tier_configs['free'])
            
            return tier_config['features'].get(feature, False)
            
        except Exception as e:
            self.logger.error(f"Error checking feature access: {str(e)}")
            return False
    
    def get_daily_usage(self, user_id: str) -> Dict:
        """
        Get user's daily usage statistics
        
        Args:
            user_id (str): User identifier
            
        Returns:
            Dict: Usage statistics
        """
        # This would normally check a database
        # For now, returning sample data
        return {
            'stocks_analyzed': 0,
            'nl_queries_used': 0,
            'alerts_set': 0,
            'api_calls_made': 0,
            'last_reset': datetime.now().isoformat()
        }
    
    def check_daily_limit(self, user_id: str, resource: str) -> bool:
        """
        Check if user has exceeded daily limits
        
        Args:
            user_id (str): User identifier
            resource (str): Resource type ('stocks', 'nl_queries', etc.)
            
        Returns:
            bool: True if within limits
        """
        try:
            user_tier = self.get_user_tier(user_id)
            tier_config = self.tier_configs.get(user_tier, self.tier_configs['free'])
            usage = self.get_daily_usage(user_id)
            
            limit_mapping = {
                'stocks': ('daily_stock_limit', 'stocks_analyzed'),
                'nl_queries': ('nl_queries_per_day', 'nl_queries_used'),
                'alerts': ('alerts', 'alerts_set')
            }
            
            if resource in limit_mapping:
                limit_key, usage_key = limit_mapping[resource]
                limit = tier_config.get(limit_key, 0)
                used = usage.get(usage_key, 0)
                
                return used < limit
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking daily limit: {str(e)}")
            return False
    
    def filter_analysis_by_tier(self, analysis_data: Dict, user_id: str) -> Dict:
        """
        Filter analysis data based on user's tier
        
        Args:
            analysis_data (Dict): Full analysis data
            user_id (str): User identifier
            
        Returns:
            Dict: Filtered analysis data
        """
        try:
            user_tier = self.get_user_tier(user_id)
            tier_config = self.tier_configs.get(user_tier, self.tier_configs['free'])
            
            filtered_data = {}
            
            # Filter technical indicators
            if 'ta_results' in analysis_data:
                ta_results = analysis_data['ta_results'].copy()
                allowed_indicators = tier_config['features']['technical_indicators']
                
                if allowed_indicators != 'all':
                    # Filter indicators based on tier
                    filtered_indicators = {}
                    for indicator in allowed_indicators:
                        if indicator in ta_results.get('indicators', {}):
                            filtered_indicators[indicator] = ta_results['indicators'][indicator]
                    ta_results['indicators'] = filtered_indicators
                
                filtered_data['ta_results'] = ta_results
            
            # Filter fundamental metrics
            if 'fa_results' in analysis_data:
                fa_results = analysis_data['fa_results'].copy()
                allowed_metrics = tier_config['features']['fundamental_metrics']
                
                if allowed_metrics != 'all' and fa_results:
                    # Filter metrics based on tier
                    filtered_metrics = {}
                    for metric in allowed_metrics:
                        if metric in fa_results.get('metrics', {}):
                            filtered_metrics[metric] = fa_results['metrics'][metric]
                    fa_results['metrics'] = filtered_metrics
                
                filtered_data['fa_results'] = fa_results
            
            # Filter recommendation detail
            if 'recommendation' in analysis_data:
                rec = analysis_data['recommendation'].copy()
                detail_level = tier_config['features']['recommendation_detail']
                
                if detail_level == 'basic':
                    # Only include basic recommendation info
                    filtered_rec = {
                        'action': rec.get('action', 'HOLD'),
                        'final_score': rec.get('final_score', 50),
                        'reasoning': rec.get('reasoning', '')[:200] + '...' if len(rec.get('reasoning', '')) > 200 else rec.get('reasoning', '')
                    }
                    filtered_data['recommendation'] = filtered_rec
                elif detail_level == 'detailed':
                    # Include most recommendation info
                    filtered_rec = {
                        'action': rec.get('action', 'HOLD'),
                        'final_score': rec.get('final_score', 50),
                        'ta_score': rec.get('ta_score', 50),
                        'fa_score': rec.get('fa_score', 50),
                        'reasoning': rec.get('reasoning', ''),
                        'confidence': rec.get('confidence', 'Medium'),
                        'risk_assessment': rec.get('risk_assessment', 'Medium')
                    }
                    filtered_data['recommendation'] = filtered_rec
                else:  # comprehensive
                    filtered_data['recommendation'] = rec
            
            # Filter news articles
            if 'scraped_data' in analysis_data:
                scraped_data = analysis_data['scraped_data'].copy()
                news_limit = tier_config['features']['news_articles']
                
                if 'news' in scraped_data:
                    scraped_data['news'] = scraped_data['news'][:news_limit]
                
                # Filter other scraped data based on tier
                if not tier_config['features'].get('insider_trading_data', False):
                    scraped_data.pop('insider_trading', None)
                
                if not tier_config['features'].get('analyst_consensus', False):
                    scraped_data.pop('analyst_ratings', None)
                
                filtered_data['scraped_data'] = scraped_data
            
            # Add tier-specific metadata
            filtered_data['tier_info'] = {
                'user_tier': user_tier,
                'tier_name': tier_config['name'],
                'upgrade_available': user_tier != 'extra_premium'
            }
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error filtering analysis by tier: {str(e)}")
            return analysis_data
    
    def get_upgrade_suggestions(self, user_id: str, requested_feature: str = None) -> Dict:
        """
        Get upgrade suggestions for user
        
        Args:
            user_id (str): User identifier
            requested_feature (str): Specific feature user wants
            
        Returns:
            Dict: Upgrade suggestions
        """
        try:
            current_tier = self.get_user_tier(user_id)
            current_config = self.tier_configs[current_tier]
            
            suggestions = {
                'current_tier': current_tier,
                'upgrade_options': [],
                'feature_comparison': {}
            }
            
            # Find upgrade options
            tier_order = ['free', 'premium', 'extra_premium']
            current_index = tier_order.index(current_tier)
            
            for tier in tier_order[current_index + 1:]:
                tier_config = self.tier_configs[tier]
                
                upgrade_benefits = []
                
                # Compare features
                for feature, value in tier_config['features'].items():
                    current_value = current_config['features'].get(feature, False)
                    
                    if value != current_value:
                        if isinstance(value, bool) and value:
                            upgrade_benefits.append(f"Access to {feature.replace('_', ' ').title()}")
                        elif isinstance(value, (int, float)) and value > current_value:
                            upgrade_benefits.append(f"Increased {feature.replace('_', ' ')} to {value}")
                        elif isinstance(value, str) and value != current_value:
                            upgrade_benefits.append(f"Enhanced {feature.replace('_', ' ')}: {value}")
                
                suggestions['upgrade_options'].append({
                    'tier': tier,
                    'name': tier_config['name'],
                    'price': tier_config['price'],
                    'benefits': upgrade_benefits[:10]  # Top 10 benefits
                })
            
            # Feature comparison
            if requested_feature:
                for tier, config in self.tier_configs.items():
                    if tier != current_tier:
                        feature_available = config['features'].get(requested_feature, False)
                        suggestions['feature_comparison'][tier] = {
                            'available': feature_available,
                            'price': config['price']
                        }
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting upgrade suggestions: {str(e)}")
            return {'error': str(e)}
    
    def get_tier_features_summary(self, tier: str) -> Dict:
        """
        Get summary of features for a specific tier
        
        Args:
            tier (str): Tier name
            
        Returns:
            Dict: Feature summary
        """
        try:
            if tier not in self.tier_configs:
                return {'error': 'Invalid tier'}
            
            config = self.tier_configs[tier]
            
            return {
                'tier': tier,
                'name': config['name'],
                'price': config['price'],
                'daily_stock_limit': config['daily_stock_limit'],
                'key_features': [
                    f"{len(config['features']['technical_indicators'])} Technical Indicators" if config['features']['technical_indicators'] != 'all' else "All Technical Indicators",
                    f"{len(config['features']['fundamental_metrics'])} Fundamental Metrics" if config['features']['fundamental_metrics'] != 'all' else "All Fundamental Metrics",
                    f"{config['features']['nl_queries_per_day']} Natural Language Queries/Day",
                    f"{config['features']['news_articles']} News Articles per Stock",
                    f"Historical Data: {config['features']['historical_data'].replace('_', ' ').title()}",
                    f"Recommendation Detail: {config['features']['recommendation_detail'].title()}",
                    f"Pattern Detection: {config['features']['pattern_detection'].replace('_', ' ').title()}"
                ],
                'premium_features': [
                    feature.replace('_', ' ').title() 
                    for feature, value in config['features'].items() 
                    if isinstance(value, bool) and value and feature not in ['basic_analysis', 'chart_access']
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting tier features summary: {str(e)}")
            return {'error': str(e)}
    
    def increment_usage(self, user_id: str, resource: str, amount: int = 1):
        """
        Increment user's usage counter
        
        Args:
            user_id (str): User identifier
            resource (str): Resource type
            amount (int): Amount to increment
        """
        try:
            # This would normally update a database
            # For now, just logging
            self.logger.info(f"Usage incremented for {user_id}: {resource} +{amount}")
            
        except Exception as e:
            self.logger.error(f"Error incrementing usage: {str(e)}")
    
    def reset_daily_usage(self, user_id: str):
        """
        Reset user's daily usage counters
        
        Args:
            user_id (str): User identifier
        """
        try:
            # This would normally update a database
            # For now, just logging
            self.logger.info(f"Daily usage reset for {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error resetting daily usage: {str(e)}")

# Global service tier manager instance
service_tier_manager = ServiceTierManager()