"""
Natural Language Interface for AI Stock Analysis Bot
Provides conversational AI capabilities for discussing stock analysis
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import re

# Import OpenAI for natural language processing
try:
    from openai import OpenAI
    import os
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class NaturalLanguageInterface:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversation_history = []
        self.context_cache = {}
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                self.ai_enabled = True
            else:
                self.ai_enabled = False
                self.logger.warning("OpenAI API key not found. Natural language features will be limited.")
        else:
            self.ai_enabled = False
            self.logger.warning("OpenAI not available. Natural language features will be limited.")
    
    def process_natural_language_query(self, query: str, analysis_context: Dict) -> Dict:
        """
        Process natural language query about stock analysis
        
        Args:
            query (str): User's natural language query
            analysis_context (Dict): Current analysis context
            
        Returns:
            Dict: Response with analysis and explanation
        """
        try:
            # Clean and prepare query
            cleaned_query = self.clean_query(query)
            
            # Determine query type
            query_type = self.classify_query(cleaned_query)
            
            # Generate response based on query type
            if self.ai_enabled:
                response = self.generate_ai_response(cleaned_query, analysis_context, query_type)
            else:
                response = self.generate_rule_based_response(cleaned_query, analysis_context, query_type)
            
            # Store conversation
            self.conversation_history.append({
                'query': query,
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'context': analysis_context.get('ticker', 'general')
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing natural language query: {str(e)}")
            return self.generate_error_response(str(e))
    
    def clean_query(self, query: str) -> str:
        """Clean and normalize the user query"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase for processing
        return query.lower()
    
    def classify_query(self, query: str) -> str:
        """
        Classify the type of query
        
        Args:
            query (str): Cleaned query
            
        Returns:
            str: Query type
        """
        # Define query patterns
        patterns = {
            'technical_analysis': [
                'rsi', 'macd', 'moving average', 'technical', 'indicator', 'pattern',
                'support', 'resistance', 'trend', 'bollinger', 'volume'
            ],
            'fundamental_analysis': [
                'pe ratio', 'earnings', 'revenue', 'debt', 'fundamental', 'financial',
                'profit', 'margin', 'growth', 'valuation', 'balance sheet'
            ],
            'recommendation': [
                'buy', 'sell', 'hold', 'recommend', 'should i', 'opinion',
                'advice', 'target price', 'rating'
            ],
            'risk_assessment': [
                'risk', 'safe', 'volatile', 'dangerous', 'stable', 'uncertainty'
            ],
            'comparison': [
                'compare', 'vs', 'versus', 'better than', 'worse than', 'similar'
            ],
            'news_sentiment': [
                'news', 'sentiment', 'market', 'opinion', 'what happened',
                'why', 'recent', 'latest'
            ],
            'explanation': [
                'explain', 'why', 'how', 'what does', 'what is', 'tell me about'
            ]
        }
        
        # Check for pattern matches
        for query_type, keywords in patterns.items():
            for keyword in keywords:
                if keyword in query:
                    return query_type
        
        return 'general'
    
    def generate_ai_response(self, query: str, analysis_context: Dict, query_type: str) -> Dict:
        """Generate AI-powered response using OpenAI"""
        try:
            # Build context for AI
            context_prompt = self.build_context_prompt(analysis_context, query_type)
            
            # Create system prompt
            system_prompt = f"""
            You are an expert stock analysis assistant. You have access to comprehensive stock analysis data.
            
            Context: {context_prompt}
            
            Guidelines:
            - Provide clear, actionable insights
            - Use data from the analysis context
            - Be specific about numbers and metrics
            - Explain technical terms in simple language
            - Give balanced, objective analysis
            - Include relevant warnings about risks
            - Format responses in a conversational but professional tone
            
            Query Type: {query_type}
            """
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for better analysis
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract key insights
            insights = self.extract_insights(ai_response, analysis_context)
            
            return {
                'response': ai_response,
                'insights': insights,
                'query_type': query_type,
                'confidence': 'high',
                'ai_powered': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating AI response: {str(e)}")
            return self.generate_rule_based_response(query, analysis_context, query_type)
    
    def build_context_prompt(self, analysis_context: Dict, query_type: str) -> str:
        """Build context prompt for AI"""
        context_parts = []
        
        if 'ticker' in analysis_context:
            context_parts.append(f"Stock: {analysis_context['ticker']}")
        
        if 'recommendation' in analysis_context:
            rec = analysis_context['recommendation']
            context_parts.append(f"Recommendation: {rec.get('action', 'N/A')}")
            context_parts.append(f"Final Score: {rec.get('final_score', 'N/A')}")
        
        if 'ta_results' in analysis_context:
            ta = analysis_context['ta_results']
            context_parts.append(f"Technical Analysis Score: {ta.get('score', 'N/A')}")
            
            if 'indicators' in ta:
                indicators = ta['indicators']
                if 'RSI' in indicators and not indicators['RSI'].empty:
                    context_parts.append(f"RSI: {indicators['RSI'].iloc[-1]:.2f}")
                if 'MACD' in indicators and not indicators['MACD'].empty:
                    context_parts.append(f"MACD: {indicators['MACD'].iloc[-1]:.2f}")
        
        if 'fa_results' in analysis_context:
            fa = analysis_context['fa_results']
            if fa:
                context_parts.append(f"Fundamental Analysis Score: {fa.get('score', 'N/A')}")
                if 'metrics' in fa:
                    metrics = fa['metrics']
                    if 'pe_ratio' in metrics:
                        context_parts.append(f"P/E Ratio: {metrics['pe_ratio']}")
                    if 'roe' in metrics:
                        context_parts.append(f"ROE: {metrics['roe']:.2%}")
        
        return "; ".join(context_parts)
    
    def generate_rule_based_response(self, query: str, analysis_context: Dict, query_type: str) -> Dict:
        """Generate rule-based response when AI is not available"""
        try:
            response_templates = {
                'technical_analysis': self.handle_technical_query,
                'fundamental_analysis': self.handle_fundamental_query,
                'recommendation': self.handle_recommendation_query,
                'risk_assessment': self.handle_risk_query,
                'explanation': self.handle_explanation_query,
                'general': self.handle_general_query
            }
            
            handler = response_templates.get(query_type, self.handle_general_query)
            response_text = handler(query, analysis_context)
            
            return {
                'response': response_text,
                'insights': [],
                'query_type': query_type,
                'confidence': 'medium',
                'ai_powered': False,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating rule-based response: {str(e)}")
            return self.generate_error_response(str(e))
    
    def handle_technical_query(self, query: str, context: Dict) -> str:
        """Handle technical analysis queries"""
        ticker = context.get('ticker', 'this stock')
        ta_results = context.get('ta_results', {})
        
        if 'rsi' in query:
            indicators = ta_results.get('indicators', {})
            if 'RSI' in indicators and not indicators['RSI'].empty:
                rsi_value = indicators['RSI'].iloc[-1]
                if rsi_value > 70:
                    return f"The RSI for {ticker} is {rsi_value:.1f}, indicating overbought conditions. This suggests the stock may be due for a pullback."
                elif rsi_value < 30:
                    return f"The RSI for {ticker} is {rsi_value:.1f}, indicating oversold conditions. This could present a buying opportunity."
                else:
                    return f"The RSI for {ticker} is {rsi_value:.1f}, which is in the neutral range (30-70)."
        
        if 'macd' in query:
            indicators = ta_results.get('indicators', {})
            if 'MACD' in indicators and not indicators['MACD'].empty:
                macd_value = indicators['MACD'].iloc[-1]
                return f"The MACD for {ticker} is {macd_value:.3f}. This indicator helps identify trend changes and momentum."
        
        score = ta_results.get('score', 0)
        return f"The technical analysis score for {ticker} is {score:.1f}/100. This is based on various indicators including RSI, MACD, and moving averages."
    
    def handle_fundamental_query(self, query: str, context: Dict) -> str:
        """Handle fundamental analysis queries"""
        ticker = context.get('ticker', 'this stock')
        fa_results = context.get('fa_results', {})
        
        if not fa_results:
            return f"Fundamental analysis data is not available for {ticker}."
        
        metrics = fa_results.get('metrics', {})
        
        if 'pe ratio' in query or 'p/e' in query:
            pe_ratio = metrics.get('pe_ratio', 0)
            if pe_ratio > 0:
                if pe_ratio < 15:
                    return f"The P/E ratio for {ticker} is {pe_ratio:.1f}, which suggests the stock may be undervalued."
                elif pe_ratio > 25:
                    return f"The P/E ratio for {ticker} is {pe_ratio:.1f}, which suggests the stock may be overvalued."
                else:
                    return f"The P/E ratio for {ticker} is {pe_ratio:.1f}, which is in a reasonable range."
        
        if 'earnings' in query or 'revenue' in query:
            revenue_growth = metrics.get('revenue_growth', 0)
            if revenue_growth > 0:
                return f"The revenue growth for {ticker} is {revenue_growth:.1%}, indicating positive business expansion."
            else:
                return f"The revenue growth for {ticker} is {revenue_growth:.1%}, which shows declining revenues."
        
        score = fa_results.get('score', 0)
        return f"The fundamental analysis score for {ticker} is {score:.1f}/100, based on financial metrics like P/E ratio, ROE, and growth rates."
    
    def handle_recommendation_query(self, query: str, context: Dict) -> str:
        """Handle recommendation queries"""
        ticker = context.get('ticker', 'this stock')
        recommendation = context.get('recommendation', {})
        
        if not recommendation:
            return f"No recommendation is available for {ticker}."
        
        action = recommendation.get('action', 'HOLD')
        final_score = recommendation.get('final_score', 50)
        
        return f"My recommendation for {ticker} is {action} with a final score of {final_score:.1f}/100. This is based on a combination of technical and fundamental analysis."
    
    def handle_risk_query(self, query: str, context: Dict) -> str:
        """Handle risk assessment queries"""
        ticker = context.get('ticker', 'this stock')
        recommendation = context.get('recommendation', {})
        
        risk_level = recommendation.get('risk_assessment', 'Medium')
        return f"The risk assessment for {ticker} is {risk_level}. This considers factors like volatility, debt levels, and market conditions."
    
    def handle_explanation_query(self, query: str, context: Dict) -> str:
        """Handle explanation queries"""
        ticker = context.get('ticker', 'this stock')
        
        if 'why' in query:
            recommendation = context.get('recommendation', {})
            reasoning = recommendation.get('reasoning', '')
            if reasoning:
                return f"Here's why I made this recommendation for {ticker}: {reasoning[:300]}..."
        
        return f"I can explain various aspects of the analysis for {ticker}. Could you be more specific about what you'd like to understand?"
    
    def handle_general_query(self, query: str, context: Dict) -> str:
        """Handle general queries"""
        ticker = context.get('ticker', 'this stock')
        return f"I can help you understand the analysis for {ticker}. Feel free to ask about technical indicators, fundamental metrics, or my recommendation."
    
    def extract_insights(self, response: str, context: Dict) -> List[str]:
        """Extract key insights from AI response"""
        insights = []
        
        # Extract key phrases and numbers
        import re
        
        # Look for percentage mentions
        percentages = re.findall(r'(\d+(?:\.\d+)?%)', response)
        for pct in percentages:
            insights.append(f"Key metric: {pct}")
        
        # Look for buy/sell/hold mentions
        actions = re.findall(r'\b(buy|sell|hold|bullish|bearish)\b', response.lower())
        for action in set(actions):
            insights.append(f"Sentiment: {action.capitalize()}")
        
        return insights[:5]  # Return top 5 insights
    
    def generate_error_response(self, error_message: str) -> Dict:
        """Generate error response"""
        return {
            'response': f"I apologize, but I encountered an error processing your query: {error_message}. Please try rephrasing your question.",
            'insights': [],
            'query_type': 'error',
            'confidence': 'low',
            'ai_powered': False,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_suggested_questions(self, context: Dict) -> List[str]:
        """Generate suggested questions based on analysis context"""
        ticker = context.get('ticker', 'this stock')
        suggestions = []
        
        # Basic questions
        suggestions.append(f"Why do you recommend {ticker}?")
        suggestions.append(f"What are the main risks for {ticker}?")
        suggestions.append(f"How does {ticker} compare to its peers?")
        
        # Technical analysis questions
        if 'ta_results' in context:
            suggestions.append(f"What do the technical indicators say about {ticker}?")
            suggestions.append(f"Is {ticker} overbought or oversold?")
        
        # Fundamental analysis questions
        if 'fa_results' in context:
            suggestions.append(f"What are the key financial metrics for {ticker}?")
            suggestions.append(f"Is {ticker} undervalued or overvalued?")
        
        # Recent news questions
        if 'scraped_data' in context:
            suggestions.append(f"What's the latest news about {ticker}?")
            suggestions.append(f"What are analysts saying about {ticker}?")
        
        return suggestions[:6]  # Return top 6 suggestions