"""
Machine Learning Engine for AI Stock Analysis Bot
Implements ML models for pattern recognition and recommendation improvement
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import logging
from datetime import datetime, timedelta
import os

class MLEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pattern_classifier = None
        self.score_predictor = None
        self.scaler = StandardScaler()
        self.model_dir = "ml_models"
        self.ensure_model_dir()
        
    def ensure_model_dir(self):
        """Create model directory if it doesn't exist"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def prepare_features(self, data, indicators):
        """
        Prepare features for ML models
        
        Args:
            data (pd.DataFrame): Stock price data
            indicators (dict): Technical indicators
            
        Returns:
            pd.DataFrame: Features for ML
        """
        try:
            features = []
            
            # Price-based features
            if not data.empty:
                features.extend([
                    data['Close'].iloc[-1],  # Current price
                    data['Volume'].iloc[-1],  # Current volume
                    data['Close'].pct_change().iloc[-1],  # Daily return
                    data['Close'].pct_change().std(),  # Volatility
                    (data['Close'].iloc[-1] - data['Low'].min()) / (data['High'].max() - data['Low'].min()),  # Position in range
                ])
            
            # Technical indicator features
            for indicator_name, values in indicators.items():
                if isinstance(values, pd.Series) and not values.empty:
                    features.extend([
                        values.iloc[-1],  # Latest value
                        values.mean(),    # Average
                        values.std(),     # Volatility
                    ])
            
            # Ensure we have consistent feature count
            while len(features) < 50:
                features.append(0.0)
            
            return np.array(features[:50])  # Limit to 50 features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return np.zeros(50)
    
    def train_pattern_classifier(self, training_data):
        """
        Train pattern classification model
        
        Args:
            training_data (list): List of training examples
        """
        try:
            if len(training_data) < 10:
                self.logger.warning("Not enough training data for pattern classifier")
                return
            
            X = []
            y = []
            
            for example in training_data:
                features = self.prepare_features(example['data'], example['indicators'])
                X.append(features)
                y.append(len(example['patterns']))  # Number of patterns detected
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.pattern_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.pattern_classifier.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.pattern_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            joblib.dump(self.pattern_classifier, f"{self.model_dir}/pattern_classifier.pkl")
            joblib.dump(self.scaler, f"{self.model_dir}/scaler.pkl")
            
            self.logger.info(f"Pattern classifier trained with accuracy: {accuracy:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error training pattern classifier: {str(e)}")
    
    def train_score_predictor(self, training_data):
        """
        Train score prediction model
        
        Args:
            training_data (list): List of training examples with actual outcomes
        """
        try:
            if len(training_data) < 10:
                self.logger.warning("Not enough training data for score predictor")
                return
            
            X = []
            y = []
            
            for example in training_data:
                features = self.prepare_features(example['data'], example['indicators'])
                X.append(features)
                y.append(example['actual_score'])  # Actual performance score
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.score_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.score_predictor.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.score_predictor.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            
            # Save model
            joblib.dump(self.score_predictor, f"{self.model_dir}/score_predictor.pkl")
            
            self.logger.info(f"Score predictor trained with MSE: {mse:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error training score predictor: {str(e)}")
    
    def load_models(self):
        """Load trained models"""
        try:
            pattern_path = f"{self.model_dir}/pattern_classifier.pkl"
            score_path = f"{self.model_dir}/score_predictor.pkl"
            scaler_path = f"{self.model_dir}/scaler.pkl"
            
            if os.path.exists(pattern_path):
                self.pattern_classifier = joblib.load(pattern_path)
            
            if os.path.exists(score_path):
                self.score_predictor = joblib.load(score_path)
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.logger.info("ML models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
    
    def predict_patterns(self, data, indicators):
        """
        Predict number of patterns using ML
        
        Args:
            data (pd.DataFrame): Stock data
            indicators (dict): Technical indicators
            
        Returns:
            int: Predicted number of patterns
        """
        try:
            if self.pattern_classifier is None:
                return 0
            
            features = self.prepare_features(data, indicators)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            prediction = self.pattern_classifier.predict(features_scaled)[0]
            return int(prediction)
            
        except Exception as e:
            self.logger.error(f"Error predicting patterns: {str(e)}")
            return 0
    
    def predict_score(self, data, indicators):
        """
        Predict stock score using ML
        
        Args:
            data (pd.DataFrame): Stock data
            indicators (dict): Technical indicators
            
        Returns:
            float: Predicted score
        """
        try:
            if self.score_predictor is None:
                return 50.0
            
            features = self.prepare_features(data, indicators)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            prediction = self.score_predictor.predict(features_scaled)[0]
            return max(0, min(100, prediction))  # Clamp to 0-100
            
        except Exception as e:
            self.logger.error(f"Error predicting score: {str(e)}")
            return 50.0
    
    def get_feature_importance(self):
        """Get feature importance from trained models"""
        try:
            importance = {}
            
            if self.pattern_classifier is not None:
                importance['pattern_classifier'] = self.pattern_classifier.feature_importances_
            
            if self.score_predictor is not None:
                importance['score_predictor'] = self.score_predictor.feature_importances_
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def update_training_data(self, ticker, data, indicators, patterns, actual_performance):
        """
        Update training data with new examples
        
        Args:
            ticker (str): Stock ticker
            data (pd.DataFrame): Stock data
            indicators (dict): Technical indicators
            patterns (list): Detected patterns
            actual_performance (float): Actual performance score
        """
        try:
            training_example = {
                'ticker': ticker,
                'data': data,
                'indicators': indicators,
                'patterns': patterns,
                'actual_score': actual_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to training data file
            training_file = f"{self.model_dir}/training_data.csv"
            
            # This is a simplified version - in production, you'd use a proper database
            self.logger.info(f"Updated training data for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error updating training data: {str(e)}")
    
    def continuous_learning(self, db):
        """
        Implement continuous learning by retraining models periodically
        
        Args:
            db: Database instance
        """
        try:
            # This would fetch recent performance data and retrain models
            # For now, it's a placeholder for the continuous learning process
            self.logger.info("Continuous learning cycle initiated")
            
        except Exception as e:
            self.logger.error(f"Error in continuous learning: {str(e)}")