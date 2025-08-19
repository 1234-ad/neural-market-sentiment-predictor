"""
Advanced Multi-Modal Market Prediction Model
Combines sentiment, technical indicators, and market data for price prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PredictionResult:
    """Structured prediction result"""
    price_target: float
    confidence: float
    direction: str  # 'up', 'down', 'sideways'
    timeframe: str  # '1h', '4h', '1d', '1w'
    probability_up: float
    probability_down: float
    risk_score: float
    explanation: Dict[str, float]  # Feature importance
    timestamp: datetime

class AttentionFusion(nn.Module):
    """Multi-head attention for fusing different data modalities"""
    
    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None):
        # Apply attention
        attn_output, attn_weights = self.attention(query, key, value, attn_mask=mask)
        
        # Residual connection and normalization
        output = self.norm(query + self.dropout(attn_output))
        
        return output, attn_weights

class SentimentEncoder(nn.Module):
    """Encode sentiment features into dense representation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
        
        # Attention for different sentiment sources
        self.source_attention = nn.MultiheadAttention(output_dim, 4, batch_first=True)
        
    def forward(self, sentiment_features):
        # Encode sentiment features
        encoded = self.encoder(sentiment_features)
        
        # Apply self-attention (assuming batch dimension)
        if len(encoded.shape) == 2:
            encoded = encoded.unsqueeze(1)  # Add sequence dimension
        
        attended, _ = self.source_attention(encoded, encoded, encoded)
        
        return attended.squeeze(1)  # Remove sequence dimension

class TechnicalEncoder(nn.Module):
    """Encode technical indicators using LSTM and attention"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, output_dim: int = 128):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention mechanism for temporal features
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim * 2, 8, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, technical_sequence):
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(technical_sequence)
        
        # Apply temporal attention
        attended, attention_weights = self.temporal_attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Global average pooling with attention weights
        pooled = torch.mean(attended, dim=1)
        
        # Final projection
        output = self.output_proj(pooled)
        
        return output, attention_weights

class MarketRegimeDetector(nn.Module):
    """Detect market regime (bull, bear, sideways, volatile)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, 4)  # 4 regimes
        )
        
    def forward(self, features):
        regime_logits = self.detector(features)
        regime_probs = F.softmax(regime_logits, dim=-1)
        return regime_probs

class MultiModalMarketPredictor(nn.Module):
    """
    Advanced multi-modal market prediction model
    Combines sentiment, technical, and market data
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input dimensions
        self.sentiment_dim = config.get('sentiment_dim', 50)
        self.technical_dim = config.get('technical_dim', 20)
        self.market_dim = config.get('market_dim', 10)
        self.sequence_length = config.get('sequence_length', 168)  # 1 week of hourly data
        
        # Encoding dimensions
        self.encoding_dim = config.get('encoding_dim', 128)
        
        # Initialize encoders
        self.sentiment_encoder = SentimentEncoder(
            self.sentiment_dim, 
            hidden_dim=256, 
            output_dim=self.encoding_dim
        )
        
        self.technical_encoder = TechnicalEncoder(
            self.technical_dim,
            hidden_dim=256,
            num_layers=2,
            output_dim=self.encoding_dim
        )
        
        # Market data encoder (for volume, volatility, etc.)
        self.market_encoder = nn.Sequential(
            nn.Linear(self.market_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, self.encoding_dim),
            nn.Tanh()
        )
        
        # Cross-modal attention fusion
        self.fusion_attention = AttentionFusion(self.encoding_dim, num_heads=8)
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(self.encoding_dim * 3)
        
        # Multi-timeframe prediction heads
        self.prediction_heads = nn.ModuleDict({
            '1h': self._create_prediction_head(),
            '4h': self._create_prediction_head(),
            '1d': self._create_prediction_head(),
            '1w': self._create_prediction_head()
        })
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.encoding_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Risk estimator
        self.risk_estimator = nn.Sequential(
            nn.Linear(self.encoding_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Feature importance calculator
        self.feature_importance = nn.Sequential(
            nn.Linear(self.encoding_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # sentiment, technical, market
            nn.Softmax(dim=-1)
        )
        
        logging.info(f"MultiModalMarketPredictor initialized on {self.device}")
    
    def _create_prediction_head(self):
        """Create a prediction head for specific timeframe"""
        return nn.Sequential(
            nn.Linear(self.encoding_dim * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output: [price_change, direction_prob_up, direction_prob_down]
            nn.Linear(64, 3)
        )
    
    def forward(self, sentiment_features, technical_sequence, market_features, timeframe='1h'):
        """Forward pass through the model"""
        
        # Encode different modalities
        sentiment_encoded = self.sentiment_encoder(sentiment_features)
        technical_encoded, tech_attention = self.technical_encoder(technical_sequence)
        market_encoded = self.market_encoder(market_features)
        
        # Prepare for fusion (add sequence dimension for attention)
        modalities = torch.stack([
            sentiment_encoded,
            technical_encoded,
            market_encoded
        ], dim=1)  # [batch, 3, encoding_dim]
        
        # Cross-modal attention fusion
        fused_features, fusion_attention = self.fusion_attention(
            modalities, modalities, modalities
        )
        
        # Flatten fused features
        fused_flat = fused_features.view(fused_features.size(0), -1)
        
        # Market regime detection
        regime_probs = self.regime_detector(fused_flat)
        
        # Timeframe-specific prediction
        prediction_raw = self.prediction_heads[timeframe](fused_flat)
        
        # Extract components
        price_change = prediction_raw[:, 0]
        direction_logits = prediction_raw[:, 1:3]
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # Confidence and risk estimation
        confidence = self.confidence_estimator(fused_flat).squeeze(-1)
        risk_score = self.risk_estimator(fused_flat).squeeze(-1)
        
        # Feature importance
        importance = self.feature_importance(fused_flat)
        
        return {
            'price_change': price_change,
            'direction_probs': direction_probs,
            'regime_probs': regime_probs,
            'confidence': confidence,
            'risk_score': risk_score,
            'feature_importance': importance,
            'attention_weights': {
                'technical': tech_attention,
                'fusion': fusion_attention
            }
        }
    
    def predict(self, sentiment_features: np.ndarray, 
                technical_sequence: np.ndarray,
                market_features: np.ndarray,
                current_price: float,
                timeframe: str = '1h') -> PredictionResult:
        """Make a prediction and return structured result"""
        
        self.eval()
        
        with torch.no_grad():
            # Convert to tensors
            sentiment_tensor = torch.FloatTensor(sentiment_features).unsqueeze(0).to(self.device)
            technical_tensor = torch.FloatTensor(technical_sequence).unsqueeze(0).to(self.device)
            market_tensor = torch.FloatTensor(market_features).unsqueeze(0).to(self.device)
            
            # Forward pass
            outputs = self.forward(sentiment_tensor, technical_tensor, market_tensor, timeframe)
            
            # Extract predictions
            price_change = outputs['price_change'].item()
            direction_probs = outputs['direction_probs'][0].cpu().numpy()
            confidence = outputs['confidence'].item()
            risk_score = outputs['risk_score'].item()
            importance = outputs['feature_importance'][0].cpu().numpy()
            
            # Calculate target price
            price_target = current_price * (1 + price_change)
            
            # Determine direction
            prob_up, prob_down = direction_probs
            if prob_up > prob_down + 0.1:
                direction = 'up'
            elif prob_down > prob_up + 0.1:
                direction = 'down'
            else:
                direction = 'sideways'
            
            # Create explanation
            explanation = {
                'sentiment_importance': float(importance[0]),
                'technical_importance': float(importance[1]),
                'market_importance': float(importance[2]),
                'price_change_prediction': price_change,
                'regime_analysis': self._interpret_regime(outputs['regime_probs'][0])
            }
            
            return PredictionResult(
                price_target=price_target,
                confidence=confidence,
                direction=direction,
                timeframe=timeframe,
                probability_up=float(prob_up),
                probability_down=float(prob_down),
                risk_score=risk_score,
                explanation=explanation,
                timestamp=datetime.now()
            )
    
    def _interpret_regime(self, regime_probs):
        """Interpret market regime probabilities"""
        regime_names = ['bull', 'bear', 'sideways', 'volatile']
        regime_dict = {name: float(prob) for name, prob in zip(regime_names, regime_probs)}
        dominant_regime = regime_names[torch.argmax(regime_probs).item()]
        
        return {
            'dominant_regime': dominant_regime,
            'regime_probabilities': regime_dict
        }

class AdaptiveLearningSystem:
    """Adaptive learning system for continuous model improvement"""
    
    def __init__(self, model: MultiModalMarketPredictor, config: Dict):
        self.model = model
        self.config = config
        self.device = model.device
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.memory_size = config.get('memory_size', 10000)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10
        )
        
        # Experience replay buffer
        self.experience_buffer = []
        
        # Performance tracking
        self.performance_history = []
        
        # Concept drift detection
        self.drift_detector = ConceptDriftDetector()
        
    def update_model(self, new_data: Dict, actual_outcomes: Dict):
        """Update model with new data and outcomes"""
        
        # Add to experience buffer
        experience = {
            'data': new_data,
            'outcomes': actual_outcomes,
            'timestamp': datetime.now()
        }
        
        self.experience_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.experience_buffer) > self.memory_size:
            self.experience_buffer.pop(0)
        
        # Check for concept drift
        drift_detected = self.drift_detector.detect_drift(
            new_data, actual_outcomes
        )
        
        if drift_detected:
            logging.info("Concept drift detected, adapting model...")
            self._adapt_to_drift()
        
        # Periodic retraining
        if len(self.experience_buffer) % 100 == 0:
            self._retrain_model()
    
    def _adapt_to_drift(self):
        """Adapt model to concept drift"""
        
        # Increase learning rate temporarily
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 2
        
        # Focus on recent data
        recent_data = self.experience_buffer[-1000:]
        self._train_on_batch(recent_data, epochs=5)
        
        # Reset learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 2
    
    def _retrain_model(self):
        """Retrain model on experience buffer"""
        
        if len(self.experience_buffer) < 50:
            return
        
        # Sample from experience buffer
        sample_size = min(500, len(self.experience_buffer))
        sample_indices = np.random.choice(
            len(self.experience_buffer), 
            sample_size, 
            replace=False
        )
        
        sample_data = [self.experience_buffer[i] for i in sample_indices]
        
        # Train on sample
        self._train_on_batch(sample_data, epochs=3)
    
    def _train_on_batch(self, batch_data: List[Dict], epochs: int = 1):
        """Train model on a batch of data"""
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for experience in batch_data:
                # Extract data
                data = experience['data']
                outcomes = experience['outcomes']
                
                # Convert to tensors
                sentiment_tensor = torch.FloatTensor(data['sentiment']).unsqueeze(0).to(self.device)
                technical_tensor = torch.FloatTensor(data['technical']).unsqueeze(0).to(self.device)
                market_tensor = torch.FloatTensor(data['market']).unsqueeze(0).to(self.device)
                
                # Target values
                actual_price_change = torch.FloatTensor([outcomes['price_change']]).to(self.device)
                actual_direction = torch.LongTensor([outcomes['direction']]).to(self.device)
                
                # Forward pass
                outputs = self.model(sentiment_tensor, technical_tensor, market_tensor)
                
                # Calculate losses
                price_loss = F.mse_loss(outputs['price_change'], actual_price_change)
                direction_loss = F.cross_entropy(outputs['direction_probs'], actual_direction)
                
                # Combined loss
                total_loss_item = price_loss + direction_loss
                total_loss += total_loss_item.item()
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_item.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step(total_loss / len(batch_data))
        
        self.model.eval()

class ConceptDriftDetector:
    """Detect concept drift in market data"""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_distribution = None
        self.current_window = []
        
    def detect_drift(self, new_data: Dict, outcomes: Dict) -> bool:
        """Detect if concept drift has occurred"""
        
        # Extract key features for drift detection
        features = self._extract_drift_features(new_data, outcomes)
        
        self.current_window.append(features)
        
        # Maintain window size
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
        
        # Need enough data for comparison
        if len(self.current_window) < self.window_size:
            return False
        
        # Initialize reference distribution
        if self.reference_distribution is None:
            self.reference_distribution = np.array(self.current_window)
            return False
        
        # Statistical test for drift
        current_dist = np.array(self.current_window)
        
        # Kolmogorov-Smirnov test for each feature
        from scipy import stats
        
        drift_detected = False
        for i in range(current_dist.shape[1]):
            statistic, p_value = stats.ks_2samp(
                self.reference_distribution[:, i],
                current_dist[:, i]
            )
            
            if p_value < self.threshold:
                drift_detected = True
                break
        
        # Update reference if drift detected
        if drift_detected:
            self.reference_distribution = current_dist
        
        return drift_detected
    
    def _extract_drift_features(self, data: Dict, outcomes: Dict) -> List[float]:
        """Extract features for drift detection"""
        
        features = []
        
        # Sentiment features
        if 'sentiment' in data:
            sentiment = data['sentiment']
            features.extend([
                np.mean(sentiment),
                np.std(sentiment),
                np.min(sentiment),
                np.max(sentiment)
            ])
        
        # Technical features
        if 'technical' in data:
            technical = data['technical']
            if len(technical.shape) > 1:
                technical = technical[-1]  # Last timestep
            features.extend([
                np.mean(technical),
                np.std(technical)
            ])
        
        # Outcome features
        features.extend([
            outcomes.get('price_change', 0),
            outcomes.get('volatility', 0)
        ])
        
        return features

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'sentiment_dim': 50,
        'technical_dim': 20,
        'market_dim': 10,
        'sequence_length': 168,
        'encoding_dim': 128,
        'learning_rate': 0.001
    }
    
    # Initialize model
    model = MultiModalMarketPredictor(config)
    adaptive_system = AdaptiveLearningSystem(model, config)
    
    # Generate sample data
    batch_size = 32
    sentiment_features = np.random.randn(batch_size, config['sentiment_dim'])
    technical_sequence = np.random.randn(batch_size, config['sequence_length'], config['technical_dim'])
    market_features = np.random.randn(batch_size, config['market_dim'])
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        sentiment_tensor = torch.FloatTensor(sentiment_features)
        technical_tensor = torch.FloatTensor(technical_sequence)
        market_tensor = torch.FloatTensor(market_features)
        
        outputs = model(sentiment_tensor, technical_tensor, market_tensor)
        
        print("Model outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {type(value)}")
    
    # Test prediction
    single_sentiment = sentiment_features[0]
    single_technical = technical_sequence[0]
    single_market = market_features[0]
    current_price = 50000.0
    
    prediction = model.predict(
        single_sentiment,
        single_technical,
        single_market,
        current_price,
        timeframe='1h'
    )
    
    print(f"\nPrediction Result:")
    print(f"Price Target: ${prediction.price_target:.2f}")
    print(f"Direction: {prediction.direction}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Risk Score: {prediction.risk_score:.3f}")
    print(f"Explanation: {prediction.explanation}")
    
    print("\nModel initialized successfully!")