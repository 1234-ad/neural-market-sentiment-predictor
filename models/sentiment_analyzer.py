"""
Advanced Multi-Modal Sentiment Analysis for Financial Markets
Combines multiple data sources and uses state-of-the-art NLP models
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, pipeline
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

@dataclass
class SentimentResult:
    """Structured sentiment analysis result"""
    score: float  # -1 to 1 (negative to positive)
    confidence: float  # 0 to 1
    magnitude: float  # 0 to 1 (intensity)
    source: str
    timestamp: datetime
    raw_scores: Dict[str, float]

class FinancialSentimentAnalyzer(nn.Module):
    """
    Advanced sentiment analyzer specifically designed for financial text
    Uses ensemble of models for robust predictions
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self._load_models()
        self._setup_financial_vocabulary()
        
        # Weights for different sources
        self.source_weights = {
            'twitter': 0.3,
            'reddit': 0.25,
            'news': 0.35,
            'discord': 0.1
        }
        
        logging.info(f"FinancialSentimentAnalyzer initialized on {self.device}")
    
    def _load_models(self):
        """Load pre-trained models for sentiment analysis"""
        
        # Primary FinBERT model for financial sentiment
        self.finbert_model = AutoModel.from_pretrained('ProsusAI/finbert')
        self.finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        
        # Secondary models for ensemble
        self.roberta_sentiment = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Custom financial vocabulary embeddings
        self.financial_embeddings = nn.Embedding(10000, 768)
        
        # Attention mechanism for multi-source fusion
        self.attention = nn.MultiheadAttention(768, 8, batch_first=True)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # negative, neutral, positive
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _setup_financial_vocabulary(self):
        """Setup financial-specific vocabulary and weights"""
        
        # Financial keywords with sentiment weights
        self.financial_keywords = {
            # Positive indicators
            'bullish': 0.8, 'moon': 0.7, 'pump': 0.6, 'rally': 0.7,
            'breakout': 0.6, 'support': 0.4, 'buy': 0.5, 'hodl': 0.3,
            'diamond_hands': 0.6, 'to_the_moon': 0.8, 'green': 0.4,
            
            # Negative indicators
            'bearish': -0.8, 'dump': -0.7, 'crash': -0.9, 'dip': -0.4,
            'resistance': -0.3, 'sell': -0.5, 'panic': -0.8, 'red': -0.4,
            'paper_hands': -0.6, 'liquidation': -0.9, 'rugpull': -1.0,
            
            # Neutral but important
            'analysis': 0.0, 'technical': 0.0, 'fundamental': 0.0,
            'chart': 0.0, 'pattern': 0.0, 'volume': 0.0
        }
        
        # Market regime keywords
        self.regime_keywords = {
            'bull_market': ['bull', 'bullish', 'uptrend', 'rally'],
            'bear_market': ['bear', 'bearish', 'downtrend', 'crash'],
            'sideways': ['consolidation', 'range', 'sideways', 'flat']
        }
    
    def preprocess_text(self, text: str, source: str = 'general') -> str:
        """Preprocess text for financial sentiment analysis"""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle financial symbols
        text = text.replace('$', ' dollar ')
        text = text.replace('%', ' percent ')
        
        # Handle crypto/stock symbols
        import re
        # Replace $SYMBOL with symbol name
        text = re.sub(r'\$([A-Z]{2,5})', r'\1', text)
        
        # Handle emojis (basic mapping)
        emoji_sentiment = {
            '🚀': 'rocket bullish',
            '📈': 'chart up positive',
            '📉': 'chart down negative',
            '💎': 'diamond hands hold',
            '🌙': 'moon bullish',
            '🔥': 'fire hot trending',
            '💰': 'money profit',
            '😱': 'fear panic',
            '😭': 'crying sad negative'
        }
        
        for emoji, replacement in emoji_sentiment.items():
            text = text.replace(emoji, f' {replacement} ')
        
        return text.strip()
    
    def analyze_single_text(self, text: str, source: str = 'general') -> SentimentResult:
        """Analyze sentiment of a single text"""
        
        # Preprocess text
        processed_text = self.preprocess_text(text, source)
        
        # Get embeddings from FinBERT
        inputs = self.finbert_tokenizer(
            processed_text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            # Get FinBERT embeddings
            finbert_outputs = self.finbert_model(**inputs)
            embeddings = finbert_outputs.last_hidden_state.mean(dim=1)
            
            # Get sentiment scores
            sentiment_logits = self.classifier(embeddings)
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1)
            
            # Get confidence
            confidence = self.confidence_estimator(embeddings).item()
            
            # Convert to sentiment score (-1 to 1)
            neg, neu, pos = sentiment_probs[0].cpu().numpy()
            sentiment_score = pos - neg
            
            # Calculate magnitude (how strong the sentiment is)
            magnitude = max(pos, neg)
            
        # Get additional sentiment from RoBERTa
        roberta_result = self.roberta_sentiment(processed_text)[0]
        roberta_score = self._convert_roberta_score(roberta_result)
        
        # Apply financial keyword weighting
        keyword_adjustment = self._apply_financial_keywords(processed_text)
        
        # Ensemble the results
        final_score = (
            0.6 * sentiment_score + 
            0.3 * roberta_score + 
            0.1 * keyword_adjustment
        )
        
        # Apply source weighting
        source_weight = self.source_weights.get(source, 1.0)
        final_score *= source_weight
        
        # Clamp to [-1, 1]
        final_score = np.clip(final_score, -1, 1)
        
        return SentimentResult(
            score=final_score,
            confidence=confidence,
            magnitude=magnitude,
            source=source,
            timestamp=datetime.now(),
            raw_scores={
                'finbert': sentiment_score,
                'roberta': roberta_score,
                'keywords': keyword_adjustment,
                'ensemble': final_score
            }
        )
    
    def analyze_batch(self, texts: List[str], sources: List[str] = None) -> List[SentimentResult]:
        """Analyze sentiment for a batch of texts"""
        
        if sources is None:
            sources = ['general'] * len(texts)
        
        results = []
        for text, source in zip(texts, sources):
            try:
                result = self.analyze_single_text(text, source)
                results.append(result)
            except Exception as e:
                logging.error(f"Error analyzing text: {e}")
                # Return neutral sentiment on error
                results.append(SentimentResult(
                    score=0.0,
                    confidence=0.0,
                    magnitude=0.0,
                    source=source,
                    timestamp=datetime.now(),
                    raw_scores={'error': str(e)}
                ))
        
        return results
    
    def aggregate_sentiment(self, results: List[SentimentResult], 
                          time_window: timedelta = timedelta(hours=1)) -> Dict:
        """Aggregate sentiment results over time window"""
        
        # Filter results by time window
        cutoff_time = datetime.now() - time_window
        recent_results = [r for r in results if r.timestamp >= cutoff_time]
        
        if not recent_results:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'magnitude': 0.0,
                'count': 0,
                'source_breakdown': {}
            }
        
        # Calculate weighted averages
        total_weight = sum(r.confidence for r in recent_results)
        
        if total_weight == 0:
            weighted_sentiment = np.mean([r.score for r in recent_results])
            weighted_confidence = np.mean([r.confidence for r in recent_results])
        else:
            weighted_sentiment = sum(r.score * r.confidence for r in recent_results) / total_weight
            weighted_confidence = np.mean([r.confidence for r in recent_results])
        
        weighted_magnitude = np.mean([r.magnitude for r in recent_results])
        
        # Source breakdown
        source_breakdown = {}
        for source in set(r.source for r in recent_results):
            source_results = [r for r in recent_results if r.source == source]
            source_breakdown[source] = {
                'sentiment': np.mean([r.score for r in source_results]),
                'confidence': np.mean([r.confidence for r in source_results]),
                'count': len(source_results)
            }
        
        return {
            'overall_sentiment': weighted_sentiment,
            'confidence': weighted_confidence,
            'magnitude': weighted_magnitude,
            'count': len(recent_results),
            'source_breakdown': source_breakdown,
            'time_window': str(time_window)
        }
    
    def _convert_roberta_score(self, roberta_result: Dict) -> float:
        """Convert RoBERTa result to -1 to 1 scale"""
        label = roberta_result['label']
        score = roberta_result['score']
        
        if label == 'LABEL_0':  # Negative
            return -score
        elif label == 'LABEL_1':  # Neutral
            return 0.0
        elif label == 'LABEL_2':  # Positive
            return score
        else:
            return 0.0
    
    def _apply_financial_keywords(self, text: str) -> float:
        """Apply financial keyword sentiment adjustment"""
        
        words = text.lower().split()
        keyword_score = 0.0
        keyword_count = 0
        
        for word in words:
            if word in self.financial_keywords:
                keyword_score += self.financial_keywords[word]
                keyword_count += 1
        
        if keyword_count == 0:
            return 0.0
        
        # Average keyword sentiment, scaled by frequency
        avg_keyword_sentiment = keyword_score / keyword_count
        frequency_weight = min(keyword_count / len(words), 0.5)  # Cap at 50%
        
        return avg_keyword_sentiment * frequency_weight
    
    def get_market_regime_signal(self, recent_sentiments: List[SentimentResult]) -> str:
        """Determine market regime based on sentiment patterns"""
        
        if len(recent_sentiments) < 10:
            return 'insufficient_data'
        
        # Get sentiment scores from last 24 hours
        scores = [r.score for r in recent_sentiments[-100:]]  # Last 100 data points
        
        avg_sentiment = np.mean(scores)
        sentiment_volatility = np.std(scores)
        trend = np.polyfit(range(len(scores)), scores, 1)[0]  # Linear trend
        
        # Regime classification
        if avg_sentiment > 0.3 and trend > 0.01:
            return 'strong_bullish'
        elif avg_sentiment > 0.1 and trend > 0:
            return 'bullish'
        elif avg_sentiment < -0.3 and trend < -0.01:
            return 'strong_bearish'
        elif avg_sentiment < -0.1 and trend < 0:
            return 'bearish'
        elif sentiment_volatility > 0.4:
            return 'volatile'
        else:
            return 'neutral'

class SentimentDataProcessor:
    """Process and store sentiment data for ML pipeline"""
    
    def __init__(self, analyzer: FinancialSentimentAnalyzer):
        self.analyzer = analyzer
        self.sentiment_history = []
    
    def process_social_media_data(self, data: Dict) -> List[SentimentResult]:
        """Process social media data and extract sentiment"""
        
        results = []
        
        # Process Twitter data
        if 'twitter' in data:
            twitter_texts = [tweet['text'] for tweet in data['twitter']]
            twitter_results = self.analyzer.analyze_batch(
                twitter_texts, 
                ['twitter'] * len(twitter_texts)
            )
            results.extend(twitter_results)
        
        # Process Reddit data
        if 'reddit' in data:
            reddit_texts = [post['title'] + ' ' + post.get('selftext', '') 
                           for post in data['reddit']]
            reddit_results = self.analyzer.analyze_batch(
                reddit_texts,
                ['reddit'] * len(reddit_texts)
            )
            results.extend(reddit_results)
        
        # Process news data
        if 'news' in data:
            news_texts = [article['title'] + ' ' + article.get('description', '')
                         for article in data['news']]
            news_results = self.analyzer.analyze_batch(
                news_texts,
                ['news'] * len(news_texts)
            )
            results.extend(news_results)
        
        # Store in history
        self.sentiment_history.extend(results)
        
        # Keep only last 10000 entries to manage memory
        if len(self.sentiment_history) > 10000:
            self.sentiment_history = self.sentiment_history[-10000:]
        
        return results
    
    def get_sentiment_features(self, lookback_hours: int = 24) -> Dict:
        """Extract sentiment features for ML model"""
        
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_sentiments = [s for s in self.sentiment_history if s.timestamp >= cutoff_time]
        
        if not recent_sentiments:
            return self._get_default_features()
        
        scores = [s.score for s in recent_sentiments]
        confidences = [s.confidence for s in recent_sentiments]
        magnitudes = [s.magnitude for s in recent_sentiments]
        
        # Statistical features
        features = {
            'sentiment_mean': np.mean(scores),
            'sentiment_std': np.std(scores),
            'sentiment_min': np.min(scores),
            'sentiment_max': np.max(scores),
            'sentiment_median': np.median(scores),
            'sentiment_skew': self._calculate_skewness(scores),
            'sentiment_kurtosis': self._calculate_kurtosis(scores),
            
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            
            'magnitude_mean': np.mean(magnitudes),
            'magnitude_std': np.std(magnitudes),
            
            'total_mentions': len(recent_sentiments),
            'positive_ratio': len([s for s in scores if s > 0.1]) / len(scores),
            'negative_ratio': len([s for s in scores if s < -0.1]) / len(scores),
            'neutral_ratio': len([s for s in scores if -0.1 <= s <= 0.1]) / len(scores),
        }
        
        # Time-based features
        hourly_sentiment = self._get_hourly_sentiment(recent_sentiments)
        features.update(hourly_sentiment)
        
        # Source-based features
        source_features = self._get_source_features(recent_sentiments)
        features.update(source_features)
        
        # Market regime
        features['market_regime'] = self.analyzer.get_market_regime_signal(recent_sentiments)
        
        return features
    
    def _get_default_features(self) -> Dict:
        """Return default features when no data available"""
        return {f: 0.0 for f in [
            'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max',
            'sentiment_median', 'sentiment_skew', 'sentiment_kurtosis',
            'confidence_mean', 'confidence_std', 'magnitude_mean', 'magnitude_std',
            'total_mentions', 'positive_ratio', 'negative_ratio', 'neutral_ratio'
        ]} | {'market_regime': 'neutral'}
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean([((x - mean) / std) ** 3 for x in data])
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean([((x - mean) / std) ** 4 for x in data]) - 3
    
    def _get_hourly_sentiment(self, sentiments: List[SentimentResult]) -> Dict:
        """Get sentiment features by hour"""
        
        # Group by hour
        hourly_data = {}
        for s in sentiments:
            hour = s.timestamp.hour
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append(s.score)
        
        # Calculate hourly features
        features = {}
        for hour in range(24):
            if hour in hourly_data:
                features[f'sentiment_hour_{hour}'] = np.mean(hourly_data[hour])
            else:
                features[f'sentiment_hour_{hour}'] = 0.0
        
        return features
    
    def _get_source_features(self, sentiments: List[SentimentResult]) -> Dict:
        """Get sentiment features by source"""
        
        source_data = {}
        for s in sentiments:
            if s.source not in source_data:
                source_data[s.source] = []
            source_data[s.source].append(s.score)
        
        features = {}
        for source in ['twitter', 'reddit', 'news', 'discord']:
            if source in source_data:
                features[f'sentiment_{source}'] = np.mean(source_data[source])
                features[f'mentions_{source}'] = len(source_data[source])
            else:
                features[f'sentiment_{source}'] = 0.0
                features[f'mentions_{source}'] = 0
        
        return features

# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    config = {
        'model_path': 'ProsusAI/finbert',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    analyzer = FinancialSentimentAnalyzer(config)
    processor = SentimentDataProcessor(analyzer)
    
    # Test with sample financial texts
    test_texts = [
        "Bitcoin is going to the moon! 🚀 This bull run is just getting started",
        "Market crash incoming, everyone panic selling their positions",
        "Technical analysis shows strong support at $50k level",
        "HODL diamond hands, this dip is just a buying opportunity 💎",
        "Bearish divergence on the daily chart, expecting correction"
    ]
    
    # Analyze sentiments
    results = analyzer.analyze_batch(test_texts, ['twitter'] * len(test_texts))
    
    # Print results
    for text, result in zip(test_texts, results):
        print(f"Text: {text[:50]}...")
        print(f"Sentiment: {result.score:.3f} (Confidence: {result.confidence:.3f})")
        print(f"Raw scores: {result.raw_scores}")
        print("-" * 50)
    
    # Test aggregation
    aggregated = analyzer.aggregate_sentiment(results)
    print(f"Aggregated sentiment: {aggregated}")