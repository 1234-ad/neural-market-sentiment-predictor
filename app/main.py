"""
FastAPI Application for Neural Market Sentiment Predictor
Provides REST API endpoints for real-time market predictions
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_analyzer import FinancialSentimentAnalyzer, SentimentDataProcessor
from models.market_predictor import MultiModalMarketPredictor, AdaptiveLearningSystem
from data.data_collector import RealTimeDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTC, AAPL)")
    timeframe: str = Field(default="1h", description="Prediction timeframe (1h, 4h, 1d, 1w)")
    current_price: Optional[float] = Field(None, description="Current price (optional, will fetch if not provided)")

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    price_change_percent: float
    direction: str
    confidence: float
    risk_score: float
    timeframe: str
    explanation: Dict[str, float]
    timestamp: datetime

class SentimentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    sources: Optional[List[str]] = Field(None, description="Source types for each text")

class SentimentResponse(BaseModel):
    results: List[Dict]
    aggregated: Dict
    timestamp: datetime

class MarketDataResponse(BaseModel):
    symbol: str
    data_points: int
    latest_price: Optional[float]
    sentiment_summary: Dict
    technical_summary: Dict
    timestamp: datetime

# Initialize FastAPI app
app = FastAPI(
    title="Neural Market Sentiment Predictor",
    description="AI-powered real-time market sentiment analysis and prediction system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
sentiment_analyzer = None
market_predictor = None
data_collector = None
adaptive_system = None
websocket_connections = []

@app.on_event("startup")
async def startup_event():
    """Initialize models and data collector on startup"""
    global sentiment_analyzer, market_predictor, data_collector, adaptive_system
    
    logger.info("Initializing Neural Market Sentiment Predictor...")
    
    try:
        # Initialize sentiment analyzer
        sentiment_config = {
            'model_path': 'ProsusAI/finbert',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        sentiment_analyzer = FinancialSentimentAnalyzer(sentiment_config)
        logger.info("✅ Sentiment analyzer initialized")
        
        # Initialize market predictor
        predictor_config = {
            'sentiment_dim': 50,
            'technical_dim': 20,
            'market_dim': 10,
            'sequence_length': 168,
            'encoding_dim': 128,
            'learning_rate': 0.001
        }
        market_predictor = MultiModalMarketPredictor(predictor_config)
        logger.info("✅ Market predictor initialized")
        
        # Initialize adaptive learning system
        adaptive_system = AdaptiveLearningSystem(market_predictor, predictor_config)
        logger.info("✅ Adaptive learning system initialized")
        
        # Initialize data collector (if config exists)
        config_path = 'config.json'
        if os.path.exists(config_path):
            data_collector = RealTimeDataCollector(config_path)
            # Start data collection in background
            asyncio.create_task(data_collector.start_collection())
            logger.info("✅ Data collector initialized and started")
        else:
            logger.warning("⚠️ Config file not found, data collector not initialized")
        
        logger.info("🚀 Neural Market Sentiment Predictor is ready!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global data_collector
    
    if data_collector:
        data_collector.stop_collection()
        logger.info("Data collection stopped")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": {
            "sentiment_analyzer": sentiment_analyzer is not None,
            "market_predictor": market_predictor is not None,
            "data_collector": data_collector is not None
        }
    }

# Root endpoint with API information
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Market Sentiment Predictor API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #27ae60; font-weight: bold; }
            .path { color: #3498db; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1 class="header">🧠 Neural Market Sentiment Predictor API</h1>
        <p>AI-powered real-time market sentiment analysis and prediction system</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/health</span><br>
            Health check and system status
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="path">/predict</span><br>
            Get market prediction for a symbol
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <span class="path">/sentiment</span><br>
            Analyze sentiment of text data
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <span class="path">/market-data/{symbol}</span><br>
            Get recent market data and analysis
        </div>
        
        <div class="endpoint">
            <span class="method">WebSocket</span> <span class="path">/ws</span><br>
            Real-time predictions and updates
        </div>
        
        <h2>Documentation:</h2>
        <p><a href="/docs">Interactive API Documentation (Swagger)</a></p>
        <p><a href="/redoc">Alternative Documentation (ReDoc)</a></p>
        
        <h2>System Status:</h2>
        <p>🟢 System is operational</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=PredictionResponse)
async def predict_market(request: PredictionRequest):
    """Get market prediction for a symbol"""
    
    if not market_predictor:
        raise HTTPException(status_code=503, detail="Market predictor not initialized")
    
    try:
        symbol = request.symbol.upper()
        timeframe = request.timeframe
        
        # Get current price if not provided
        current_price = request.current_price
        if not current_price:
            if data_collector:
                recent_data = data_collector.get_recent_data(symbol, hours_back=1)
                if recent_data['market']:
                    latest_market = recent_data['market'][-1]
                    current_price = latest_market.content.get('close') or latest_market.content.get('price')
            
            if not current_price:
                # Fallback: use dummy price for demo
                current_price = 50000.0 if symbol == 'BTC' else 100.0
                logger.warning(f"Using fallback price {current_price} for {symbol}")
        
        # Prepare features
        sentiment_features, technical_features, market_features = await _prepare_features(symbol)
        
        # Make prediction
        prediction = market_predictor.predict(
            sentiment_features,
            technical_features,
            market_features,
            current_price,
            timeframe
        )
        
        # Calculate price change percentage
        price_change_percent = ((prediction.price_target - current_price) / current_price) * 100
        
        return PredictionResponse(
            symbol=symbol,
            current_price=current_price,
            predicted_price=prediction.price_target,
            price_change_percent=price_change_percent,
            direction=prediction.direction,
            confidence=prediction.confidence,
            risk_score=prediction.risk_score,
            timeframe=timeframe,
            explanation=prediction.explanation,
            timestamp=prediction.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of text data"""
    
    if not sentiment_analyzer:
        raise HTTPException(status_code=503, detail="Sentiment analyzer not initialized")
    
    try:
        texts = request.texts
        sources = request.sources or ['general'] * len(texts)
        
        # Analyze sentiment
        results = sentiment_analyzer.analyze_batch(texts, sources)
        
        # Convert results to dict format
        result_dicts = []
        for result in results:
            result_dicts.append({
                'score': result.score,
                'confidence': result.confidence,
                'magnitude': result.magnitude,
                'source': result.source,
                'timestamp': result.timestamp,
                'raw_scores': result.raw_scores
            })
        
        # Aggregate results
        aggregated = sentiment_analyzer.aggregate_sentiment(results)
        
        return SentimentResponse(
            results=result_dicts,
            aggregated=aggregated,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")

@app.get("/market-data/{symbol}", response_model=MarketDataResponse)
async def get_market_data(symbol: str, hours_back: int = 24):
    """Get recent market data and analysis for a symbol"""
    
    if not data_collector:
        raise HTTPException(status_code=503, detail="Data collector not initialized")
    
    try:
        symbol = symbol.upper()
        
        # Get recent data
        recent_data = data_collector.get_recent_data(symbol, hours_back)
        
        # Calculate summaries
        sentiment_summary = _calculate_sentiment_summary(recent_data.get('social', []))
        technical_summary = _calculate_technical_summary(recent_data.get('technical', []))
        
        # Get latest price
        latest_price = None
        if recent_data['market']:
            latest_market = recent_data['market'][-1]
            latest_price = latest_market.content.get('close') or latest_market.content.get('price')
        
        total_data_points = sum(len(data_list) for data_list in recent_data.values())
        
        return MarketDataResponse(
            symbol=symbol,
            data_points=total_data_points,
            latest_price=latest_price,
            sentiment_summary=sentiment_summary,
            technical_summary=technical_summary,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=f"Market data error: {str(e)}")

@app.get("/symbols")
async def get_supported_symbols():
    """Get list of supported trading symbols"""
    
    symbols = {
        "crypto": ["BTC", "ETH", "ADA", "SOL", "MATIC", "DOT"],
        "stocks": ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "SPY", "QQQ"],
        "forex": ["EURUSD", "GBPUSD", "USDJPY"],
        "commodities": ["GOLD", "SILVER", "OIL"]
    }
    
    return {
        "supported_symbols": symbols,
        "total_symbols": sum(len(category) for category in symbols.values()),
        "note": "Add more symbols by configuring the data collector"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time predictions"""
    
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'predict':
                symbol = message.get('symbol', 'BTC')
                timeframe = message.get('timeframe', '1h')
                
                try:
                    # Make prediction
                    prediction_request = PredictionRequest(
                        symbol=symbol,
                        timeframe=timeframe
                    )
                    
                    prediction_response = await predict_market(prediction_request)
                    
                    # Send prediction back
                    await websocket.send_text(json.dumps({
                        'type': 'prediction',
                        'data': prediction_response.dict(),
                        'timestamp': datetime.now().isoformat()
                    }, default=str))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': str(e),
                        'timestamp': datetime.now().isoformat()
                    }))
            
            elif message.get('type') == 'subscribe':
                # Subscribe to real-time updates for symbols
                symbols = message.get('symbols', ['BTC'])
                await websocket.send_text(json.dumps({
                    'type': 'subscribed',
                    'symbols': symbols,
                    'message': f'Subscribed to {len(symbols)} symbols'
                }))
    
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info("WebSocket client disconnected")

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """Trigger model training with recent data"""
    
    if not adaptive_system or not data_collector:
        raise HTTPException(status_code=503, detail="Training system not available")
    
    background_tasks.add_task(_background_training)
    
    return {
        "message": "Training started in background",
        "timestamp": datetime.now()
    }

# Helper functions
async def _prepare_features(symbol: str) -> tuple:
    """Prepare features for prediction"""
    
    # Default features if no data available
    sentiment_features = np.random.randn(50)  # Placeholder
    technical_features = np.random.randn(168, 20)  # Placeholder
    market_features = np.random.randn(10)  # Placeholder
    
    if data_collector:
        try:
            # Get recent data
            recent_data = data_collector.get_recent_data(symbol, hours_back=24)
            
            # Process sentiment features
            if recent_data['social']:
                processor = SentimentDataProcessor(sentiment_analyzer)
                social_data = {
                    'twitter': [dp.content for dp in recent_data['social'] if dp.source == 'twitter'],
                    'reddit': [dp.content for dp in recent_data['social'] if dp.source == 'reddit']
                }
                processor.process_social_media_data(social_data)
                sentiment_dict = processor.get_sentiment_features(lookback_hours=24)
                sentiment_features = np.array(list(sentiment_dict.values())[:50])
                
                # Pad or truncate to correct size
                if len(sentiment_features) < 50:
                    sentiment_features = np.pad(sentiment_features, (0, 50 - len(sentiment_features)))
                else:
                    sentiment_features = sentiment_features[:50]
            
            # Process technical features
            if recent_data['technical']:
                technical_data = []
                for dp in recent_data['technical'][-168:]:  # Last week of data
                    features = [
                        dp.content.get('rsi', 50),
                        dp.content.get('macd', 0),
                        dp.content.get('macd_signal', 0),
                        dp.content.get('macd_histogram', 0),
                        dp.content.get('bb_upper', 0),
                        dp.content.get('bb_middle', 0),
                        dp.content.get('bb_lower', 0),
                        dp.content.get('price', 0),
                        dp.content.get('volume', 0),
                        0  # Placeholder for additional features
                    ]
                    technical_data.append(features[:20])  # Ensure 20 features
                
                if technical_data:
                    technical_features = np.array(technical_data)
                    
                    # Ensure correct shape
                    if technical_features.shape[0] < 168:
                        # Pad with last values
                        padding = np.tile(technical_features[-1], (168 - technical_features.shape[0], 1))
                        technical_features = np.vstack([technical_features, padding])
                    else:
                        technical_features = technical_features[-168:]  # Last 168 hours
            
            # Process market features
            if recent_data['market']:
                latest_market = recent_data['market'][-1]
                market_features = np.array([
                    latest_market.content.get('volume', 0),
                    latest_market.content.get('high', 0) - latest_market.content.get('low', 0),  # Range
                    latest_market.content.get('close', 0) - latest_market.content.get('open', 0),  # Change
                    0, 0, 0, 0, 0, 0, 0  # Placeholder features
                ])[:10]
        
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
    
    return sentiment_features, technical_features, market_features

def _calculate_sentiment_summary(social_data: List) -> Dict:
    """Calculate sentiment summary from social data"""
    
    if not social_data:
        return {
            'average_sentiment': 0.0,
            'total_mentions': 0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 0.0
        }
    
    sentiments = [dp.sentiment_score for dp in social_data if dp.sentiment_score is not None]
    
    if not sentiments:
        return {
            'average_sentiment': 0.0,
            'total_mentions': len(social_data),
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 1.0
        }
    
    positive_count = len([s for s in sentiments if s > 0.1])
    negative_count = len([s for s in sentiments if s < -0.1])
    neutral_count = len(sentiments) - positive_count - negative_count
    
    return {
        'average_sentiment': np.mean(sentiments),
        'total_mentions': len(social_data),
        'positive_ratio': positive_count / len(sentiments),
        'negative_ratio': negative_count / len(sentiments),
        'neutral_ratio': neutral_count / len(sentiments)
    }

def _calculate_technical_summary(technical_data: List) -> Dict:
    """Calculate technical analysis summary"""
    
    if not technical_data:
        return {
            'rsi': 50.0,
            'macd_signal': 'neutral',
            'bollinger_position': 'middle',
            'trend': 'sideways'
        }
    
    latest = technical_data[-1].content
    
    # RSI interpretation
    rsi = latest.get('rsi', 50)
    rsi_signal = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
    
    # MACD interpretation
    macd = latest.get('macd', 0)
    macd_signal_line = latest.get('macd_signal', 0)
    macd_signal = 'bullish' if macd > macd_signal_line else 'bearish'
    
    # Bollinger Bands position
    price = latest.get('price', 0)
    bb_upper = latest.get('bb_upper', price)
    bb_lower = latest.get('bb_lower', price)
    bb_middle = latest.get('bb_middle', price)
    
    if price > bb_upper:
        bb_position = 'above_upper'
    elif price < bb_lower:
        bb_position = 'below_lower'
    elif price > bb_middle:
        bb_position = 'upper_half'
    else:
        bb_position = 'lower_half'
    
    return {
        'rsi': rsi,
        'rsi_signal': rsi_signal,
        'macd_signal': macd_signal,
        'bollinger_position': bb_position,
        'latest_price': price
    }

async def _background_training():
    """Background training task"""
    
    try:
        logger.info("Starting background training...")
        
        # Get recent data for training
        if data_collector:
            symbols = ['BTC', 'ETH', 'AAPL', 'TSLA']
            
            for symbol in symbols:
                recent_data = data_collector.get_recent_data(symbol, hours_back=168)  # 1 week
                
                if recent_data['market']:
                    # Prepare training data
                    training_data = {
                        'sentiment': np.random.randn(50),  # Placeholder
                        'technical': np.random.randn(168, 20),  # Placeholder
                        'market': np.random.randn(10)  # Placeholder
                    }
                    
                    # Simulate outcomes (in real implementation, use actual price changes)
                    outcomes = {
                        'price_change': np.random.randn() * 0.05,  # ±5% change
                        'direction': np.random.randint(0, 2),  # 0 or 1
                        'volatility': abs(np.random.randn() * 0.02)  # Volatility
                    }
                    
                    # Update model
                    adaptive_system.update_model(training_data, outcomes)
        
        logger.info("Background training completed")
        
    except Exception as e:
        logger.error(f"Error in background training: {e}")

# Background task to broadcast real-time updates
async def broadcast_updates():
    """Broadcast real-time updates to WebSocket clients"""
    
    while True:
        try:
            if websocket_connections and data_collector:
                # Get latest data for popular symbols
                symbols = ['BTC', 'ETH', 'AAPL', 'TSLA']
                
                for symbol in symbols:
                    recent_data = data_collector.get_recent_data(symbol, hours_back=1)
                    
                    if recent_data['market']:
                        latest_price = recent_data['market'][-1].content.get('close')
                        
                        update_message = {
                            'type': 'price_update',
                            'symbol': symbol,
                            'price': latest_price,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Broadcast to all connected clients
                        disconnected = []
                        for websocket in websocket_connections:
                            try:
                                await websocket.send_text(json.dumps(update_message, default=str))
                            except:
                                disconnected.append(websocket)
                        
                        # Remove disconnected clients
                        for ws in disconnected:
                            websocket_connections.remove(ws)
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in broadcast updates: {e}")
            await asyncio.sleep(60)

# Start background tasks
@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(broadcast_updates())

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )