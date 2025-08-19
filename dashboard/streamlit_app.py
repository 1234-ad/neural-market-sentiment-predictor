"""
Interactive Streamlit Dashboard for Neural Market Sentiment Predictor
Real-time visualization of predictions, sentiment analysis, and market data
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import asyncio
import websocket
import threading
import queue
import logging

# Configure page
st.set_page_config(
    page_title="Neural Market Sentiment Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .sentiment-positive {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .sentiment-negative {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .sentiment-neutral {
        background: linear-gradient(90deg, #bdc3c7 0%, #2c3e50 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000/ws"

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = {}
if 'websocket_connected' not in st.session_state:
    st.session_state.websocket_connected = False

# Helper functions
@st.cache_data(ttl=60)
def get_supported_symbols():
    """Get supported symbols from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/symbols")
        if response.status_code == 200:
            return response.json()
        return {"supported_symbols": {"crypto": ["BTC", "ETH"], "stocks": ["AAPL", "TSLA"]}}
    except:
        return {"supported_symbols": {"crypto": ["BTC", "ETH"], "stocks": ["AAPL", "TSLA"]}}

def make_prediction(symbol, timeframe, current_price=None):
    """Make prediction via API"""
    try:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe
        }
        if current_price:
            payload["current_price"] = current_price
        
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

def analyze_sentiment(texts, sources=None):
    """Analyze sentiment via API"""
    try:
        payload = {"texts": texts}
        if sources:
            payload["sources"] = sources
        
        response = requests.post(f"{API_BASE_URL}/sentiment", json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Sentiment Analysis Error: {e}")
        return None

def get_market_data(symbol, hours_back=24):
    """Get market data via API"""
    try:
        response = requests.get(f"{API_BASE_URL}/market-data/{symbol}?hours_back={hours_back}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Market Data Error: {e}")
        return None

def create_price_chart(predictions_history):
    """Create price prediction chart"""
    if not predictions_history:
        return go.Figure()
    
    df = pd.DataFrame(predictions_history)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Predictions', 'Confidence & Risk'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['current_price'],
            mode='lines+markers',
            name='Current Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['predicted_price'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Confidence and Risk
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['risk_score'],
            mode='lines+markers',
            name='Risk Score',
            line=dict(color='orange', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Market Predictions Over Time",
        height=600,
        showlegend=True
    )
    
    return fig

def create_sentiment_gauge(sentiment_score):
    """Create sentiment gauge chart"""
    
    # Determine color based on sentiment
    if sentiment_score > 0.3:
        color = "green"
    elif sentiment_score < -0.3:
        color = "red"
    else:
        color = "yellow"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Market Sentiment"},
        delta = {'reference': 0},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [-1, -0.5], 'color': "lightcoral"},
                {'range': [-0.5, 0], 'color': "lightyellow"},
                {'range': [0, 0.5], 'color': "lightblue"},
                {'range': [0.5, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(explanation):
    """Create feature importance chart"""
    
    if not explanation:
        return go.Figure()
    
    features = []
    importance = []
    
    for key, value in explanation.items():
        if 'importance' in key:
            features.append(key.replace('_importance', '').title())
            importance.append(value)
    
    if not features:
        return go.Figure()
    
    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=importance,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        )
    ])
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance",
        height=400
    )
    
    return fig

# Main Dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Neural Market Sentiment Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Get supported symbols
    symbols_data = get_supported_symbols()
    all_symbols = []
    for category, symbols in symbols_data["supported_symbols"].items():
        all_symbols.extend(symbols)
    
    # Symbol selection
    selected_symbol = st.sidebar.selectbox(
        "📈 Select Symbol",
        all_symbols,
        index=0
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "⏰ Timeframe",
        ["1h", "4h", "1d", "1w"],
        index=0
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content area
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader(f"📊 {selected_symbol} Analysis")
        
        # Make prediction
        if st.button("🎯 Get Prediction", type="primary"):
            with st.spinner("Making prediction..."):
                prediction = make_prediction(selected_symbol, timeframe)
                
                if prediction:
                    # Add to history
                    prediction['timestamp'] = datetime.now()
                    st.session_state.predictions_history.append(prediction)
                    
                    # Keep only last 50 predictions
                    if len(st.session_state.predictions_history) > 50:
                        st.session_state.predictions_history = st.session_state.predictions_history[-50:]
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🎯 Prediction for {selected_symbol}</h3>
                        <p><strong>Current Price:</strong> ${prediction['current_price']:.2f}</p>
                        <p><strong>Predicted Price:</strong> ${prediction['predicted_price']:.2f}</p>
                        <p><strong>Change:</strong> {prediction['price_change_percent']:.2f}%</p>
                        <p><strong>Direction:</strong> {prediction['direction'].upper()}</p>
                        <p><strong>Confidence:</strong> {prediction['confidence']:.1%}</p>
                        <p><strong>Risk Score:</strong> {prediction['risk_score']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📈 Price Chart")
        
        if st.session_state.predictions_history:
            # Filter for selected symbol
            symbol_predictions = [p for p in st.session_state.predictions_history if p['symbol'] == selected_symbol]
            
            if symbol_predictions:
                chart = create_price_chart(symbol_predictions)
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info(f"No predictions yet for {selected_symbol}")
        else:
            st.info("Make a prediction to see the chart")
    
    with col3:
        st.subheader("📊 Quick Stats")
        
        # Get market data
        market_data = get_market_data(selected_symbol)
        
        if market_data:
            st.metric(
                "Data Points",
                market_data['data_points'],
                delta=None
            )
            
            if market_data['latest_price']:
                st.metric(
                    "Latest Price",
                    f"${market_data['latest_price']:.2f}",
                    delta=None
                )
            
            # Sentiment summary
            sentiment_summary = market_data.get('sentiment_summary', {})
            avg_sentiment = sentiment_summary.get('average_sentiment', 0)
            
            if avg_sentiment > 0.1:
                sentiment_class = "sentiment-positive"
                sentiment_text = "Positive"
            elif avg_sentiment < -0.1:
                sentiment_class = "sentiment-negative"
                sentiment_text = "Negative"
            else:
                sentiment_class = "sentiment-neutral"
                sentiment_text = "Neutral"
            
            st.markdown(f"""
            <div class="{sentiment_class}">
                <strong>Sentiment: {sentiment_text}</strong><br>
                Score: {avg_sentiment:.3f}
            </div>
            """, unsafe_allow_html=True)
    
    # Second row
    st.markdown("---")
    
    col4, col5 = st.columns([1, 1])
    
    with col4:
        st.subheader("💭 Sentiment Analysis")
        
        # Sentiment input
        sentiment_text = st.text_area(
            "Enter text to analyze sentiment:",
            placeholder="Bitcoin is going to the moon! 🚀",
            height=100
        )
        
        if st.button("🔍 Analyze Sentiment"):
            if sentiment_text:
                with st.spinner("Analyzing sentiment..."):
                    sentiment_result = analyze_sentiment([sentiment_text])
                    
                    if sentiment_result:
                        result = sentiment_result['results'][0]
                        
                        # Sentiment gauge
                        gauge_fig = create_sentiment_gauge(result['score'])
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        # Detailed results
                        st.json({
                            "Score": result['score'],
                            "Confidence": result['confidence'],
                            "Magnitude": result['magnitude'],
                            "Raw Scores": result['raw_scores']
                        })
    
    with col5:
        st.subheader("🔍 Feature Importance")
        
        if st.session_state.predictions_history:
            # Get latest prediction for selected symbol
            symbol_predictions = [p for p in st.session_state.predictions_history if p['symbol'] == selected_symbol]
            
            if symbol_predictions:
                latest_prediction = symbol_predictions[-1]
                explanation = latest_prediction.get('explanation', {})
                
                if explanation:
                    importance_chart = create_feature_importance_chart(explanation)
                    st.plotly_chart(importance_chart, use_container_width=True)
                    
                    # Show explanation details
                    st.json(explanation)
                else:
                    st.info("No explanation data available")
            else:
                st.info(f"No predictions for {selected_symbol}")
        else:
            st.info("Make a prediction to see feature importance")
    
    # Third row - Market Data Table
    st.markdown("---")
    st.subheader("📋 Recent Predictions")
    
    if st.session_state.predictions_history:
        # Create DataFrame
        df = pd.DataFrame(st.session_state.predictions_history)
        
        # Format DataFrame
        display_df = df[['symbol', 'current_price', 'predicted_price', 'price_change_percent', 
                        'direction', 'confidence', 'risk_score', 'timeframe']].copy()
        
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
        display_df['predicted_price'] = display_df['predicted_price'].apply(lambda x: f"${x:.2f}")
        display_df['price_change_percent'] = display_df['price_change_percent'].apply(lambda x: f"{x:.2f}%")
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.1%}")
        
        # Rename columns
        display_df.columns = ['Symbol', 'Current Price', 'Predicted Price', 'Change %', 
                             'Direction', 'Confidence', 'Risk', 'Timeframe']
        
        st.dataframe(display_df.tail(10), use_container_width=True)
    else:
        st.info("No predictions made yet")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🧠 Neural Market Sentiment Predictor | Built with Streamlit & FastAPI</p>
        <p>⚠️ This is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# Real-time WebSocket connection (simplified)
def websocket_connection():
    """Handle WebSocket connection for real-time updates"""
    try:
        import websocket
        
        def on_message(ws, message):
            data = json.loads(message)
            if data.get('type') == 'prediction':
                st.session_state.real_time_data = data['data']
        
        def on_error(ws, error):
            st.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            st.session_state.websocket_connected = False
        
        def on_open(ws):
            st.session_state.websocket_connected = True
            # Subscribe to updates
            ws.send(json.dumps({
                'type': 'subscribe',
                'symbols': ['BTC', 'ETH', 'AAPL', 'TSLA']
            }))
        
        ws = websocket.WebSocketApp(
            WEBSOCKET_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        ws.run_forever()
        
    except Exception as e:
        st.error(f"WebSocket connection failed: {e}")

# Sidebar - System Status
with st.sidebar:
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    # Check API health
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ API Connected")
            
            # Show model status
            models = health_data.get('models_loaded', {})
            for model, status in models.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {model.replace('_', ' ').title()}")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Disconnected")
    
    # WebSocket status
    if st.session_state.websocket_connected:
        st.success("✅ Real-time Connected")
    else:
        st.warning("⚠️ Real-time Disconnected")
    
    # Performance metrics
    st.markdown("---")
    st.subheader("📊 Performance")
    
    if st.session_state.predictions_history:
        total_predictions = len(st.session_state.predictions_history)
        st.metric("Total Predictions", total_predictions)
        
        # Calculate accuracy (simplified)
        recent_predictions = st.session_state.predictions_history[-10:]
        if recent_predictions:
            avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")

if __name__ == "__main__":
    main()