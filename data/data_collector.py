"""
Real-time Multi-Source Data Collection System
Collects data from social media, news, and market sources
"""

import asyncio
import aiohttp
import tweepy
import praw
import yfinance as yf
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import os
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import feedparser
import websocket
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    """Standardized data point structure"""
    source: str
    data_type: str  # 'social', 'news', 'market', 'technical'
    content: Dict[str, Any]
    timestamp: datetime
    symbol: Optional[str] = None
    sentiment_score: Optional[float] = None
    metadata: Optional[Dict] = None

class TwitterCollector:
    """Collect Twitter data using Twitter API v2"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.client = tweepy.Client(
            bearer_token=config['twitter']['bearer_token'],
            consumer_key=config['twitter']['api_key'],
            consumer_secret=config['twitter']['api_secret'],
            access_token=config['twitter']['access_token'],
            access_token_secret=config['twitter']['access_token_secret'],
            wait_on_rate_limit=True
        )
        
        # Crypto and stock symbols to track
        self.symbols = config.get('symbols', ['BTC', 'ETH', 'AAPL', 'TSLA', 'SPY'])
        
    async def collect_tweets(self, symbol: str, count: int = 100) -> List[DataPoint]:
        """Collect tweets for a specific symbol"""
        
        try:
            # Search queries for the symbol
            queries = [
                f"${symbol}",
                f"{symbol} price",
                f"{symbol} crypto" if symbol in ['BTC', 'ETH', 'ADA', 'SOL'] else f"{symbol} stock",
                f"{symbol} trading"
            ]
            
            all_tweets = []
            
            for query in queries:
                tweets = tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=query,
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                    max_results=min(count // len(queries), 100)
                ).flatten(limit=count // len(queries))
                
                for tweet in tweets:
                    data_point = DataPoint(
                        source='twitter',
                        data_type='social',
                        content={
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'author_id': tweet.author_id,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'quote_count': tweet.public_metrics['quote_count']
                        },
                        timestamp=tweet.created_at,
                        symbol=symbol,
                        metadata={'query': query}
                    )
                    all_tweets.append(data_point)
            
            logger.info(f"Collected {len(all_tweets)} tweets for {symbol}")
            return all_tweets
            
        except Exception as e:
            logger.error(f"Error collecting tweets for {symbol}: {e}")
            return []
    
    async def stream_tweets(self, symbols: List[str], callback):
        """Stream real-time tweets"""
        
        class TwitterStreamListener(tweepy.StreamingClient):
            def __init__(self, bearer_token, callback_func, target_symbols):
                super().__init__(bearer_token)
                self.callback = callback_func
                self.symbols = target_symbols
            
            def on_tweet(self, tweet):
                # Determine which symbol this tweet is about
                symbol = None
                for sym in self.symbols:
                    if sym.lower() in tweet.text.lower():
                        symbol = sym
                        break
                
                data_point = DataPoint(
                    source='twitter',
                    data_type='social',
                    content={
                        'text': tweet.text,
                        'created_at': datetime.now(),
                        'author_id': tweet.author_id if hasattr(tweet, 'author_id') else None
                    },
                    timestamp=datetime.now(),
                    symbol=symbol
                )
                
                self.callback(data_point)
        
        # Setup streaming
        stream = TwitterStreamListener(
            self.config['twitter']['bearer_token'],
            callback,
            symbols
        )
        
        # Add rules for streaming
        rules = []
        for symbol in symbols:
            rules.append(tweepy.StreamRule(f"${symbol}"))
            rules.append(tweepy.StreamRule(f"{symbol} price"))
        
        stream.add_rules(rules)
        stream.filter(tweet_fields=['created_at', 'author_id'])

class RedditCollector:
    """Collect Reddit data from relevant subreddits"""
    
    def __init__(self, config: Dict):
        self.reddit = praw.Reddit(
            client_id=config['reddit']['client_id'],
            client_secret=config['reddit']['client_secret'],
            user_agent=config['reddit']['user_agent']
        )
        
        # Relevant subreddits
        self.subreddits = [
            'CryptoCurrency', 'Bitcoin', 'ethereum', 'stocks', 'investing',
            'SecurityAnalysis', 'ValueInvesting', 'wallstreetbets', 'CryptoMarkets'
        ]
    
    async def collect_posts(self, symbol: str, count: int = 50) -> List[DataPoint]:
        """Collect Reddit posts mentioning a symbol"""
        
        all_posts = []
        
        try:
            for subreddit_name in self.subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts mentioning the symbol
                for post in subreddit.search(symbol, limit=count // len(self.subreddits)):
                    data_point = DataPoint(
                        source='reddit',
                        data_type='social',
                        content={
                            'title': post.title,
                            'selftext': post.selftext,
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'created_utc': datetime.fromtimestamp(post.created_utc),
                            'subreddit': subreddit_name,
                            'url': post.url
                        },
                        timestamp=datetime.fromtimestamp(post.created_utc),
                        symbol=symbol,
                        metadata={'subreddit': subreddit_name}
                    )
                    all_posts.append(data_point)
            
            logger.info(f"Collected {len(all_posts)} Reddit posts for {symbol}")
            return all_posts
            
        except Exception as e:
            logger.error(f"Error collecting Reddit posts for {symbol}: {e}")
            return []

class NewsCollector:
    """Collect financial news from multiple sources"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://cointelegraph.com/rss',
            'https://www.reuters.com/business/finance/rss'
        ]
        
        # News API key if available
        self.news_api_key = config.get('news_api', {}).get('api_key')
    
    async def collect_news(self, symbol: str, hours_back: int = 24) -> List[DataPoint]:
        """Collect news articles mentioning a symbol"""
        
        all_articles = []
        
        # Collect from RSS feeds
        for source_url in self.news_sources:
            try:
                feed = feedparser.parse(source_url)
                
                for entry in feed.entries:
                    # Check if symbol is mentioned
                    if symbol.lower() in entry.title.lower() or symbol.lower() in entry.summary.lower():
                        
                        # Parse publication date
                        pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                        
                        # Only include recent articles
                        if pub_date > datetime.now() - timedelta(hours=hours_back):
                            data_point = DataPoint(
                                source='news_rss',
                                data_type='news',
                                content={
                                    'title': entry.title,
                                    'summary': entry.summary,
                                    'link': entry.link,
                                    'published': pub_date,
                                    'source_url': source_url
                                },
                                timestamp=pub_date,
                                symbol=symbol,
                                metadata={'feed_source': source_url}
                            )
                            all_articles.append(data_point)
            
            except Exception as e:
                logger.error(f"Error collecting from RSS {source_url}: {e}")
        
        # Collect from News API if available
        if self.news_api_key:
            try:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': symbol,
                    'apiKey': self.news_api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'from': (datetime.now() - timedelta(hours=hours_back)).isoformat()
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        data = await response.json()
                        
                        for article in data.get('articles', []):
                            data_point = DataPoint(
                                source='news_api',
                                data_type='news',
                                content={
                                    'title': article['title'],
                                    'description': article['description'],
                                    'content': article['content'],
                                    'url': article['url'],
                                    'published': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                                    'source_name': article['source']['name']
                                },
                                timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                                symbol=symbol,
                                metadata={'news_source': article['source']['name']}
                            )
                            all_articles.append(data_point)
            
            except Exception as e:
                logger.error(f"Error collecting from News API: {e}")
        
        logger.info(f"Collected {len(all_articles)} news articles for {symbol}")
        return all_articles

class MarketDataCollector:
    """Collect market data from various exchanges and sources"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize crypto exchange
        self.crypto_exchange = ccxt.binance({
            'apiKey': config.get('binance', {}).get('api_key'),
            'secret': config.get('binance', {}).get('secret'),
            'sandbox': config.get('binance', {}).get('sandbox', True)
        })
        
        # Stock symbols mapping
        self.stock_symbols = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'AAPL': 'AAPL',
            'TSLA': 'TSLA',
            'SPY': 'SPY'
        }
    
    async def collect_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 168) -> List[DataPoint]:
        """Collect OHLCV data"""
        
        data_points = []
        
        try:
            if symbol in ['BTC', 'ETH', 'ADA', 'SOL']:  # Crypto
                # Get crypto data
                ohlcv = self.crypto_exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe, limit=limit)
                
                for candle in ohlcv:
                    timestamp, open_price, high, low, close, volume = candle
                    
                    data_point = DataPoint(
                        source='binance',
                        data_type='market',
                        content={
                            'open': open_price,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume,
                            'timeframe': timeframe
                        },
                        timestamp=datetime.fromtimestamp(timestamp / 1000),
                        symbol=symbol,
                        metadata={'data_type': 'ohlcv'}
                    )
                    data_points.append(data_point)
            
            else:  # Stocks
                # Get stock data using yfinance
                ticker = yf.Ticker(self.stock_symbols.get(symbol, symbol))
                
                # Determine period based on timeframe and limit
                if timeframe == '1h':
                    period = f"{min(limit // 24, 30)}d"
                    interval = '1h'
                elif timeframe == '1d':
                    period = f"{min(limit, 365)}d"
                    interval = '1d'
                else:
                    period = "1y"
                    interval = timeframe
                
                hist = ticker.history(period=period, interval=interval)
                
                for index, row in hist.iterrows():
                    data_point = DataPoint(
                        source='yahoo_finance',
                        data_type='market',
                        content={
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'close': row['Close'],
                            'volume': row['Volume'],
                            'timeframe': timeframe
                        },
                        timestamp=index.to_pydatetime(),
                        symbol=symbol,
                        metadata={'data_type': 'ohlcv'}
                    )
                    data_points.append(data_point)
            
            logger.info(f"Collected {len(data_points)} OHLCV data points for {symbol}")
            return data_points
            
        except Exception as e:
            logger.error(f"Error collecting OHLCV data for {symbol}: {e}")
            return []
    
    async def collect_orderbook(self, symbol: str) -> Optional[DataPoint]:
        """Collect order book data for crypto"""
        
        try:
            if symbol in ['BTC', 'ETH', 'ADA', 'SOL']:
                orderbook = self.crypto_exchange.fetch_order_book(f"{symbol}/USDT")
                
                data_point = DataPoint(
                    source='binance',
                    data_type='market',
                    content={
                        'bids': orderbook['bids'][:10],  # Top 10 bids
                        'asks': orderbook['asks'][:10],  # Top 10 asks
                        'timestamp': orderbook['timestamp']
                    },
                    timestamp=datetime.fromtimestamp(orderbook['timestamp'] / 1000),
                    symbol=symbol,
                    metadata={'data_type': 'orderbook'}
                )
                
                return data_point
        
        except Exception as e:
            logger.error(f"Error collecting orderbook for {symbol}: {e}")
            return None
    
    async def collect_market_metrics(self, symbol: str) -> Optional[DataPoint]:
        """Collect additional market metrics"""
        
        try:
            # Get ticker data
            if symbol in ['BTC', 'ETH', 'ADA', 'SOL']:
                ticker = self.crypto_exchange.fetch_ticker(f"{symbol}/USDT")
                
                data_point = DataPoint(
                    source='binance',
                    data_type='market',
                    content={
                        'price': ticker['last'],
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'volume_24h': ticker['quoteVolume'],
                        'change_24h': ticker['change'],
                        'percentage_24h': ticker['percentage'],
                        'high_24h': ticker['high'],
                        'low_24h': ticker['low']
                    },
                    timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                    symbol=symbol,
                    metadata={'data_type': 'ticker'}
                )
                
                return data_point
            
            else:  # Stocks
                ticker = yf.Ticker(self.stock_symbols.get(symbol, symbol))
                info = ticker.info
                
                data_point = DataPoint(
                    source='yahoo_finance',
                    data_type='market',
                    content={
                        'price': info.get('currentPrice'),
                        'volume': info.get('volume'),
                        'market_cap': info.get('marketCap'),
                        'pe_ratio': info.get('trailingPE'),
                        'dividend_yield': info.get('dividendYield'),
                        'beta': info.get('beta')
                    },
                    timestamp=datetime.now(),
                    symbol=symbol,
                    metadata={'data_type': 'fundamentals'}
                )
                
                return data_point
        
        except Exception as e:
            logger.error(f"Error collecting market metrics for {symbol}: {e}")
            return None

class TechnicalIndicatorCalculator:
    """Calculate technical indicators from market data"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return [50.0] * len(prices)  # Default neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean()
        avg_losses = pd.Series(losses).rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50).tolist()
    
    @staticmethod
    def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        if len(prices) < slow:
            return {
                'macd': [0.0] * len(prices),
                'signal': [0.0] * len(prices),
                'histogram': [0.0] * len(prices)
            }
        
        prices_series = pd.Series(prices)
        
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd.fillna(0).tolist(),
            'signal': signal_line.fillna(0).tolist(),
            'histogram': histogram.fillna(0).tolist()
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            avg_price = np.mean(prices)
            return {
                'upper': [avg_price] * len(prices),
                'middle': [avg_price] * len(prices),
                'lower': [avg_price] * len(prices)
            }
        
        prices_series = pd.Series(prices)
        
        middle = prices_series.rolling(window=period).mean()
        std = prices_series.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper.fillna(method='bfill').tolist(),
            'middle': middle.fillna(method='bfill').tolist(),
            'lower': lower.fillna(method='bfill').tolist()
        }

class RealTimeDataCollector:
    """Main data collection orchestrator"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize collectors
        self.twitter_collector = TwitterCollector(self.config)
        self.reddit_collector = RedditCollector(self.config)
        self.news_collector = NewsCollector(self.config)
        self.market_collector = MarketDataCollector(self.config)
        self.technical_calculator = TechnicalIndicatorCalculator()
        
        # Data storage
        self.data_queue = queue.Queue()
        self.data_history = []
        
        # Symbols to track
        self.symbols = self.config.get('symbols', ['BTC', 'ETH', 'AAPL', 'TSLA'])
        
        # Collection intervals (in seconds)
        self.collection_intervals = {
            'social': 300,  # 5 minutes
            'news': 900,    # 15 minutes
            'market': 60,   # 1 minute
            'technical': 300  # 5 minutes
        }
        
        # Running flag
        self.running = False
    
    async def collect_all_data(self, symbol: str) -> Dict[str, List[DataPoint]]:
        """Collect all types of data for a symbol"""
        
        tasks = [
            self.twitter_collector.collect_tweets(symbol, 50),
            self.reddit_collector.collect_posts(symbol, 25),
            self.news_collector.collect_news(symbol, 24),
            self.market_collector.collect_ohlcv(symbol, '1h', 168)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'social_twitter': results[0] if not isinstance(results[0], Exception) else [],
            'social_reddit': results[1] if not isinstance(results[1], Exception) else [],
            'news': results[2] if not isinstance(results[2], Exception) else [],
            'market': results[3] if not isinstance(results[3], Exception) else []
        }
    
    def calculate_technical_indicators(self, market_data: List[DataPoint]) -> List[DataPoint]:
        """Calculate technical indicators from market data"""
        
        if not market_data:
            return []
        
        # Extract prices
        prices = [dp.content['close'] for dp in market_data]
        volumes = [dp.content['volume'] for dp in market_data]
        
        # Calculate indicators
        rsi = self.technical_calculator.calculate_rsi(prices)
        macd = self.technical_calculator.calculate_macd(prices)
        bb = self.technical_calculator.calculate_bollinger_bands(prices)
        
        # Create technical indicator data points
        technical_data = []
        
        for i, dp in enumerate(market_data):
            technical_point = DataPoint(
                source='calculated',
                data_type='technical',
                content={
                    'rsi': rsi[i] if i < len(rsi) else 50.0,
                    'macd': macd['macd'][i] if i < len(macd['macd']) else 0.0,
                    'macd_signal': macd['signal'][i] if i < len(macd['signal']) else 0.0,
                    'macd_histogram': macd['histogram'][i] if i < len(macd['histogram']) else 0.0,
                    'bb_upper': bb['upper'][i] if i < len(bb['upper']) else prices[i],
                    'bb_middle': bb['middle'][i] if i < len(bb['middle']) else prices[i],
                    'bb_lower': bb['lower'][i] if i < len(bb['lower']) else prices[i],
                    'price': prices[i],
                    'volume': volumes[i]
                },
                timestamp=dp.timestamp,
                symbol=dp.symbol,
                metadata={'indicators': ['rsi', 'macd', 'bollinger_bands']}
            )
            technical_data.append(technical_point)
        
        return technical_data
    
    async def start_collection(self):
        """Start the data collection process"""
        
        self.running = True
        logger.info("Starting real-time data collection...")
        
        # Create tasks for each symbol
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self._collect_symbol_data(symbol))
            tasks.append(task)
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
    
    async def _collect_symbol_data(self, symbol: str):
        """Collect data for a specific symbol continuously"""
        
        last_collection = {
            'social': 0,
            'news': 0,
            'market': 0,
            'technical': 0
        }
        
        while self.running:
            current_time = time.time()
            
            try:
                # Check what data needs to be collected
                tasks = []
                
                # Social media data
                if current_time - last_collection['social'] >= self.collection_intervals['social']:
                    tasks.extend([
                        self.twitter_collector.collect_tweets(symbol, 20),
                        self.reddit_collector.collect_posts(symbol, 10)
                    ])
                    last_collection['social'] = current_time
                
                # News data
                if current_time - last_collection['news'] >= self.collection_intervals['news']:
                    tasks.append(self.news_collector.collect_news(symbol, 1))
                    last_collection['news'] = current_time
                
                # Market data
                if current_time - last_collection['market'] >= self.collection_intervals['market']:
                    tasks.extend([
                        self.market_collector.collect_market_metrics(symbol),
                        self.market_collector.collect_orderbook(symbol)
                    ])
                    last_collection['market'] = current_time
                
                # Execute collection tasks
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for result in results:
                        if isinstance(result, list):
                            for data_point in result:
                                self.data_queue.put(data_point)
                        elif isinstance(result, DataPoint):
                            self.data_queue.put(result)
                        elif isinstance(result, Exception):
                            logger.error(f"Collection error: {result}")
                
                # Technical indicators (calculated from recent market data)
                if current_time - last_collection['technical'] >= self.collection_intervals['technical']:
                    recent_market_data = [
                        dp for dp in self.data_history[-168:]  # Last week of hourly data
                        if dp.symbol == symbol and dp.data_type == 'market' and 'close' in dp.content
                    ]
                    
                    if recent_market_data:
                        technical_data = self.calculate_technical_indicators(recent_market_data)
                        for tech_dp in technical_data:
                            self.data_queue.put(tech_dp)
                    
                    last_collection['technical'] = current_time
                
                # Process queue
                self._process_data_queue()
                
                # Sleep for a short interval
                await asyncio.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in data collection loop for {symbol}: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    def _process_data_queue(self):
        """Process data from the queue"""
        
        while not self.data_queue.empty():
            try:
                data_point = self.data_queue.get_nowait()
                self.data_history.append(data_point)
                
                # Maintain history size (keep last 10000 points)
                if len(self.data_history) > 10000:
                    self.data_history = self.data_history[-10000:]
                
                # Log data point
                logger.debug(f"Processed data point: {data_point.source} - {data_point.symbol}")
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing data point: {e}")
    
    def get_recent_data(self, symbol: str, hours_back: int = 24) -> Dict[str, List[DataPoint]]:
        """Get recent data for a symbol"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_data = {
            'social': [],
            'news': [],
            'market': [],
            'technical': []
        }
        
        for dp in self.data_history:
            if dp.symbol == symbol and dp.timestamp >= cutoff_time:
                recent_data[dp.data_type].append(dp)
        
        return recent_data
    
    def stop_collection(self):
        """Stop the data collection process"""
        self.running = False
        logger.info("Stopping data collection...")
    
    def save_data(self, filepath: str):
        """Save collected data to file"""
        
        data_to_save = []
        for dp in self.data_history:
            data_to_save.append({
                'source': dp.source,
                'data_type': dp.data_type,
                'content': dp.content,
                'timestamp': dp.timestamp.isoformat(),
                'symbol': dp.symbol,
                'sentiment_score': dp.sentiment_score,
                'metadata': dp.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        
        logger.info(f"Saved {len(data_to_save)} data points to {filepath}")

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        "symbols": ["BTC", "ETH", "AAPL", "TSLA"],
        "twitter": {
            "bearer_token": "your_bearer_token",
            "api_key": "your_api_key",
            "api_secret": "your_api_secret",
            "access_token": "your_access_token",
            "access_token_secret": "your_access_token_secret"
        },
        "reddit": {
            "client_id": "your_client_id",
            "client_secret": "your_client_secret",
            "user_agent": "MarketPredictor/1.0"
        },
        "news_api": {
            "api_key": "your_news_api_key"
        },
        "binance": {
            "api_key": "your_binance_api_key",
            "secret": "your_binance_secret",
            "sandbox": True
        }
    }
    
    # Save config to file
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize collector
    collector = RealTimeDataCollector('config.json')
    
    # Test data collection
    async def test_collection():
        # Collect data for BTC
        data = await collector.collect_all_data('BTC')
        
        print("Collected data:")
        for data_type, data_points in data.items():
            print(f"{data_type}: {len(data_points)} points")
            if data_points:
                print(f"  Sample: {data_points[0].content}")
        
        # Test technical indicators
        if data['market']:
            technical_data = collector.calculate_technical_indicators(data['market'])
            print(f"Technical indicators: {len(technical_data)} points")
            if technical_data:
                print(f"  Sample: {technical_data[-1].content}")
    
    # Run test
    asyncio.run(test_collection())
    
    print("Data collector initialized successfully!")