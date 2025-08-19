"""
Comprehensive Model Training Script
Trains the Neural Market Sentiment Predictor with MLOps integration
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import mlflow
import mlflow.pytorch
import optuna
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_analyzer import FinancialSentimentAnalyzer, SentimentDataProcessor
from models.market_predictor import MultiModalMarketPredictor, AdaptiveLearningSystem
from data.data_collector import RealTimeDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MarketDataset(Dataset):
    """PyTorch Dataset for market prediction data"""
    
    def __init__(self, sentiment_features: np.ndarray, technical_features: np.ndarray, 
                 market_features: np.ndarray, targets: np.ndarray, 
                 scalers: Optional[Dict] = None, fit_scalers: bool = True):
        
        self.sentiment_features = sentiment_features
        self.technical_features = technical_features
        self.market_features = market_features
        self.targets = targets
        
        # Initialize scalers
        if scalers is None:
            self.scalers = {
                'sentiment': StandardScaler(),
                'technical': StandardScaler(),
                'market': StandardScaler(),
                'targets': StandardScaler()
            }
        else:
            self.scalers = scalers
        
        # Fit and transform data
        if fit_scalers:
            self.sentiment_features = self._fit_transform_sentiment()
            self.technical_features = self._fit_transform_technical()
            self.market_features = self._fit_transform_market()
            self.targets = self._fit_transform_targets()
        else:
            self.sentiment_features = self._transform_sentiment()
            self.technical_features = self._transform_technical()
            self.market_features = self._transform_market()
            self.targets = self._transform_targets()
    
    def _fit_transform_sentiment(self):
        """Fit and transform sentiment features"""
        return self.scalers['sentiment'].fit_transform(self.sentiment_features)
    
    def _transform_sentiment(self):
        """Transform sentiment features"""
        return self.scalers['sentiment'].transform(self.sentiment_features)
    
    def _fit_transform_technical(self):
        """Fit and transform technical features"""
        # Reshape for scaling
        original_shape = self.technical_features.shape
        reshaped = self.technical_features.reshape(-1, original_shape[-1])
        scaled = self.scalers['technical'].fit_transform(reshaped)
        return scaled.reshape(original_shape)
    
    def _transform_technical(self):
        """Transform technical features"""
        original_shape = self.technical_features.shape
        reshaped = self.technical_features.reshape(-1, original_shape[-1])
        scaled = self.scalers['technical'].transform(reshaped)
        return scaled.reshape(original_shape)
    
    def _fit_transform_market(self):
        """Fit and transform market features"""
        return self.scalers['market'].fit_transform(self.market_features)
    
    def _transform_market(self):
        """Transform market features"""
        return self.scalers['market'].transform(self.market_features)
    
    def _fit_transform_targets(self):
        """Fit and transform targets"""
        return self.scalers['targets'].fit_transform(self.targets.reshape(-1, 1)).flatten()
    
    def _transform_targets(self):
        """Transform targets"""
        return self.scalers['targets'].transform(self.targets.reshape(-1, 1)).flatten()
    
    def __len__(self):
        return len(self.sentiment_features)
    
    def __getitem__(self, idx):
        return {
            'sentiment': torch.FloatTensor(self.sentiment_features[idx]),
            'technical': torch.FloatTensor(self.technical_features[idx]),
            'market': torch.FloatTensor(self.market_features[idx]),
            'target': torch.FloatTensor([self.targets[idx]])
        }

class ModelTrainer:
    """Comprehensive model trainer with MLOps integration"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
        mlflow.set_experiment(config.get('experiment_name', 'neural_market_predictor'))
        
        # Initialize model
        self.model = MultiModalMarketPredictor(config['model'])
        self.model.to(self.device)
        
        # Training parameters
        self.learning_rate = config['training'].get('learning_rate', 0.001)
        self.batch_size = config['training'].get('batch_size', 32)
        self.epochs = config['training'].get('epochs', 100)
        self.patience = config['training'].get('patience', 10)
        self.min_delta = config['training'].get('min_delta', 0.001)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Loss functions
        self.price_loss_fn = nn.MSELoss()
        self.direction_loss_fn = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"ModelTrainer initialized on {self.device}")
    
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test data"""
        
        logger.info("Preparing training data...")
        
        # Load data
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                raw_data = json.load(f)
        else:
            # Generate synthetic data for demonstration
            logger.warning("Data file not found, generating synthetic data")
            raw_data = self._generate_synthetic_data()
        
        # Process data
        sentiment_features, technical_features, market_features, targets = self._process_raw_data(raw_data)
        
        # Split data
        indices = np.arange(len(sentiment_features))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        # Create datasets
        train_dataset = MarketDataset(
            sentiment_features[train_idx],
            technical_features[train_idx],
            market_features[train_idx],
            targets[train_idx],
            fit_scalers=True
        )
        
        val_dataset = MarketDataset(
            sentiment_features[val_idx],
            technical_features[val_idx],
            market_features[val_idx],
            targets[val_idx],
            scalers=train_dataset.scalers,
            fit_scalers=False
        )
        
        test_dataset = MarketDataset(
            sentiment_features[test_idx],
            technical_features[test_idx],
            market_features[test_idx],
            targets[test_idx],
            scalers=train_dataset.scalers,
            fit_scalers=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _generate_synthetic_data(self, n_samples: int = 10000) -> List[Dict]:
        """Generate synthetic data for demonstration"""
        
        logger.info(f"Generating {n_samples} synthetic data points...")
        
        synthetic_data = []
        
        for i in range(n_samples):
            # Generate synthetic features
            sentiment_features = np.random.randn(50)
            technical_features = np.random.randn(168, 20)  # 1 week of hourly data
            market_features = np.random.randn(10)
            
            # Generate synthetic target (price change)
            # Add some correlation with features
            sentiment_influence = np.mean(sentiment_features) * 0.1
            technical_influence = np.mean(technical_features[-24:]) * 0.05  # Last day
            market_influence = np.mean(market_features) * 0.03
            
            price_change = (
                sentiment_influence + 
                technical_influence + 
                market_influence + 
                np.random.randn() * 0.02  # Noise
            )
            
            synthetic_data.append({
                'sentiment_features': sentiment_features.tolist(),
                'technical_features': technical_features.tolist(),
                'market_features': market_features.tolist(),
                'price_change': price_change,
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat()
            })
        
        return synthetic_data
    
    def _process_raw_data(self, raw_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process raw data into training format"""
        
        sentiment_features = []
        technical_features = []
        market_features = []
        targets = []
        
        for item in raw_data:
            sentiment_features.append(item['sentiment_features'])
            technical_features.append(item['technical_features'])
            market_features.append(item['market_features'])
            targets.append(item['price_change'])
        
        return (
            np.array(sentiment_features),
            np.array(technical_features),
            np.array(market_features),
            np.array(targets)
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move data to device
            sentiment = batch['sentiment'].to(self.device)
            technical = batch['technical'].to(self.device)
            market = batch['market'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            outputs = self.model(sentiment, technical, market)
            
            # Calculate losses
            price_loss = self.price_loss_fn(outputs['price_change'], target.squeeze())
            
            # Direction loss (convert price change to direction)
            direction_target = (target.squeeze() > 0).long()
            direction_loss = self.direction_loss_fn(outputs['direction_probs'], direction_target)
            
            # Combined loss
            total_loss_batch = price_loss + 0.5 * direction_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Metrics
        price_predictions = []
        price_targets = []
        direction_predictions = []
        direction_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                sentiment = batch['sentiment'].to(self.device)
                technical = batch['technical'].to(self.device)
                market = batch['market'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(sentiment, technical, market)
                
                # Calculate losses
                price_loss = self.price_loss_fn(outputs['price_change'], target.squeeze())
                direction_target = (target.squeeze() > 0).long()
                direction_loss = self.direction_loss_fn(outputs['direction_probs'], direction_target)
                
                total_loss_batch = price_loss + 0.5 * direction_loss
                total_loss += total_loss_batch.item()
                num_batches += 1
                
                # Collect predictions for metrics
                price_predictions.extend(outputs['price_change'].cpu().numpy())
                price_targets.extend(target.squeeze().cpu().numpy())
                
                direction_pred = torch.argmax(outputs['direction_probs'], dim=1)
                direction_predictions.extend(direction_pred.cpu().numpy())
                direction_targets.extend(direction_target.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        
        price_mse = mean_squared_error(price_targets, price_predictions)
        price_mae = mean_absolute_error(price_targets, price_predictions)
        direction_accuracy = accuracy_score(direction_targets, direction_predictions)
        
        metrics = {
            'val_loss': avg_loss,
            'price_mse': price_mse,
            'price_mae': price_mae,
            'direction_accuracy': direction_accuracy
        }
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Full training loop"""
        
        logger.info("Starting training...")
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config['model'])
            mlflow.log_params(self.config['training'])
            
            best_model_state = None
            
            for epoch in range(self.epochs):
                logger.info(f"Epoch {epoch+1}/{self.epochs}")
                
                # Train
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss, val_metrics = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                # Update scheduler
                self.scheduler.step(val_loss)
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **val_metrics,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch)
                
                # Early stopping check
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    
                    # Save best model
                    torch.save({
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'config': self.config,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'metrics': val_metrics
                    }, 'best_model.pth')
                    
                else:
                    self.patience_counter += 1
                
                logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"Direction Acc: {val_metrics['direction_accuracy']:.3f}")
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model
            if best_model_state:
                self.model.load_state_dict(best_model_state)
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            # Log training artifacts
            mlflow.log_artifact('training.log')
            mlflow.log_artifact('best_model.pth')
            
            training_summary = {
                'best_val_loss': self.best_val_loss,
                'total_epochs': len(self.train_losses),
                'final_lr': self.optimizer.param_groups[0]['lr']
            }
            
            logger.info("Training completed!")
            return training_summary
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        
        logger.info("Evaluating model on test set...")
        
        self.model.eval()
        
        price_predictions = []
        price_targets = []
        direction_predictions = []
        direction_targets = []
        confidence_scores = []
        risk_scores = []
        
        with torch.no_grad():
            for batch in test_loader:
                sentiment = batch['sentiment'].to(self.device)
                technical = batch['technical'].to(self.device)
                market = batch['market'].to(self.device)
                target = batch['target'].to(self.device)
                
                outputs = self.model(sentiment, technical, market)
                
                price_predictions.extend(outputs['price_change'].cpu().numpy())
                price_targets.extend(target.squeeze().cpu().numpy())
                
                direction_pred = torch.argmax(outputs['direction_probs'], dim=1)
                direction_predictions.extend(direction_pred.cpu().numpy())
                direction_targets.extend((target.squeeze() > 0).long().cpu().numpy())
                
                confidence_scores.extend(outputs['confidence'].cpu().numpy())
                risk_scores.extend(outputs['risk_score'].cpu().numpy())
        
        # Calculate comprehensive metrics
        test_metrics = {
            'price_mse': mean_squared_error(price_targets, price_predictions),
            'price_mae': mean_absolute_error(price_targets, price_predictions),
            'price_rmse': np.sqrt(mean_squared_error(price_targets, price_predictions)),
            'direction_accuracy': accuracy_score(direction_targets, direction_predictions),
            'avg_confidence': np.mean(confidence_scores),
            'avg_risk_score': np.mean(risk_scores)
        }
        
        # Calculate additional metrics
        price_corr = np.corrcoef(price_targets, price_predictions)[0, 1]
        test_metrics['price_correlation'] = price_corr if not np.isnan(price_corr) else 0.0
        
        # Directional accuracy by confidence quartiles
        confidence_quartiles = np.percentile(confidence_scores, [25, 50, 75])
        for i, threshold in enumerate(confidence_quartiles):
            mask = np.array(confidence_scores) >= threshold
            if np.sum(mask) > 0:
                acc = accuracy_score(
                    np.array(direction_targets)[mask],
                    np.array(direction_predictions)[mask]
                )
                test_metrics[f'direction_accuracy_q{i+1}'] = acc
        
        logger.info("Test Evaluation Results:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return test_metrics

def hyperparameter_optimization(config: Dict, data_path: str, n_trials: int = 50):
    """Hyperparameter optimization using Optuna"""
    
    def objective(trial):
        # Suggest hyperparameters
        trial_config = config.copy()
        
        trial_config['model']['encoding_dim'] = trial.suggest_categorical('encoding_dim', [64, 128, 256])
        trial_config['training']['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        trial_config['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        trial_config['training']['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
        
        # Create trainer
        trainer = ModelTrainer(trial_config)
        
        # Prepare data
        train_loader, val_loader, _ = trainer.prepare_data(data_path)
        
        # Train with limited epochs for optimization
        original_epochs = trainer.epochs
        trainer.epochs = 20  # Reduce epochs for faster optimization
        
        try:
            training_summary = trainer.train(train_loader, val_loader)
            return training_summary['best_val_loss']
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('inf')
    
    # Create study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Hyperparameter optimization completed!")
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best value: {study.best_value}")
    
    return study.best_params

def main():
    parser = argparse.ArgumentParser(description='Train Neural Market Sentiment Predictor')
    parser.add_argument('--config', type=str, default='config/training_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--data', type=str, default='data/training_data.json',
                       help='Path to training data file')
    parser.add_argument('--optimize', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of optimization trials')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'sentiment_dim': 50,
                'technical_dim': 20,
                'market_dim': 10,
                'sequence_length': 168,
                'encoding_dim': 128
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 10,
                'min_delta': 0.001,
                'weight_decay': 0.01
            },
            'mlflow_uri': 'http://localhost:5000',
            'experiment_name': 'neural_market_predictor'
        }
        
        # Save default config
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created default configuration at {args.config}")
    
    # Hyperparameter optimization
    if args.optimize:
        logger.info("Starting hyperparameter optimization...")
        best_params = hyperparameter_optimization(config, args.data, args.trials)
        
        # Update config with best parameters
        for key, value in best_params.items():
            if key in config['model']:
                config['model'][key] = value
            elif key in config['training']:
                config['training'][key] = value
        
        # Save optimized config
        optimized_config_path = args.config.replace('.json', '_optimized.json')
        with open(optimized_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Optimized configuration saved to {optimized_config_path}")
    
    # Train model
    logger.info("Starting model training...")
    trainer = ModelTrainer(config)
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(args.data)
    
    # Train
    training_summary = trainer.train(train_loader, val_loader)
    
    # Evaluate
    test_metrics = trainer.evaluate(test_loader)
    
    # Save final results
    results = {
        'training_summary': training_summary,
        'test_metrics': test_metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to training_results.json")

if __name__ == "__main__":
    main()