"""
Safe version of the training pipeline for cryptocurrency forecasting models.
This version avoids deep learning models that might cause segmentation faults.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Suppress warnings and optimize for CPU
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'  # Use 4 CPU threads (adjust based on your CPU)
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # For Mac compatibility

# Darts imports - better models for crypto
from darts import TimeSeries
from darts.models import (
    LightGBMModel, ExponentialSmoothing, NBEATSModel,
    RandomForestModel, XGBModel, RNNModel, TiDEModel, TFTModel
)
from darts.metrics import mape, rmse, mae

# Local imports
from utils.data_loader import (
    get_latest_crypto_data, 
    prepare_darts_timeseries, 
    train_test_split_series,
    validate_data_quality
)


class SafeCryptoForecaster:
    """
    A safe cryptocurrency forecasting system using stable models only.
    """
    
    def __init__(self, crypto_ticker: str = 'BTC-USD', forecast_horizon: int = 7):
        """
        Initialize the SafeCryptoForecaster.
        
        Args:
            crypto_ticker: Cryptocurrency ticker symbol (default: 'BTC-USD')
            forecast_horizon: Number of days to forecast ahead (default: 7)
        """
        self.crypto_ticker = crypto_ticker
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.data = None
        self.train_series = None
        self.test_series = None
        self.predictions = {}
        self.evaluation_results = {}
        
        # Create directories
        self.data_dir = Path('data')
        self.models_dir = Path('models')
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        print("="*60)
        print(f"SafeCryptoForecaster initialized for {crypto_ticker}")
        print(f"Forecast horizon: {forecast_horizon} days")
        print("="*60)
    
    def load_and_prepare_data(self, start_date: str = '2020-01-01') -> None:
        """
        Load and prepare cryptocurrency data for training.
        
        Args:
            start_date: Start date for data loading (default: '2020-01-01')
        """
        print("\n📊 Loading and preparing data...")
        
        try:
            # Calculate days back from start_date to today
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            days_back = (datetime.now() - start_dt).days
            
            # Fetch data
            self.data = get_latest_crypto_data(self.crypto_ticker, days_back=days_back)
            
            # Validate data quality
            if not validate_data_quality(self.data, min_rows=200):
                raise ValueError("Data quality check failed")
            
            # Save processed data
            data_file = self.data_dir / f"{self.crypto_ticker.replace('-', '_')}_processed.csv"
            self.data.to_csv(data_file)
            
            # Prepare features for modeling (fewer features for stability)
            value_cols = ['Close', 'Volume', 'RSI', 'MA_7', 'MA_30']
            
            # Create TimeSeries object
            full_series = prepare_darts_timeseries(self.data, value_cols)
            
            # Split into train/test
            self.train_series, self.test_series = train_test_split_series(
                full_series, train_ratio=0.85
            )
            
            print(f"✅ Data loaded successfully!")
            print(f"   Total samples: {len(self.data)}")
            print(f"   Date range: {self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}")
            print(f"   Train samples: {len(self.train_series)}")
            print(f"   Test samples: {len(self.test_series)}")
            print(f"   Features: {len(value_cols)}")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def train_lightgbm_model(self) -> None:
        """Train LightGBM model for cryptocurrency forecasting."""
        print("\n🚀 Training LightGBM model...")
        
        try:
            # Use more conservative settings to avoid crashes
            model = LightGBMModel(
                lags=7,  # Reduced from 14 for stability
                output_chunk_length=self.forecast_horizon,
                random_state=42,
                verbose=-1,
                n_jobs=1  # Single thread to avoid conflicts
            )
            
            # Train the model
            print("   Fitting LightGBM model...")
            print("   \u23f3 Training in progress (this should be quick)...")
            start_time = time.time()
            model.fit(self.train_series)
            end_time = time.time()
            print(f"   \u2705 LightGBM training completed in {end_time - start_time:.1f} seconds")
            
            # Make predictions on test set
            print("   Making predictions...")
            predictions = model.predict(n=len(self.test_series))
            
            # Extract only Close price component to match evaluation
            predictions = predictions.univariate_component('Close')
            
            # Store model and predictions
            self.models['lightgbm'] = model
            self.predictions['lightgbm'] = predictions
            
            print("✅ LightGBM model trained successfully!")
            
        except Exception as e:
            print(f"❌ Error training LightGBM model: {str(e)}")
            self.models['lightgbm'] = None
            self.predictions['lightgbm'] = None
    
    def train_exponential_smoothing_model(self) -> None:
        """Train Exponential Smoothing model as a statistical baseline."""
        print("\n📈 Training Exponential Smoothing model...")
        
        try:
            # Use only the close price for exponential smoothing
            close_series = self.train_series.univariate_component('Close')
            
            # Try simpler parameters to avoid errors
            model = ExponentialSmoothing()
            
            # Train the model
            print("   Fitting Exponential Smoothing model...")
            print("   ⏳ Training in progress (statistical model)...")
            start_time = time.time()
            model.fit(close_series)
            end_time = time.time()
            print(f"   ✅ Exponential Smoothing completed in {end_time - start_time:.1f} seconds")
            
            # Make predictions on test set
            print("   Making predictions...")
            predictions = model.predict(n=len(self.test_series))
            
            # Store model and predictions
            self.models['exponential_smoothing'] = model
            self.predictions['exponential_smoothing'] = predictions
            
            print("✅ Exponential Smoothing model trained successfully!")
            
        except Exception as e:
            print(f"❌ Error training Exponential Smoothing model: {str(e)}")
            self.models['exponential_smoothing'] = None
            self.predictions['exponential_smoothing'] = None
    
    def train_nbeats_model(self) -> None:
        """Train N-BEATS deep learning model with careful error handling."""
        print("\n🧠 Training N-BEATS model...")
        
        try:
            # Use only close price for N-BEATS to avoid complexity
            close_series = self.train_series.univariate_component('Close')
            
            model = NBEATSModel(
                input_chunk_length=14,  # Reduced from 30 for faster training
                output_chunk_length=self.forecast_horizon,
                n_epochs=50,  # Much fewer epochs for speed (was 100)
                batch_size=64,  # Larger batch size for CPU efficiency
                num_blocks=2,  # Reduce model complexity
                num_layers=2,  # Reduce model complexity
                layer_widths=64,  # Smaller layer width
                random_state=42,
                pl_trainer_kwargs={
                    "accelerator": "cpu",
                    "enable_progress_bar": True,  # Show progress for feedback
                    "enable_model_summary": False,
                    "logger": False,
                    "enable_checkpointing": False,
                    "max_epochs": 50,  # Match n_epochs
                    "fast_dev_run": False,
                    "num_sanity_val_steps": 0  # Skip validation sanity checks
                },
                model_name="nbeats_crypto_fast",
                force_reset=True,
                save_checkpoints=False
            )
            
            # Train the model with proper error handling
            print("   Fitting N-BEATS model (this may take a few minutes)...")
            model.fit(close_series, verbose=False)
            
            # Make predictions on test set
            print("   Making predictions...")
            predictions = model.predict(n=len(self.test_series))
            
            # Store model and predictions
            self.models['nbeats'] = model
            self.predictions['nbeats'] = predictions
            
            print("✅ N-BEATS model trained successfully!")
            
        except Exception as e:
            print(f"❌ Error training N-BEATS model: {str(e)}")
            print("   This might be due to environment issues. Continuing without N-BEATS...")
            self.models['nbeats'] = None
            self.predictions['nbeats'] = None
    
    def train_lstm_model(self) -> None:
        """Train LSTM/GRU model - excellent for volatile crypto data."""
        print("\n🔗 Training LSTM model...")
        
        try:
            # Use close price for LSTM
            close_series = self.train_series.univariate_component('Close')
            
            model = RNNModel(
                model='LSTM',  # Can also use 'GRU' or 'RNN'
                input_chunk_length=21,  # 3 weeks of data
                output_chunk_length=self.forecast_horizon,
                hidden_dim=64,  # Good balance of complexity
                n_rnn_layers=2,  # 2 layers for better learning
                dropout=0.2,  # Prevent overfitting
                batch_size=32,
                n_epochs=50,  # Good for crypto volatility
                optimizer_kwargs={'lr': 0.001},
                random_state=42,
                pl_trainer_kwargs={
                    "accelerator": "cpu",
                    "enable_progress_bar": True,
                    "enable_model_summary": True,
                    "logger": True,
                    "enable_checkpointing": False,
                    "max_epochs": 50,
                    "log_every_n_steps": 10
                },
                model_name="lstm_crypto",
                force_reset=True,
                save_checkpoints=False
            )
            
            # Train the model
            print("   Fitting LSTM model (optimized for crypto volatility)...")
            print("   Progress will be shown below (50 epochs):")
            start_time = time.time()
            model.fit(close_series, verbose=True)
            end_time = time.time()
            print(f"   ✅ LSTM training completed in {end_time - start_time:.1f} seconds")
            
            # Make predictions
            print("   Making predictions...")
            predictions = model.predict(n=len(self.test_series))
            
            # Store model and predictions
            self.models['lstm'] = model
            self.predictions['lstm'] = predictions
            
            print("✅ LSTM model trained successfully!")
            
        except Exception as e:
            print(f"❌ Error training LSTM model: {str(e)}")
            self.models['lstm'] = None
            self.predictions['lstm'] = None
    
    def train_xgboost_model(self) -> None:
        """Train XGBoost model - excellent for financial time series."""
        print("\n🚀 Training XGBoost model...")
        
        try:
            # Use only close price to avoid multivariate complexity
            close_series = self.train_series.univariate_component('Close')
            
            model = XGBModel(
                lags=14,  # Reduced from 21 for faster training
                output_chunk_length=self.forecast_horizon,
                random_state=42,
                n_estimators=100,  # Reduced from 200 for speed
                max_depth=4,  # Reduced from 6 for speed
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=1  # Single thread to avoid hanging
            )
            
            # Train the model
            print("   Fitting XGBoost model...")
            print("   ⏳ Training 100 trees with 4 max depth (this may take 2-5 minutes)...")
            start_time = time.time()
            
            # Add timeout protection
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("XGBoost training timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 minute timeout
            
            try:
                model.fit(close_series)
                signal.alarm(0)  # Cancel timeout
            except TimeoutError:
                print("   XGBoost training timed out, skipping...")
                raise
            
            end_time = time.time()
            print(f"   Training completed in {end_time - start_time:.1f} seconds")
            
            # Make predictions
            print("   Making predictions...")
            predictions = model.predict(n=len(self.test_series))
            
            # Store model and predictions
            self.models['xgboost'] = model
            self.predictions['xgboost'] = predictions
            
            print("✅ XGBoost model trained successfully!")
            
        except Exception as e:
            print(f"❌ Error training XGBoost model: {str(e)}")
            print("   Continuing without XGBoost model...")
            self.models['xgboost'] = None
            self.predictions['xgboost'] = None
    
    def train_transformer_model(self) -> None:
        """Train TiDE (Transformer) model - modern approach for time series."""
        print("\n🔮 Training TiDE Transformer model...")
        
        try:
            # Use close price for transformer
            close_series = self.train_series.univariate_component('Close')
            
            model = TiDEModel(
                input_chunk_length=28,  # 4 weeks of data
                output_chunk_length=self.forecast_horizon,
                num_encoder_layers=2,  # Moderate complexity
                num_decoder_layers=2,
                decoder_output_dim=16,
                hidden_dim=128,
                temporal_width_past=4,
                temporal_width_future=2,
                n_epochs=30,  # Reasonable for transformers
                batch_size=32,
                dropout=0.1,
                random_state=42,
                pl_trainer_kwargs={
                    "accelerator": "cpu",
                    "enable_progress_bar": True,
                    "enable_model_summary": True,
                    "logger": True,
                    "enable_checkpointing": False,
                    "max_epochs": 30,
                    "log_every_n_steps": 5
                },
                model_name="tide_crypto",
                force_reset=True,
                save_checkpoints=False
            )
            
            # Train the model
            print("   Fitting TiDE model (modern transformer approach)...")
            print("   Training with 30 epochs, progress will be shown below:")
            start_time = time.time()
            model.fit(close_series, verbose=True)
            end_time = time.time()
            print(f"   ✅ TiDE training completed in {end_time - start_time:.1f} seconds")
            
            # Make predictions
            print("   Making predictions...")
            predictions = model.predict(n=len(self.test_series))
            
            # Store model and predictions
            self.models['transformer'] = model
            self.predictions['transformer'] = predictions
            
            print("✅ TiDE Transformer model trained successfully!")
            
        except Exception as e:
            print(f"❌ Error training TiDE model: {str(e)}")
            print("   Continuing without transformer model...")
            self.models['transformer'] = None
            self.predictions['transformer'] = None
    
    def evaluate_models(self) -> None:
        """Evaluate all trained models using various metrics."""
        print("\n📊 Evaluating models...")
        
        # Get actual test values (close price)
        actual = self.test_series.univariate_component('Close')
        
        evaluation_data = []
        
        for model_name, prediction in self.predictions.items():
            if prediction is not None:
                try:
                    # Calculate metrics
                    mape_score = mape(actual, prediction)
                    rmse_score = rmse(actual, prediction)
                    mae_score = mae(actual, prediction)
                    
                    # Store results
                    self.evaluation_results[model_name] = {
                        'MAPE': mape_score,
                        'RMSE': rmse_score,
                        'MAE': mae_score
                    }
                    
                    evaluation_data.append({
                        'Model': model_name,
                        'MAPE': mape_score,
                        'RMSE': rmse_score,
                        'MAE': mae_score
                    })
                    
                    print(f"✅ {model_name.upper()}:")
                    print(f"   MAPE: {mape_score:.4f}")
                    print(f"   RMSE: {rmse_score:.4f}")
                    print(f"   MAE: {mae_score:.4f}")
                    
                except Exception as e:
                    print(f"❌ Error evaluating {model_name}: {str(e)}")
        
        # Save evaluation results
        if evaluation_data:
            eval_df = pd.DataFrame(evaluation_data)
            eval_file = self.models_dir / 'evaluation_results.csv'
            eval_df.to_csv(eval_file, index=False)
            print(f"\n💾 Evaluation results saved to {eval_file}")
        
        print("\n🏆 Model Performance Summary:")
        print("=" * 50)
        for model_name, metrics in self.evaluation_results.items():
            print(f"{model_name.upper()}: MAPE={metrics['MAPE']:.4f}, RMSE={metrics['RMSE']:.4f}")
    
    def save_models(self) -> None:
        """Save all trained models to disk."""
        print("\n💾 Saving models...")
        
        for model_name, model in self.models.items():
            if model is not None:
                try:
                    model_file = self.models_dir / f"{model_name}_{self.crypto_ticker.replace('-', '_')}.pkl"
                    model.save(str(model_file))
                    print(f"✅ {model_name} model saved to {model_file}")
                except Exception as e:
                    print(f"❌ Error saving {model_name} model: {str(e)}")
    
    def visualize_predictions(self) -> None:
        """Create visualization comparing all model predictions with actual values."""
        print("\n📈 Creating prediction visualizations...")
        
        try:
            plt.figure(figsize=(15, 10))
            
            # Get actual values - convert TimeSeries to pandas DataFrame
            actual = self.test_series.univariate_component('Close')
            actual_values = actual.values().flatten()
            actual_dates = actual.time_index
            
            # Plot actual values
            plt.plot(actual_dates, actual_values, 
                    label='Actual', color='black', linewidth=2)
            
            # Plot predictions from each model
            colors = ['red', 'blue', 'green', 'orange']
            
            for i, (model_name, prediction) in enumerate(self.predictions.items()):
                if prediction is not None:
                    try:
                        pred_values = prediction.values().flatten()
                        pred_dates = prediction.time_index
                        plt.plot(pred_dates, pred_values, 
                                label=f'{model_name.upper()}', 
                                color=colors[i % len(colors)], 
                                linestyle='--', alpha=0.8)
                    except Exception as e:
                        print(f"Warning: Could not plot {model_name}: {str(e)}")
            
            plt.title(f'{self.crypto_ticker} Price Predictions - Model Comparison', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price ($)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_file = self.models_dir / f"predictions_{self.crypto_ticker.replace('-', '_')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Prediction plot saved to {plot_file}")
            
        except Exception as e:
            print(f"❌ Error creating visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_full_pipeline(self, start_date: str = '2020-01-01') -> None:
        """
        Run the complete training pipeline safely.
        
        Args:
            start_date: Start date for data loading
        """
        print("🚀 Starting safe training pipeline...")
        print("="*60)
        
        # Track overall progress
        total_start_time = time.time()
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data(start_date)
            
            # Step 2: Train models (focus on crypto-optimized models)
            print("\n🎯 Training baseline models...")
            self.train_lightgbm_model()
            self.train_exponential_smoothing_model()
            
            print("\n🎯 Training advanced models for crypto volatility...")
            self.train_xgboost_model()  # Excellent for financial data
            self.train_lstm_model()     # Great for sequential/volatile data
            
            print("\n🎯 Training cutting-edge models...")
            self.train_transformer_model()  # Modern transformer approach
            # Skip N-BEATS as it performs poorly on crypto data
            # self.train_nbeats_model()
            
            # Step 3: Evaluate models
            self.evaluate_models()
            
            # Step 4: Save models
            self.save_models()
            
            # Step 5: Create visualizations
            self.visualize_predictions()
            
            # Calculate total time
            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            
            print("\n🎉 Safe training pipeline completed successfully!")
            print("="*60)
            
            # Print summary
            print(f"\nSUMMARY:")
            print(f"Cryptocurrency: {self.crypto_ticker}")
            print(f"Models trained: {len([m for m in self.models.values() if m is not None])}")
            print(f"Data samples: {len(self.data) if self.data is not None else 0}")
            print(f"Forecast horizon: {self.forecast_horizon} days")
            print(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution function."""
    print("💰 QuantBase ML Forecasting Pipeline (Safe Version)")
    print("="*60)
    
    # Initialize forecaster for Bitcoin
    forecaster = SafeCryptoForecaster(crypto_ticker='BTC-USD', forecast_horizon=7)
    
    # Run the full pipeline
    forecaster.run_full_pipeline(start_date='2020-01-01')
    
    print("\n🏁 All done! Check the models/ directory for saved models and results.")


if __name__ == "__main__":
    main()