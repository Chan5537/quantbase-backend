"""
Main training pipeline for cryptocurrency forecasting models.

This module implements the CryptoForecaster class that handles training
of multiple time series forecasting models using the Darts library.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Darts imports
from darts import TimeSeries
from darts.models import LightGBMModel, ExponentialSmoothing, NBEATSModel
from darts.metrics import mape, rmse, mae
from darts.utils.likelihood_models import GaussianLikelihood

# Local imports
from utils.data_loader import (
    get_latest_crypto_data, 
    prepare_darts_timeseries, 
    train_test_split_series,
    validate_data_quality
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CryptoForecaster:
    """
    A comprehensive cryptocurrency forecasting system using multiple models.
    """
    
    def __init__(self, crypto_ticker: str = 'BTC-USD', forecast_horizon: int = 7):
        """
        Initialize the CryptoForecaster.
        
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
        print(f"CryptoForecaster initialized for {crypto_ticker}")
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
            
            # Prepare features for modeling
            value_cols = [
                'Close', 'Volume', 'RSI', 'MACD', 'MA_7', 'MA_30',
                'BB_High', 'BB_Low', 'BB_Middle', 'Volatility'
            ]
            
            # Create TimeSeries object
            full_series = prepare_darts_timeseries(self.data, value_cols)
            
            # Split into train/test
            self.train_series, self.test_series = train_test_split_series(
                full_series, train_ratio=0.85
            )
            
            print(f"✅ Data loaded successfully\!")
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
            model = LightGBMModel(
                lags=14,  # Use 14 days of historical data
                output_chunk_length=self.forecast_horizon,
                random_state=42,
                verbose=-1  # Suppress LightGBM output
            )
            
            # Train the model
            model.fit(self.train_series)
            
            # Make predictions on test set
            predictions = model.predict(n=len(self.test_series))
            
            # Store model and predictions
            self.models['lightgbm'] = model
            self.predictions['lightgbm'] = predictions
            
            print("✅ LightGBM model trained successfully\!")
            
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
            
            model = ExponentialSmoothing(
                trend='add',
                seasonal='add',
                seasonal_periods=7  # Weekly seasonality
            )
            
            # Train the model
            model.fit(close_series)
            
            # Make predictions on test set
            predictions = model.predict(n=len(self.test_series))
            
            # Store model and predictions
            self.models['exponential_smoothing'] = model
            self.predictions['exponential_smoothing'] = predictions
            
            print("✅ Exponential Smoothing model trained successfully\!")
            
        except Exception as e:
            print(f"❌ Error training Exponential Smoothing model: {str(e)}")
            self.models['exponential_smoothing'] = None
            self.predictions['exponential_smoothing'] = None
    
    def train_nbeats_model(self) -> None:
        """Train N-BEATS deep learning model (optional, time permitting)."""
        print("\n🧠 Training N-BEATS model...")
        
        try:
            model = NBEATSModel(
                input_chunk_length=30,
                output_chunk_length=self.forecast_horizon,
                n_epochs=50,  # Limited epochs for speed
                random_state=42,
                pl_trainer_kwargs={
                    "accelerator": "cpu",  # Use CPU for compatibility
                    "enable_progress_bar": False  # Reduce output noise
                }
            )
            
            # Train the model (use only close price for simplicity)
            close_series = self.train_series.univariate_component('Close')
            model.fit(close_series)
            
            # Make predictions on test set
            predictions = model.predict(n=len(self.test_series))
            
            # Store model and predictions
            self.models['nbeats'] = model
            self.predictions['nbeats'] = predictions
            
            print("✅ N-BEATS model trained successfully\!")
            
        except Exception as e:
            print(f"❌ Error training N-BEATS model: {str(e)}")
            print("   Skipping N-BEATS model due to error...")
            self.models['nbeats'] = None
            self.predictions['nbeats'] = None
    
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
            
            # Get actual values
            actual = self.test_series.univariate_component('Close')
            actual_df = actual.pd_dataframe()
            
            # Plot actual values
            plt.plot(actual_df.index, actual_df.values, 
                    label='Actual', color='black', linewidth=2)
            
            # Plot predictions from each model
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, (model_name, prediction) in enumerate(self.predictions.items()):
                if prediction is not None:
                    try:
                        pred_df = prediction.pd_dataframe()
                        plt.plot(pred_df.index, pred_df.values, 
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
    
    def run_full_pipeline(self, start_date: str = '2020-01-01') -> None:
        """
        Run the complete training pipeline.
        
        Args:
            start_date: Start date for data loading
        """
        print("🚀 Starting full training pipeline...")
        print("="*60)
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data(start_date)
            
            # Step 2: Train models
            self.train_lightgbm_model()
            self.train_exponential_smoothing_model()
            
            # Step 3: Optional N-BEATS model (skip if errors)
            try:
                self.train_nbeats_model()
            except Exception as e:
                print(f"Skipping N-BEATS due to error: {str(e)}")
            
            # Step 4: Evaluate models
            self.evaluate_models()
            
            # Step 5: Save models
            self.save_models()
            
            # Step 6: Create visualizations
            self.visualize_predictions()
            
            print("\n🎉 Training pipeline completed successfully\!")
            print("="*60)
            
            # Print summary
            print(f"\nSUMMARY:")
            print(f"Cryptocurrency: {self.crypto_ticker}")
            print(f"Models trained: {len([m for m in self.models.values() if m is not None])}")
            print(f"Data samples: {len(self.data) if self.data is not None else 0}")
            print(f"Forecast horizon: {self.forecast_horizon} days")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise


def main():
    """Main execution function."""
    print("💰 QuantBase ML Forecasting Pipeline")
    print("="*60)
    
    # Initialize forecaster for Bitcoin
    forecaster = CryptoForecaster(crypto_ticker='BTC-USD', forecast_horizon=7)
    
    # Run the full pipeline
    forecaster.run_full_pipeline(start_date='2020-01-01')
    
    print("\n🏁 All done! Check the models/ directory for saved models and results.")


if __name__ == "__main__":
    main()