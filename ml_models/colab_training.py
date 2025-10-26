"""
Google Colab Training Script for QuantBase ML Models
Upload this to Google Colab for FREE GPU training (much faster!)

Instructions:
1. Go to https://colab.research.google.com/
2. Upload this file
3. Change runtime to GPU (Runtime -> Change runtime type -> GPU)
4. Run all cells
"""

# Install dependencies in Colab
!pip install darts[all] yfinance ta lightgbm xgboost

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# GPU optimization for Colab
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Data loading functions (simplified for Colab)
def fetch_crypto_data(ticker='BTC-USD', days_back=365):
    """Fetch and process crypto data"""
    import yfinance as yf
    import ta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    crypto = yf.Ticker(ticker)
    data = crypto.history(start=start_date, end=end_date)
    data.index = data.index.tz_localize(None)
    
    # Add technical indicators
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(close=data['Close']).macd()
    data['MA_7'] = ta.trend.SMAIndicator(close=data['Close'], window=7).sma_indicator()
    data['MA_30'] = ta.trend.SMAIndicator(close=data['Close'], window=30).sma_indicator()
    data['Volatility'] = data['Close'].rolling(window=14).std()
    
    return data.dropna()

def train_fast_models():
    """Train models optimized for speed and GPU"""
    from darts import TimeSeries
    from darts.models import LightGBMModel, RNNModel, TiDEModel
    from darts.metrics import mape, rmse, mae
    
    print("🚀 QuantBase Fast Training (GPU Optimized)")
    print("="*50)
    
    # Load data
    print("📊 Loading BTC data...")
    data = fetch_crypto_data('BTC-USD', days_back=1000)  # More data for better training
    print(f"   Loaded {len(data)} days of data")
    
    # Prepare time series
    close_series = TimeSeries.from_dataframe(
        data[['Close']], 
        time_col=None, 
        freq='D'
    )
    
    # Split data
    split_point = int(len(close_series) * 0.85)
    train_series = close_series[:split_point]
    test_series = close_series[split_point:]
    
    print(f"   Train: {len(train_series)} days, Test: {len(test_series)} days")
    
    models = {}
    predictions = {}
    results = {}
    
    # 1. LightGBM (Fast baseline)
    print("\n🚀 Training LightGBM...")
    start_time = time.time()
    lgb_model = LightGBMModel(
        lags=14,
        output_chunk_length=7,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(train_series)
    lgb_pred = lgb_model.predict(n=len(test_series))
    models['LightGBM'] = lgb_model
    predictions['LightGBM'] = lgb_pred
    print(f"   ✅ Completed in {time.time() - start_time:.1f}s")
    
    # 2. LSTM (GPU accelerated)
    print("\n🔗 Training LSTM (GPU)...")
    start_time = time.time()
    lstm_model = RNNModel(
        model='LSTM',
        input_chunk_length=21,
        output_chunk_length=7,
        hidden_dim=128,  # Larger since we have GPU
        n_rnn_layers=3,  # More layers with GPU
        dropout=0.2,
        batch_size=64,   # Larger batch for GPU
        n_epochs=100,    # More epochs with GPU speed
        optimizer_kwargs={'lr': 0.001},
        pl_trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "enable_progress_bar": True,
            "max_epochs": 100
        },
        random_state=42
    )
    lstm_model.fit(train_series, verbose=True)
    lstm_pred = lstm_model.predict(n=len(test_series))
    models['LSTM'] = lstm_model
    predictions['LSTM'] = lstm_pred
    print(f"   ✅ Completed in {time.time() - start_time:.1f}s")
    
    # 3. TiDE Transformer (GPU accelerated)
    print("\n🔮 Training TiDE Transformer (GPU)...")
    start_time = time.time()
    tide_model = TiDEModel(
        input_chunk_length=28,
        output_chunk_length=7,
        num_encoder_layers=4,    # More layers with GPU
        num_decoder_layers=4,
        hidden_dim=256,          # Larger model with GPU
        n_epochs=50,
        batch_size=64,
        pl_trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "enable_progress_bar": True,
            "max_epochs": 50
        },
        random_state=42
    )
    tide_model.fit(train_series, verbose=True)
    tide_pred = tide_model.predict(n=len(test_series))
    models['TiDE'] = tide_model
    predictions['TiDE'] = tide_pred
    print(f"   ✅ Completed in {time.time() - start_time:.1f}s")
    
    # Evaluate all models
    print("\n📊 Model Evaluation:")
    print("="*40)
    
    for name, pred in predictions.items():
        mape_score = mape(test_series, pred)
        rmse_score = rmse(test_series, pred)
        mae_score = mae(test_series, pred)
        
        results[name] = {
            'MAPE': mape_score,
            'RMSE': rmse_score,
            'MAE': mae_score
        }
        
        print(f"{name}:")
        print(f"  MAPE: {mape_score:.4f}")
        print(f"  RMSE: {rmse_score:.2f}")
        print(f"  MAE:  {mae_score:.2f}")
        print()
    
    # Plot results
    plt.figure(figsize=(15, 8))
    
    # Plot actual
    actual_values = test_series.values().flatten()
    actual_dates = test_series.time_index
    plt.plot(actual_dates, actual_values, label='Actual', color='black', linewidth=2)
    
    # Plot predictions
    colors = ['red', 'blue', 'green']
    for i, (name, pred) in enumerate(predictions.items()):
        pred_values = pred.values().flatten()
        pred_dates = pred.time_index
        plt.plot(pred_dates, pred_values, 
                label=name, color=colors[i], linestyle='--', alpha=0.8)
    
    plt.title('BTC Price Predictions - GPU Trained Models')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['MAPE'])
    print(f"🏆 Best Model: {best_model} (MAPE: {results[best_model]['MAPE']:.4f})")
    
    return models, predictions, results

# Check GPU availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print("🚀 Ready for fast GPU training!")
else:
    print("⚠️  No GPU detected, will use CPU (slower)")

# Run training
if __name__ == "__main__":
    models, predictions, results = train_fast_models()
    
    print("\n🎉 Training Complete!")
    print("Models are ready for use in your QuantBase application.")