# QuantBase ML Forecasting Models 🚀

## 🎯 Hackathon Project: Cryptocurrency Price Prediction for Trading Bot

This directory contains a complete ML pipeline for cryptocurrency forecasting, specifically trained on **Solana (SOL-USD)** data for algorithmic trading integration.

## 📊 Key Deliverable

### 🔥 **TRADING BOT DATA**: `prediction/FUTURE_SOL_USD_7day_forecasts.csv`
- **7-day SOL price forecasts** from 8 different ML models
- **Ensemble predictions** (recommended for trading)
- **Expected price changes** and confidence intervals
- **Ready for immediate bot integration**

## 🤖 Models Trained (8 Total)

1. **LightGBM** - Fast gradient boosting (⭐ Best overall)
2. **XGBoost** - Extreme gradient boosting  
3. **Random Forest** - Ensemble tree method
4. **Exponential Smoothing** - Statistical baseline
5. **N-BEATS** - Deep learning for time series
6. **LSTM** - Neural network for sequences  
7. **TiDE Transformer** - Modern transformer approach
8. **TFT** - Temporal Fusion Transformer

## 📁 Directory Structure

```
ml_models/
├── 📄 README.md                           # This file
├── 📓 QuantBase_ML_Training_Colab.ipynb   # GPU training notebook
├── 🔮 prediction/
│   └── FUTURE_SOL_USD_7day_forecasts.csv  # 🎯 MAIN DELIVERABLE
├── 🔧 utils/
│   ├── data_loader.py                      # Crypto data fetching
│   └── __init__.py
├── 🤖 models/                              # Trained model files (not committed)
├── 📊 data/                                # Training data (not committed)  
├── 📋 requirements.txt                     # Python dependencies
├── 🐍 train_models.py                      # Local training script
├── 🐍 train_models_safe.py                # CPU-safe version
├── 🐍 colab_training.py                   # Colab helper script
└── 📖 COLAB_DOWNLOAD_GUIDE.md            # Colab setup instructions
```

## 🚀 Quick Start for Trading Bot Integration

### 1. Load SOL Predictions
```python
import pandas as pd

# Load the main deliverable file
df = pd.read_csv('ml_models/prediction/FUTURE_SOL_USD_7day_forecasts.csv')

# Get ensemble forecast (recommended)
ensemble_forecasts = df['ensemble_forecast'].tolist()
forecast_dates = pd.to_datetime(df['forecast_date'])

print(f"7-day SOL forecasts: {ensemble_forecasts}")
print(f"Target price (7 days): ${ensemble_forecasts[-1]:.2f}")
```

### 2. Individual Model Predictions
```python
# Access individual model predictions
models = ['LightGBM', 'XGBoost', 'LSTM', 'TiDE_Transformer']

for model in models:
    if model in df.columns:
        prediction = df[model].iloc[-1]  # 7-day prediction
        change_pct = df[f'{model}_change_pct'].iloc[-1]
        print(f"{model}: ${prediction:.2f} ({change_pct:+.1f}%)")
```

### 3. Trading Signal Generation
```python
def generate_trading_signal(forecast_row):
    current_price = forecast_row['current_price']
    predicted_price = forecast_row['ensemble_forecast']
    change_pct = forecast_row['ensemble_forecast_change_pct']
    
    if change_pct > 5:
        return 'STRONG_BUY'
    elif change_pct > 2:
        return 'BUY'
    elif change_pct < -5:
        return 'STRONG_SELL'
    elif change_pct < -2:
        return 'SELL'
    else:
        return 'HOLD'

# Example usage
latest_forecast = df.iloc[-1]
signal = generate_trading_signal(latest_forecast)
print(f"Trading Signal: {signal}")
```

## 📊 Model Performance & Training

- **Training Data**: 1000+ days of SOL price history
- **GPU Training**: Google Colab with T4/V100 acceleration  
- **Features**: OHLCV + Technical indicators (RSI, MACD, Bollinger Bands)
- **Validation**: 85% train / 15% test split
- **Ensemble Method**: Equal-weight average of all models

## 🔧 Technical Stack

- **Data Source**: yfinance (Yahoo Finance)
- **ML Library**: Darts (unified time series forecasting)
- **Deep Learning**: PyTorch + PyTorch Lightning
- **Technical Indicators**: ta (Technical Analysis library)
- **Visualization**: matplotlib, pandas

## 📝 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. For Local Training (CPU)
```bash
python train_models_safe.py
```

### 3. For GPU Training (Recommended)
1. Open `QuantBase_ML_Training_Colab.ipynb` in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells
4. Download results using the download cells

## 🎯 Integration Points

### For Backend Developers
- **Main file**: `prediction/FUTURE_SOL_USD_7day_forecasts.csv`
- **Columns**: `ensemble_forecast`, `ensemble_forecast_change_pct`
- **Format**: Standard CSV with datetime index

### For Frontend Developers  
- **Visualization data**: Individual model predictions
- **Charts**: Price forecasts, confidence intervals, model comparison
- **Real-time updates**: Retrain models daily/weekly

### For Trading Algorithm
- **Signal generation**: Based on ensemble forecasts
- **Risk management**: Use confidence intervals
- **Position sizing**: Scale by prediction confidence

## 🏆 Results Summary

- **Best Model**: XGBoost (lowest MAPE)
- **Most Stable**: Ensemble of all 8 models
- **Fastest**: LightGBM (sub-second predictions)
- **Most Accurate**: Deep learning ensemble (LSTM + TiDE + TFT)

## 🔄 Model Updates

To retrain with fresh data:
1. **Google Colab** (Recommended): Run notebook with latest data
2. **Local**: Update data in `utils/data_loader.py` and run training script
3. **Frequency**: Daily for production, weekly for development

## 🚨 Important Notes

- **Crypto Volatility**: Models trained for high-volatility environment
- **Risk Management**: Always implement stop-losses and position limits
- **Backtesting**: Test strategies thoroughly before live deployment
- **Data Quality**: Models perform best with clean, recent data

## 🤝 Team Integration

1. **Clone repo** and checkout `feature/ml-forecasting-models` branch
2. **Review predictions** in `prediction/FUTURE_SOL_USD_7day_forecasts.csv`
3. **Integrate** forecast data into your trading bot
4. **Test** with paper trading before live deployment

## 📞 Support

- **Training Issues**: Check `COLAB_DOWNLOAD_GUIDE.md`
- **Integration Help**: See code examples above
- **Model Questions**: Review model performance in Colab outputs

---

**🎉 Ready for hackathon demo!** 

*GPU-trained models with production-ready forecasts for SOL trading bot integration.*

**Created with ❤️ by the QuantBase ML team**