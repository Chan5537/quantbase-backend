"""
Data loading and preprocessing utilities for cryptocurrency forecasting.

This module handles data fetching from yfinance, technical indicator calculation,
and preparation for Darts time series models.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from typing import Tuple, List, Optional
from darts import TimeSeries
from datetime import datetime, timedelta


def fetch_crypto_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch cryptocurrency data from Yahoo Finance.
    
    Args:
        ticker: Cryptocurrency ticker symbol (e.g., 'BTC-USD', 'ETH-USD')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with OHLCV data and timezone removed
    """
    try:
        crypto = yf.Ticker(ticker)
        data = crypto.history(start=start_date, end=end_date)
        
        # Remove timezone info for Darts compatibility
        data.index = data.index.tz_localize(None)
        
        # Ensure daily frequency
        data = data.asfreq('D')
        
        return data
    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to cryptocurrency data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical indicator columns
    """
    data = df.copy()
    
    # RSI (14-day window)
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    
    # MACD
    macd_indicator = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd_indicator.macd()
    data['MACD_Signal'] = macd_indicator.macd_signal()
    data['MACD_Histogram'] = macd_indicator.macd_diff()
    
    # Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb_indicator.bollinger_hband()
    data['BB_Low'] = bb_indicator.bollinger_lband()
    data['BB_Middle'] = bb_indicator.bollinger_mavg()
    data['BB_Width'] = data['BB_High'] - data['BB_Low']
    
    # Moving Averages
    data['MA_7'] = ta.trend.SMAIndicator(close=data['Close'], window=7).sma_indicator()
    data['MA_30'] = ta.trend.SMAIndicator(close=data['Close'], window=30).sma_indicator()
    
    # Volume Moving Average
    data['Volume_MA_7'] = ta.trend.SMAIndicator(close=data['Volume'], window=7).sma_indicator()
    
    # Price momentum indicators
    data['Price_Change'] = data['Close'].pct_change()
    data['High_Low_Ratio'] = data['High'] / data['Low']
    
    # Volatility
    data['Volatility'] = data['Close'].rolling(window=14).std()
    
    # Remove NaN values
    data = data.dropna()
    
    return data


def prepare_darts_timeseries(df: pd.DataFrame, value_cols: List[str]) -> TimeSeries:
    """
    Convert pandas DataFrame to Darts TimeSeries object.
    
    Args:
        df: DataFrame with time series data
        value_cols: List of column names to include in the time series
        
    Returns:
        Darts TimeSeries object
    """
    try:
        # Ensure the DataFrame has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        
        # Select only the specified columns
        data = df[value_cols].copy()
        
        # Create TimeSeries with daily frequency
        ts = TimeSeries.from_dataframe(
            data,
            time_col=None,  # Use index as time column
            freq='D',
            fill_missing_dates=True
        )
        
        return ts
    except Exception as e:
        raise Exception(f"Error creating TimeSeries: {str(e)}")


def train_test_split_series(series: TimeSeries, train_ratio: float = 0.85) -> Tuple[TimeSeries, TimeSeries]:
    """
    Split time series into train and test sets.
    
    Args:
        series: Darts TimeSeries object
        train_ratio: Ratio of data to use for training (default: 0.85)
        
    Returns:
        Tuple of (train_series, test_series)
    """
    split_point = int(len(series) * train_ratio)
    train = series[:split_point]
    test = series[split_point:]
    
    return train, test


def get_latest_crypto_data(ticker: str, days_back: int = 365) -> pd.DataFrame:
    """
    Fetch the most recent cryptocurrency data.
    
    Args:
        ticker: Cryptocurrency ticker symbol
        days_back: Number of days to look back from today
        
    Returns:
        DataFrame with processed cryptocurrency data
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    # Fetch data
    data = fetch_crypto_data(ticker, start_date, end_date)
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    return data


def validate_data_quality(df: pd.DataFrame, min_rows: int = 100) -> bool:
    """
    Validate data quality for time series forecasting.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        
    Returns:
        True if data quality is sufficient, False otherwise
    """
    if len(df) < min_rows:
        print(f"Warning: Only {len(df)} rows available, minimum {min_rows} recommended")
        return False
    
    # Check for excessive missing values
    missing_pct = df.isnull().sum().max() / len(df) * 100
    if missing_pct > 5:
        print(f"Warning: High percentage of missing values: {missing_pct:.2f}%")
        return False
    
    # Check for constant values
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns and df[col].nunique() < 10:
            print(f"Warning: Column {col} has very few unique values")
            return False
    
    return True


if __name__ == "__main__":
    # Test the data loading functionality
    print("Testing data loading for BTC-USD...")
    
    try:
        # Fetch recent BTC data
        btc_data = get_latest_crypto_data('BTC-USD', days_back=365)
        print(f"Successfully loaded {len(btc_data)} days of BTC data")
        print(f"Date range: {btc_data.index.min()} to {btc_data.index.max()}")
        print(f"Columns: {list(btc_data.columns)}")
        
        # Validate data quality
        is_valid = validate_data_quality(btc_data)
        print(f"Data quality check: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test Darts conversion
        value_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MA_7', 'MA_30']
        ts = prepare_darts_timeseries(btc_data, value_cols)
        print(f"Successfully created TimeSeries with shape: {ts.values().shape}")
        
        # Test train/test split
        train, test = train_test_split_series(ts, train_ratio=0.85)
        print(f"Train set: {len(train)} samples, Test set: {len(test)} samples")
        
        print("Data loading test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")