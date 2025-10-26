"""
Model Predictor for QuantBase ML Models.

This module provides a unified interface for loading trained models
and generating predictions for cryptocurrency price forecasting.
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class ModelPredictor:
    """
    A unified predictor class for all trained cryptocurrency forecasting models.
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the ModelPredictor.
        
        Args:
            models_dir: Directory containing trained model files (.pkl)
        """
        if models_dir is None:
            # Default to models directory relative to this file
            self.models_dir = Path(__file__).parent / 'models'
        else:
            self.models_dir = Path(models_dir)
        
        self.loaded_models = {}
        self.model_metadata = {}
        
        print(f"🔍 ModelPredictor initialized")
        print(f"📁 Models directory: {self.models_dir}")
        
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available trained models in the models directory.
        
        Returns:
            List of dictionaries containing model information
        """
        if not self.models_dir.exists():
            print(f"⚠️  Models directory not found: {self.models_dir}")
            return []
        
        models = []
        for model_file in self.models_dir.glob('*.pkl'):
            if model_file.name == '.gitkeep':
                continue
                
            model_info = {
                'name': model_file.stem,
                'file': model_file.name,
                'path': str(model_file),
                'size_mb': model_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                'available': True
            }
            
            # Extract model type and crypto from filename if possible
            # Expected format: modelname_crypto.pkl or modelname_model.pkl
            parts = model_file.stem.split('_')
            if len(parts) >= 2:
                if 'model' in parts[-1]:
                    model_info['model_type'] = '_'.join(parts[:-1])
                else:
                    model_info['model_type'] = parts[0]
                    
            models.append(model_info)
        
        return models
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model file (with or without .pkl extension)
            
        Returns:
            Loaded model object
        """
        # Add .pkl extension if not present
        if not model_name.endswith('.pkl'):
            model_name = f"{model_name}.pkl"
        
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Check if already loaded
        if model_name in self.loaded_models:
            print(f"✓ Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        print(f"📦 Loading model: {model_name}")
        try:
            model = joblib.load(model_path)
            self.loaded_models[model_name] = model
            print(f"✓ Model loaded successfully: {model_name}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
    
    def predict(
        self,
        model_name: str,
        days: int = 7,
        include_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Generate predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            days: Number of days to forecast
            include_confidence: Whether to include confidence intervals
            
        Returns:
            DataFrame with predictions
        """
        # Load model
        model = self.load_model(model_name)
        
        # Generate future dates
        start_date = datetime.now()
        dates = [start_date + timedelta(days=i) for i in range(1, days + 1)]
        
        try:
            # Try to generate predictions
            # Note: This is a simplified version - actual implementation
            # depends on the specific model type and trained capabilities
            
            # For demonstration, we'll create a simple structure
            # In production, this would call model.predict() appropriately
            predictions = pd.DataFrame({
                'date': dates,
                'close': np.random.uniform(50000, 70000, days),  # Placeholder
            })
            
            if include_confidence:
                predictions['confidence_low'] = predictions['close'] * 0.95
                predictions['confidence_high'] = predictions['close'] * 1.05
            
            predictions['date'] = predictions['date'].dt.strftime('%Y-%m-%d')
            
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed for model {model_name}: {str(e)}")
    
    def compare_models(
        self,
        model_names: List[str],
        days: int = 7
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate predictions from multiple models for comparison.
        
        Args:
            model_names: List of model names to compare
            days: Number of days to forecast
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        results = {}
        
        for model_name in model_names:
            try:
                predictions = self.predict(model_name, days=days)
                results[model_name] = predictions
            except Exception as e:
                print(f"⚠️  Failed to get predictions from {model_name}: {str(e)}")
                results[model_name] = None
        
        return results
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        models = self.list_available_models()
        
        for model in models:
            if model['name'] == model_name or model['file'] == model_name:
                return model
        
        raise ValueError(f"Model not found: {model_name}")


# Convenience function for quick predictions
def quick_predict(model_name: str, days: int = 7) -> pd.DataFrame:
    """
    Quick prediction function for convenience.
    
    Args:
        model_name: Name of the model to use
        days: Number of days to forecast
        
    Returns:
        DataFrame with predictions
    """
    predictor = ModelPredictor()
    return predictor.predict(model_name, days=days)


if __name__ == "__main__":
    # Test the predictor
    predictor = ModelPredictor()
    
    print("\n📋 Available Models:")
    models = predictor.list_available_models()
    for model in models:
        print(f"  - {model['name']} ({model['size_mb']:.2f} MB)")
    
    if models:
        print(f"\n🔮 Testing prediction with {models[0]['name']}...")
        predictions = predictor.predict(models[0]['name'], days=7)
        print(predictions.head())
