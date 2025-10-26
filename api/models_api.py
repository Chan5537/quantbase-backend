"""
ML Model Prediction API Endpoints for QuantBase.

This module implements REST API endpoints for cryptocurrency price
forecasting using trained ML models.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field, validator

# Add parent directory to path to import ml_models
sys.path.append(str(Path(__file__).parent.parent))

from ml_models.predict import ModelPredictor
from api.database import get_database, db_manager


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for POST /api/predict endpoint."""
    model_name: str = Field(..., description="Name of the model to use")
    crypto: str = Field(default="BTC-USD", description="Cryptocurrency ticker (e.g., BTC-USD, ETH-USD)")
    days: int = Field(default=7, ge=1, le=30, description="Number of days to forecast (1-30)")
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Remove .pkl extension if present."""
        if v.endswith('.pkl'):
            return v[:-4]
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction endpoints."""
    model: str
    crypto: str
    forecast_horizon: int
    predictions: List[Dict[str, Any]]
    generated_at: str
    processing_time_ms: int
    cached: bool = False


class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    file: str
    size_mb: float
    modified: str
    available: bool
    model_type: Optional[str] = None


class ModelsListResponse(BaseModel):
    """Response model for listing models."""
    models: List[ModelInfo]
    count: int


# Initialize router
router = APIRouter(prefix="/api", tags=["ML Models"])

# Initialize model predictor (singleton pattern)
_predictor: Optional[ModelPredictor] = None


def get_predictor() -> ModelPredictor:
    """
    Get or create ModelPredictor instance.
    
    Returns:
        ModelPredictor instance
    """
    global _predictor
    if _predictor is None:
        models_dir = Path(__file__).parent.parent / 'ml_models' / 'models'
        _predictor = ModelPredictor(models_dir=str(models_dir))
    return _predictor


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    List all available trained ML models.
    
    Returns:
        List of available models with metadata
    
    Example:
        ```
        GET /api/models
        ```
    """
    try:
        predictor = get_predictor()
        models = predictor.list_available_models()
        
        return ModelsListResponse(
            models=[ModelInfo(**model) for model in models],
            count=len(models)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_post(request: PredictionRequest):
    """
    Generate cryptocurrency price predictions using a specific model (POST method).
    
    Args:
        request: Prediction request with model_name, crypto, and days
        
    Returns:
        Prediction results with forecast data
        
    Example:
        ```json
        POST /api/predict
        {
            "model_name": "lightgbm",
            "crypto": "BTC-USD",
            "days": 7
        }
        ```
    """
    start_time = time.time()
    
    try:
        predictor = get_predictor()
        
        # Check if model exists
        available_models = predictor.list_available_models()
        model_names = [m['name'] for m in available_models]
        
        if request.model_name not in model_names and f"{request.model_name}_model" not in model_names:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found. Available models: {model_names}"
            )
        
        # Determine actual model name (with or without _model suffix)
        actual_model = request.model_name
        if actual_model not in model_names:
            actual_model = f"{request.model_name}_model"
        
        # Generate predictions
        predictions_df = predictor.predict(
            model_name=actual_model,
            days=request.days,
            include_confidence=True
        )
        
        # Convert DataFrame to list of dicts
        predictions_list = predictions_df.to_dict('records')
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Prepare response
        response = PredictionResponse(
            model=actual_model,
            crypto=request.crypto,
            forecast_horizon=request.days,
            predictions=predictions_list,
            generated_at=datetime.utcnow().isoformat() + 'Z',
            processing_time_ms=processing_time
        )
        
        # Save to database if available
        if db_manager.connected:
            await db_manager.save_prediction(
                model_name=actual_model,
                crypto=request.crypto,
                predictions=predictions_list,
                metadata={
                    'forecast_horizon': request.days,
                    'processing_time_ms': processing_time
                }
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/predict/{model_name}/{crypto}", response_model=PredictionResponse)
async def predict_get(
    model_name: str,
    crypto: str,
    days: int = Query(default=7, ge=1, le=30, description="Number of days to forecast")
):
    """
    Generate cryptocurrency price predictions using a specific model (GET method).
    
    Args:
        model_name: Name of the model to use
        crypto: Cryptocurrency ticker (e.g., BTC-USD)
        days: Number of days to forecast (1-30)
        
    Returns:
        Prediction results with forecast data
        
    Example:
        ```
        GET /api/predict/lightgbm/BTC-USD?days=7
        ```
    """
    # Reuse POST endpoint logic
    request = PredictionRequest(
        model_name=model_name,
        crypto=crypto,
        days=days
    )
    return await predict_post(request)


@router.get("/compare/{crypto}")
async def compare_models(
    crypto: str,
    days: int = Query(default=7, ge=1, le=30, description="Number of days to forecast")
):
    """
    Compare predictions from ALL available models for a specific cryptocurrency.
    
    Args:
        crypto: Cryptocurrency ticker (e.g., BTC-USD)
        days: Number of days to forecast (1-30)
        
    Returns:
        Comparative predictions from all models
        
    Example:
        ```
        GET /api/compare/BTC-USD?days=7
        ```
    """
    start_time = time.time()
    
    try:
        predictor = get_predictor()
        
        # Get all available models
        available_models = predictor.list_available_models()
        model_names = [m['name'] for m in available_models]
        
        if not model_names:
            raise HTTPException(
                status_code=503,
                detail="No models available for comparison"
            )
        
        # Get predictions from all models
        results = {}
        errors = {}
        
        for model_name in model_names:
            try:
                predictions_df = predictor.predict(
                    model_name=model_name,
                    days=days,
                    include_confidence=True
                )
                results[model_name] = predictions_df.to_dict('records')
            except Exception as e:
                errors[model_name] = str(e)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "crypto": crypto,
            "forecast_horizon": days,
            "models_compared": len(results),
            "predictions": results,
            "errors": errors if errors else None,
            "generated_at": datetime.utcnow().isoformat() + 'Z',
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model comparison failed: {str(e)}"
        )


@router.get("/model/{model_name}/info")
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model information and metadata
        
    Example:
        ```
        GET /api/model/lightgbm/info
        ```
    """
    try:
        predictor = get_predictor()
        
        # Get model info from predictor
        model_info = predictor.get_model_info(model_name)
        
        # Try to get additional metadata from database
        db_metadata = None
        if db_manager.connected:
            db_metadata = await db_manager.get_model_metadata(model_name)
        
        return {
            "model_info": model_info,
            "database_metadata": db_metadata,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/crypto")
async def get_crypto_data():
    """
    Get available cryptocurrency data and recent predictions.
    
    Returns:
        Available cryptocurrencies and recent prediction data
        
    Example:
        ```
        GET /api/crypto
        ```
    """
    try:
        # Mock crypto data for now (you can enhance this later)
        crypto_data = [
            {
                "symbol": "BTC-USD",
                "name": "Bitcoin",
                "current_price": 67420.50,
                "change_24h": 2.45,
                "change_percentage_24h": 3.77,
                "market_cap": 1330000000000,
                "last_updated": datetime.utcnow().isoformat() + 'Z'
            },
            {
                "symbol": "ETH-USD", 
                "name": "Ethereum",
                "current_price": 2650.30,
                "change_24h": 45.20,
                "change_percentage_24h": 1.73,
                "market_cap": 318000000000,
                "last_updated": datetime.utcnow().isoformat() + 'Z'
            },
            {
                "symbol": "SOL-USD",
                "name": "Solana", 
                "current_price": 185.75,
                "change_24h": 8.90,
                "change_percentage_24h": 5.03,
                "market_cap": 87000000000,
                "last_updated": datetime.utcnow().isoformat() + 'Z'
            }
        ]
        
        # Get available models
        predictor = get_predictor()
        available_models = predictor.list_available_models()
        
        return {
            "cryptocurrencies": crypto_data,
            "available_models": [m['name'] for m in available_models],
            "total_cryptos": len(crypto_data),
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get crypto data: {str(e)}"
        )


@router.get("/history/{model_name}")
async def get_prediction_history(
    model_name: str,
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results")
):
    """
    Get prediction history for a specific model from database.
    
    Args:
        model_name: Name of the model
        limit: Maximum number of results to return
        
    Returns:
        List of historical predictions
        
    Example:
        ```
        GET /api/history/lightgbm?limit=10
        ```
    """
    if not db_manager.connected:
        raise HTTPException(
            status_code=503,
            detail="Database not available. History feature requires database connection."
        )
    
    try:
        history = await db_manager.get_prediction_history(
            model_name=model_name,
            limit=limit
        )
        
        return {
            "model_name": model_name,
            "count": len(history),
            "history": history,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )
