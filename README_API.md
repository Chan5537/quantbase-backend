# QuantBase API Documentation

**Version:** 1.0.0  
**Base URL (Local):** `http://localhost:8000`  
**Base URL (Production):** `https://[your-railway-url].railway.app` _(Update after deployment)_

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Authentication](#authentication)
4. [Endpoints](#endpoints)
5. [Request/Response Examples](#requestresponse-examples)
6. [Error Codes](#error-codes)
7. [Integration Guide](#integration-guide)
8. [Rate Limits](#rate-limits)

---

## Overview

The QuantBase API provides machine learning-powered cryptocurrency price predictions using multiple trained forecasting models. The API is built with FastAPI and deployed on Railway.app.

### Features

- 🔮 **Multiple ML Models**: LightGBM, XGBoost, Random Forest, Neural Networks (NBEATS, TFT, TIDE)
- 📈 **Flexible Forecasting**: 1-30 day price predictions
- 🎯 **Confidence Intervals**: Uncertainty estimation for predictions
- 🔄 **Model Comparison**: Compare predictions from all available models
- 💾 **History Tracking**: View past predictions (requires MongoDB)
- 📊 **Model Metadata**: Access model performance metrics

---

## Getting Started

### Base URL

Set your API base URL as an environment variable in your Next.js frontend:

```bash
# .env.local (Next.js)
NEXT_PUBLIC_API_URL=https://your-railway-url.railway.app
```

### Interactive Documentation

Once deployed, you can access interactive API documentation at:

- **Swagger UI**: `https://your-railway-url.railway.app/docs`
- **ReDoc**: `https://your-railway-url.railway.app/redoc`

---

## Authentication

**Authentication:** None required for MVP.

All endpoints are currently public. Authentication will be added in future versions.

---

## Endpoints

### 1. Root - Health Check

**GET** `/`

Basic health check to verify API is online.

**Response:**
```json
{
  "message": "Welcome to QuantBase API",
  "status": "online",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### 2. Detailed Health Check

**GET** `/health`

Get detailed health status of API and its dependencies.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1698234567.89,
  "environment": "production",
  "version": "1.0.0",
  "components": {
    "database": {
      "status": "connected",
      "healthy": true,
      "required": false
    },
    "ml_models": {
      "status": "available",
      "count": 8,
      "healthy": true,
      "required": true
    }
  }
}
```

---

### 3. List Available Models

**GET** `/api/models`

List all available trained ML models.

**Response:**
```json
{
  "models": [
    {
      "name": "lightgbm_model",
      "file": "lightgbm_model.pkl",
      "size_mb": 0.52,
      "modified": "2024-10-25T10:30:00",
      "available": true,
      "model_type": "lightgbm"
    },
    {
      "name": "xgboost_model",
      "file": "xgboost_model.pkl",
      "size_mb": 0.48,
      "modified": "2024-10-25T10:30:00",
      "available": true,
      "model_type": "xgboost"
    }
  ],
  "count": 8
}
```

---

### 4. Generate Prediction (POST)

**POST** `/api/predict`

Generate cryptocurrency price predictions using a specific model.

**Request Body:**
```json
{
  "model_name": "lightgbm",
  "crypto": "BTC-USD",
  "days": 7
}
```

**Parameters:**
- `model_name` (string, required): Name of the model to use (e.g., "lightgbm", "xgboost")
- `crypto` (string, optional): Cryptocurrency ticker (default: "BTC-USD")
- `days` (integer, optional): Number of days to forecast (1-30, default: 7)

**Response:**
```json
{
  "model": "lightgbm_model",
  "crypto": "BTC-USD",
  "forecast_horizon": 7,
  "predictions": [
    {
      "date": "2024-10-26",
      "close": 67234.56,
      "confidence_low": 65000.12,
      "confidence_high": 69000.45
    },
    {
      "date": "2024-10-27",
      "close": 67890.12,
      "confidence_low": 65500.34,
      "confidence_high": 69500.78
    }
  ],
  "generated_at": "2024-10-25T15:30:00Z",
  "processing_time_ms": 145,
  "cached": false
}
```

---

### 5. Generate Prediction (GET)

**GET** `/api/predict/{model_name}/{crypto}?days=7`

Generate predictions using query parameters (alternative to POST).

**Parameters:**
- `model_name` (path, required): Name of the model
- `crypto` (path, required): Cryptocurrency ticker
- `days` (query, optional): Number of days to forecast (1-30, default: 7)

**Example:**
```
GET /api/predict/lightgbm/BTC-USD?days=7
```

**Response:** Same as POST `/api/predict`

---

### 6. Compare Models

**GET** `/api/compare/{crypto}?days=7`

Get predictions from ALL available models for comparison.

**Parameters:**
- `crypto` (path, required): Cryptocurrency ticker
- `days` (query, optional): Number of days to forecast (1-30, default: 7)

**Example:**
```
GET /api/compare/BTC-USD?days=7
```

**Response:**
```json
{
  "crypto": "BTC-USD",
  "forecast_horizon": 7,
  "models_compared": 8,
  "predictions": {
    "lightgbm_model": [
      {"date": "2024-10-26", "close": 67234.56, "confidence_low": 65000, "confidence_high": 69000}
    ],
    "xgboost_model": [
      {"date": "2024-10-26", "close": 67150.23, "confidence_low": 64900, "confidence_high": 68900}
    ]
  },
  "errors": null,
  "generated_at": "2024-10-25T15:30:00Z",
  "processing_time_ms": 523
}
```

---

### 7. Get Model Information

**GET** `/api/model/{model_name}/info`

Get detailed information about a specific model.

**Example:**
```
GET /api/model/lightgbm/info
```

**Response:**
```json
{
  "model_info": {
    "name": "lightgbm_model",
    "file": "lightgbm_model.pkl",
    "size_mb": 0.52,
    "modified": "2024-10-25T10:30:00",
    "available": true,
    "model_type": "lightgbm"
  },
  "database_metadata": {
    "model_name": "lightgbm_model",
    "metadata": {
      "mape": 3.45,
      "rmse": 1234.56
    }
  },
  "timestamp": "2024-10-25T15:30:00Z"
}
```

---

### 8. Get Prediction History

**GET** `/api/history/{model_name}?limit=10`

Retrieve past predictions for a specific model (requires MongoDB).

**Parameters:**
- `model_name` (path, required): Name of the model
- `limit` (query, optional): Maximum results (1-100, default: 10)

**Example:**
```
GET /api/history/lightgbm?limit=10
```

**Response:**
```json
{
  "model_name": "lightgbm",
  "count": 10,
  "history": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "model_name": "lightgbm_model",
      "crypto": "BTC-USD",
      "predictions": [...],
      "created_at": "2024-10-25T14:00:00",
      "metadata": {...}
    }
  ],
  "timestamp": "2024-10-25T15:30:00Z"
}
```

---

## Request/Response Examples

### JavaScript/TypeScript (Next.js)

#### API Client Setup

```typescript
// lib/api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function getModels() {
  const response = await fetch(`${API_URL}/api/models`);
  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.statusText}`);
  }
  return response.json();
}

export async function getPrediction(
  modelName: string,
  crypto: string = 'BTC-USD',
  days: number = 7
) {
  const response = await fetch(
    `${API_URL}/api/predict/${modelName}/${crypto}?days=${days}`
  );
  if (!response.ok) {
    throw new Error(`Failed to get prediction: ${response.statusText}`);
  }
  return response.json();
}

export async function comparePredictions(crypto: string = 'BTC-USD', days: number = 7) {
  const response = await fetch(`${API_URL}/api/compare/${crypto}?days=${days}`);
  if (!response.ok) {
    throw new Error(`Failed to compare models: ${response.statusText}`);
  }
  return response.json();
}

export async function postPrediction(
  modelName: string,
  crypto: string = 'BTC-USD',
  days: number = 7
) {
  const response = await fetch(`${API_URL}/api/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_name: modelName,
      crypto: crypto,
      days: days,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to get prediction: ${response.statusText}`);
  }
  return response.json();
}

export async function checkHealth() {
  const response = await fetch(`${API_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}
```

#### React Component Example

```tsx
// components/PredictionChart.tsx
'use client';

import { useEffect, useState } from 'react';
import { getPrediction } from '@/lib/api';

interface Prediction {
  date: string;
  close: number;
  confidence_low?: number;
  confidence_high?: number;
}

export default function PredictionChart() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadPredictions() {
      try {
        setLoading(true);
        const result = await getPrediction('lightgbm', 'BTC-USD', 7);
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load predictions');
      } finally {
        setLoading(false);
      }
    }

    loadPredictions();
  }, []);

  if (loading) return <div>Loading predictions...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!data) return <div>No data available</div>;

  return (
    <div>
      <h2>{data.model} - {data.crypto}</h2>
      <p>Generated: {new Date(data.generated_at).toLocaleString()}</p>
      <p>Processing time: {data.processing_time_ms}ms</p>
      
      <ul>
        {data.predictions.map((pred: Prediction, i: number) => (
          <li key={i}>
            {pred.date}: ${pred.close.toFixed(2)}
            {pred.confidence_low && pred.confidence_high && (
              <span> (${pred.confidence_low.toFixed(2)} - ${pred.confidence_high.toFixed(2)})</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/api/models

# Get prediction (GET)
curl "http://localhost:8000/api/predict/lightgbm/BTC-USD?days=7"

# Get prediction (POST)
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lightgbm",
    "crypto": "BTC-USD",
    "days": 7
  }'

# Compare models
curl "http://localhost:8000/api/compare/BTC-USD?days=7"

# Get model info
curl http://localhost:8000/api/model/lightgbm/info
```

---

## Error Codes

| Status Code | Description | Example Response |
|------------|-------------|------------------|
| `200` | Success | Request completed successfully |
| `400` | Bad Request | Invalid parameters (e.g., days > 30) |
| `404` | Not Found | Model or endpoint not found |
| `422` | Validation Error | Request body validation failed |
| `500` | Internal Server Error | Server-side error occurred |
| `503` | Service Unavailable | Models not loaded or database unavailable |

### Error Response Format

```json
{
  "error": "Model 'invalid_model' not found",
  "status_code": 404,
  "path": "/api/predict/invalid_model/BTC-USD"
}
```

### Validation Error Example

```json
{
  "error": "Validation Error",
  "detail": [
    {
      "loc": ["body", "days"],
      "msg": "ensure this value is less than or equal to 30",
      "type": "value_error.number.not_le"
    }
  ]
}
```

---

## Integration Guide

### Step 1: Set Environment Variable

In your Next.js project, create `.env.local`:

```bash
NEXT_PUBLIC_API_URL=https://your-railway-url.railway.app
```

### Step 2: Create API Client

Use the TypeScript code from the examples above to create `lib/api.ts`.

### Step 3: Use in Components

Import and use the API functions in your React components:

```tsx
import { getPrediction, getModels } from '@/lib/api';
```

### Step 4: Handle Errors

Always wrap API calls in try-catch blocks:

```tsx
try {
  const data = await getPrediction('lightgbm', 'BTC-USD', 7);
  // Handle success
} catch (error) {
  // Handle error
  console.error('API Error:', error);
}
```

### Step 5: Type Safety (Optional)

Define TypeScript interfaces for type safety:

```typescript
// types/api.ts
export interface PredictionPoint {
  date: string;
  close: number;
  confidence_low?: number;
  confidence_high?: number;
}

export interface PredictionResponse {
  model: string;
  crypto: string;
  forecast_horizon: number;
  predictions: PredictionPoint[];
  generated_at: string;
  processing_time_ms: number;
  cached: boolean;
}

export interface ModelInfo {
  name: string;
  file: string;
  size_mb: number;
  modified: string;
  available: boolean;
  model_type?: string;
}
```

---

## Rate Limits

**Current:** No rate limiting implemented (MVP).

**Future:** Rate limiting will be added in production:
- **Default**: 60 requests per minute per IP
- **Configurable** via `RATE_LIMIT_PER_MINUTE` environment variable

---

## Support & Contact

For questions, issues, or feature requests:

- **GitHub Issues**: [3arii/project-steve](https://github.com/3arii/project-steve/issues)
- **Email**: Contact your QuantBase team lead

---

## Changelog

### Version 1.0.0 (2024-10-25)

- ✅ Initial API release
- ✅ 8 trained ML models
- ✅ Prediction endpoints (GET/POST)
- ✅ Model comparison endpoint
- ✅ MongoDB integration for history
- ✅ Health check endpoints
- ✅ CORS configuration for frontend

---

## Appendix

### Available Model Types

| Model Name | Type | Description |
|-----------|------|-------------|
| `lightgbm` | Gradient Boosting | Fast, efficient tree-based model |
| `xgboost` | Gradient Boosting | Powerful ensemble learning |
| `random_forest` | Ensemble | Robust random forest regressor |
| `nbeats` | Neural Network | N-BEATS deep learning model |
| `tft` | Transformer | Temporal Fusion Transformer |
| `tide` | Transformer | Time-series Dense Encoder |
| `lstm` | RNN | Long Short-Term Memory network |
| `exponential_smoothing` | Statistical | Classical time series method |

### Cryptocurrency Tickers

Currently supported:
- `BTC-USD` - Bitcoin
- `ETH-USD` - Ethereum (if trained)
- `SOL-USD` - Solana (if trained)

---

**Last Updated:** 2024-10-25  
**API Version:** 1.0.0
