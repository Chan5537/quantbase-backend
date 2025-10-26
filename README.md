# Quantbase: Democratizing algorithmic trading with AI

> The world's first AI-powered Algo Trading Bot Marketplace

Quantbase is a machine learning-powered cryptocurrency trading prediction API that combines multiple ML models with Claude AI to create, manage, and personalize automated trading bots.

## Features

- **Multi-Model ML Predictions**: Ensemble approach using LightGBM, XGBoost, Random Forest, and Neural Networks
- **Real-Time Price Predictions**: Get cryptocurrency price predictions powered by advanced ML models
- **Trading Bot Marketplace**: Create, customize, and manage automated trading bots
- **Claude AI Integration**: Personalize trading bots with natural language using Anthropic's Claude
- **MongoDB Persistence**: Robust data storage for bots, predictions, and trading history
- **RESTful API**: Clean, well-documented API built with FastAPI

## Tech Stack

- **Framework**: FastAPI
- **Server**: Uvicorn (ASGI)
- **Database**: MongoDB
- **ML Models**: LightGBM, XGBoost, Random Forest, Neural Networks
- **AI**: Claude API (Anthropic)
- **Deployment**: Railway (recommended)

## Prerequisites

- Python 3.8+
- MongoDB instance (local or cloud)
- Anthropic Claude API key

## ⚡ Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/Chan5537/quantbase-backend.git
cd quantbase-backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your MongoDB URI and Claude API key

# Run the API
python -m uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## 🔧 Configuration

### Required Environment Variables

```env
MONGODB_URI=your_mongodb_connection_string
CLAUDE_API_KEY=your_anthropic_api_key
```

### Optional Environment Variables

```env
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
ENVIRONMENT=development
API_VERSION=1.0.0
```

## API Documentation

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/docs` | Interactive Swagger UI documentation |
| GET | `/health` | Health check endpoint |
| POST | `/api/predict` | Generate ML-powered price predictions |
| GET | `/api/bots` | List all trading bots |
| POST | `/api/bots/create` | Create a new trading bot |

### Interactive Documentation

Once the API is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

For detailed API documentation, see [README_API.md](README_API.md)

## Machine Learning Models

Quantbase uses an ensemble of state-of-the-art ML models:

1. **LightGBM**: Fast gradient boosting framework
2. **XGBoost**: Extreme gradient boosting for high accuracy
3. **Random Forest**: Ensemble learning for robust predictions
4. **Neural Networks**: Deep learning for complex pattern recognition

Models are trained on historical cryptocurrency data and continuously updated for optimal performance.

## Use Cases

- Create AI-powered trading bots with custom strategies
- Get real-time cryptocurrency price predictions
- Backtest trading algorithms
- Personalize bot behavior using natural language (Claude AI)
- Build a marketplace of trading strategies

## Development

### Project Structure

```
quantbase-backend/
├── api/
│   ├── main.py              # FastAPI application
│   ├── models/              # ML model implementations
│   ├── routes/              # API route handlers
│   └── services/            # Business logic
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

### Running Tests

```bash
pytest
```

## Contributing

This is a hackathon project! Contributions, issues, and feature requests are welcome.

**Quantbase** - Democratizing algorithmic trading with AI
