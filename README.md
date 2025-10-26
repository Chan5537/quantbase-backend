# QuantBase API

Machine learning-powered cryptocurrency trading prediction API.

## 🚀 Quick Start

### Deploy to Railway (Recommended)

Railway is perfect for this project because it:
- Has generous size limits for ML models
- Provides always-on VMs (no cold starts)
- Includes one-click MongoDB setup
- Offers a free tier

**5-minute deploy**: See [QUICK_START_RAILWAY.md](QUICK_START_RAILWAY.md)

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your keys

# Run the API
python -m uvicorn api.main:app --port 8000
```

API will be available at http://localhost:8000

## 📚 Documentation

- **Quick Start**: [QUICK_START_RAILWAY.md](QUICK_START_RAILWAY.md)
- **Full Deployment Guide**: [DEPLOY_TO_RAILWAY.md](DEPLOY_TO_RAILWAY.md)
- **Environment Variables**: [RAILWAY_ENV_VARS.md](RAILWAY_ENV_VARS.md)
- **API Documentation**: [README_API.md](README_API.md)

## 🌟 Features

- Multiple ML models (LightGBM, XGBoost, Random Forest, Neural Networks)
- Real-time cryptocurrency price predictions
- Trading bot management
- Claude AI-powered bot personalization
- MongoDB integration for data persistence

## 📖 API Endpoints

- `GET /docs` - Interactive API documentation
- `GET /health` - Health check
- `POST /api/predict` - Generate ML predictions
- `GET /api/bots` - List all bots
- `POST /api/bots/create` - Create new bot

## 🔧 Configuration

Required environment variables:

- `MONGODB_URI` - MongoDB connection string
- `CLAUDE_API_KEY` - Anthropic API key

Optional:

- `ALLOWED_ORIGINS` - CORS origins (default: localhost:3000)
- `ENVIRONMENT` - Environment name (default: development)
- `API_VERSION` - API version (default: 1.0.0)

## 📦 Tech Stack

- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **MongoDB** - Database
- **LightGBM, XGBoost** - ML models
- **Claude AI** - Bot personalization

## 📝 License

MIT
