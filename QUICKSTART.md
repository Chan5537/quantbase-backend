# QuantBase API - Quick Start Guide

Get the QuantBase API running locally in under 5 minutes! ⚡

## 🎯 Prerequisites

- Python 3.8+ installed
- Git repository cloned
- Terminal/Command Line access

## 🚀 Quick Start (Local Development)

### 1. Navigate to Project Directory

```bash
cd /Users/chanyeong/Desktop/Hackathons/project-steve
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
.\venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (web framework)
- Motor & PyMongo (MongoDB)
- Darts, LightGBM, XGBoost (ML libraries)
- All other dependencies

**Note:** Installation may take 5-10 minutes due to ML libraries.

### 4. Verify Environment Variables

Check that `.env` file exists with MongoDB credentials:

```bash
cat .env
```

Should contain:
```bash
MONGODB_URI=mongodb+srv://denizlapsekili_db_user:4nUh2jZIkgYEAfWM@solana-data.cx4xp7h.mongodb.net/?appName=solana-data
DATABASE_NAME=quantbase
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

### 5. Start the Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:

```
╔═══════════════════════════════════════════════════════════════╗
║                    QUANTBASE API STARTING                     ║
╚═══════════════════════════════════════════════════════════════╝

🚀 QuantBase API Starting...
📝 Environment: development
🔧 API Version: 1.0.0
============================================================

🗄️  Initializing Database Connection...
✓ Connected to MongoDB: quantbase

🤖 Initializing ML Models...
✓ Found 8 trained models:
  • lightgbm_model (0.52 MB)
  • xgboost_model (0.48 MB)
  ...

============================================================
✅ QuantBase API is ready!
📚 Documentation: http://localhost:8000/docs
🔍 Health Check: http://localhost:8000/health
============================================================
```

### 6. Test the API

Open a new terminal and run:

```bash
# Test health endpoint
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/api/models

# Get a prediction
curl "http://localhost:8000/api/predict/lightgbm/BTC-USD?days=7"
```

Or use the test script:

```bash
./test_api.sh
```

### 7. Access Interactive Documentation

Open in your browser:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test all endpoints interactively!

## 🧪 Testing

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/api/models

# GET prediction
curl "http://localhost:8000/api/predict/lightgbm/BTC-USD?days=7"

# POST prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name":"lightgbm","crypto":"BTC-USD","days":7}'

# Compare models
curl "http://localhost:8000/api/compare/BTC-USD?days=7"
```

### Automated Testing

Run the test script:

```bash
./test_api.sh
```

## 📁 Project Structure

```
project-steve/
├── api/
│   ├── __init__.py          # API package
│   ├── main.py              # FastAPI app entry point
│   ├── models_api.py        # ML prediction endpoints
│   └── database.py          # MongoDB utilities
├── ml_models/
│   ├── models/              # Trained .pkl files
│   ├── predict.py           # ModelPredictor class
│   ├── utils/               # Helper functions
│   └── ...
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
├── .env.example             # Template
├── Procfile                 # Railway deployment
├── railway.json             # Railway config
├── README_API.md            # API documentation
├── DEPLOYMENT.md            # Deployment guide
└── test_api.sh              # Test script
```

## 🔧 Common Issues

### Issue: Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
uvicorn api.main:app --reload --port 8001
```

### Issue: Module Not Found

```bash
# Make sure you're in the right directory
pwd  # Should show: /Users/chanyeong/Desktop/Hackathons/project-steve

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: MongoDB Connection Failed

Don't worry! The API will still work without MongoDB. You'll see:

```
⚠️  Connection attempt 3 failed
❌ Failed to connect to MongoDB after all retries
```

But models will still load and predictions will work.

### Issue: No Models Found

```bash
# Check if model files exist
ls -lh ml_models/models/*.pkl

# If missing, you need to train models first
# or download them from cloud storage
```

## 📚 Next Steps

1. ✅ **Read API Documentation**: `README_API.md`
2. ✅ **Deploy to Railway**: Follow `DEPLOYMENT.md`
3. ✅ **Integrate with Frontend**: Share Railway URL with team
4. ✅ **Monitor Logs**: `railway logs --follow`

## 🎉 You're Ready!

Your QuantBase API is now running locally! Start building your frontend integration.

### Useful Links

- **Local API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 💡 Pro Tips

1. **Keep Server Running**: Use `--reload` flag for auto-restart on code changes
2. **Check Logs**: Watch terminal output for errors
3. **Test Often**: Use `test_api.sh` after making changes
4. **Read Docs**: Interactive docs at `/docs` are your friend

## 🆘 Need Help?

- Check `README_API.md` for endpoint documentation
- Check `DEPLOYMENT.md` for deployment issues
- Review logs in terminal
- Test with `curl` or Postman

---

**Happy Coding!** 🚀
