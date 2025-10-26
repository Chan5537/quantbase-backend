# QuantBase API - Railway Deployment Guide

This guide walks you through deploying the QuantBase FastAPI backend to Railway.app.

## 📋 Prerequisites

- [x] All API files created
- [x] ML models trained and saved in `ml_models/models/`
- [x] MongoDB Atlas connection string ready
- [x] Railway account created at [railway.app](https://railway.app)
- [x] GitHub repository pushed

## 🚀 Deployment Steps

### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
```

Or with Homebrew (macOS):
```bash
brew install railway
```

### Step 2: Login to Railway

```bash
railway login
```

This will open your browser for authentication.

### Step 3: Initialize Railway Project

Navigate to your project directory:

```bash
cd /Users/chanyeong/Desktop/Hackathons/project-steve
```

Initialize Railway:

```bash
railway init
```

- Choose **"Create new project"**
- Name it: **quantbase-backend**

### Step 4: Link to GitHub (Optional but Recommended)

For automatic deployments:

1. Go to [railway.app/dashboard](https://railway.app/dashboard)
2. Click on your project
3. Go to **Settings** → **Connect to GitHub**
4. Select repository: **3arii/project-steve**
5. Enable auto-deploy on branch: **main** or **feature/ml-forecasting-models**

### Step 5: Set Environment Variables

Set all required environment variables:

```bash
# MongoDB URI
railway variables set MONGODB_URI="mongodb+srv://denizlapsekili_db_user:4nUh2jZIkgYEAfWM@solana-data.cx4xp7h.mongodb.net/?appName=solana-data"

# Database name
railway variables set DATABASE_NAME="quantbase"

# Environment
railway variables set ENVIRONMENT="production"

# API Configuration
railway variables set API_TITLE="QuantBase API"
railway variables set API_VERSION="1.0.0"

# CORS (update after frontend deployment)
railway variables set ALLOWED_ORIGINS="http://localhost:3000,https://*.vercel.app"
```

### Step 6: Deploy

Deploy your application:

```bash
railway up
```

Railway will:
1. ✅ Detect Python project
2. ✅ Install dependencies from `requirements.txt`
3. ✅ Execute the `Procfile` command
4. ✅ Assign a public URL

Wait for deployment to complete (usually 2-5 minutes).

### Step 7: Get Your Deployment URL

Check deployment status:

```bash
railway status
```

Open in browser:

```bash
railway open
```

Your API will be available at: `https://[project-name].up.railway.app`

### Step 8: Generate Custom Domain (Optional)

```bash
railway domain
```

This creates a custom domain: `https://quantbase-backend.railway.app`

### Step 9: View Logs

Monitor your application:

```bash
railway logs
```

Follow logs in real-time:

```bash
railway logs --follow
```

## ✅ Post-Deployment Verification

### 1. Test Health Endpoint

```bash
BACKEND_URL="https://your-project.railway.app"
curl $BACKEND_URL/health
```

Expected response:
```json
{
  "status": "healthy",
  "components": {
    "database": {"status": "connected", "healthy": true},
    "ml_models": {"status": "available", "count": 8, "healthy": true}
  }
}
```

### 2. Test Models Endpoint

```bash
curl $BACKEND_URL/api/models
```

### 3. Test Prediction

```bash
curl "$BACKEND_URL/api/predict/lightgbm/BTC-USD?days=7"
```

### 4. Check Interactive Documentation

Visit in browser:
- Swagger UI: `https://your-project.railway.app/docs`
- ReDoc: `https://your-project.railway.app/redoc`

### 5. Verify MongoDB Connection

Check logs for connection message:

```bash
railway logs | grep "MongoDB"
```

Should see: `✓ Connected to MongoDB: quantbase`

## 🔄 Update CORS for Frontend

After deploying your Next.js frontend to Vercel, update CORS:

```bash
railway variables set ALLOWED_ORIGINS="http://localhost:3000,https://quantbase-frontend.vercel.app,https://*.vercel.app"
```

Railway will automatically redeploy with new settings.

## 🔧 Troubleshooting

### Issue: Models Not Found

**Problem:** API shows "No models available"

**Solution:**
1. Check if `.pkl` files are in `ml_models/models/`
2. Verify `.gitignore` allows `!ml_models/models/*.pkl`
3. Commit and push model files:
   ```bash
   git add ml_models/models/*.pkl -f
   git commit -m "Add trained models"
   git push
   ```

If models are too large for Git, use Git LFS:
```bash
git lfs track "*.pkl"
git add .gitattributes
git add ml_models/models/*.pkl
git commit -m "Add models with LFS"
git push
```

### Issue: MongoDB Connection Failed

**Problem:** Database shows "disconnected"

**Solution:**
1. Check MongoDB Atlas IP whitelist: Should be `0.0.0.0/0` (allow all)
2. Verify connection string is correct
3. Test connection locally first
4. Check Railway logs: `railway logs | grep "MongoDB"`

### Issue: Module Not Found

**Problem:** `ModuleNotFoundError: No module named 'ml_models'`

**Solution:**
1. Ensure `ml_models/` directory exists in repository
2. Check `.gitignore` doesn't exclude it
3. Verify `ml_models/__init__.py` exists
4. Rebuild: `railway up --detach`

### Issue: Port Already in Use (Local)

**Problem:** `Address already in use`

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uvicorn api.main:app --port 8001
```

### Issue: CORS Errors from Frontend

**Problem:** Browser console shows CORS errors

**Solution:**
1. Add your frontend URL to `ALLOWED_ORIGINS`
2. Ensure URL doesn't have trailing slash
3. Redeploy Railway after updating

```bash
railway variables set ALLOWED_ORIGINS="https://your-frontend.vercel.app,http://localhost:3000"
```

## 📊 Monitoring & Logs

### View Real-Time Logs

```bash
railway logs --follow
```

### Check Service Status

```bash
railway status
```

### Open Railway Dashboard

```bash
railway open
```

From dashboard you can:
- View metrics (CPU, Memory, Network)
- See deployment history
- Manage environment variables
- View build logs
- Configure custom domains

## 🔄 Continuous Deployment

Once connected to GitHub:

1. **Make changes** to code
2. **Commit and push** to GitHub:
   ```bash
   git add .
   git commit -m "Update API endpoints"
   git push origin main
   ```
3. **Railway automatically deploys** - Monitor via:
   ```bash
   railway logs --follow
   ```

## 🎯 Production Checklist

Before going live:

- [ ] All environment variables set
- [ ] MongoDB connection working
- [ ] All model files uploaded
- [ ] Health check returns "healthy"
- [ ] All endpoints tested
- [ ] CORS configured for production frontend
- [ ] Error handling tested
- [ ] Documentation reviewed
- [ ] Frontend team has Railway URL
- [ ] Rate limiting configured (if needed)
- [ ] Monitoring set up

## 📞 Support

If you encounter issues:

1. Check Railway logs: `railway logs`
2. Check Railway status page: [status.railway.app](https://status.railway.app)
3. Railway Discord: [discord.gg/railway](https://discord.gg/railway)
4. Railway Docs: [docs.railway.app](https://docs.railway.app)

## 🎉 Success!

Once deployed successfully:

1. ✅ Share Railway URL with frontend team
2. ✅ Update README with production URL
3. ✅ Test all endpoints from frontend
4. ✅ Monitor initial traffic and errors

Your QuantBase API is now live! 🚀

## 📝 Next Steps

1. **Frontend Integration**: Share `README_API.md` with frontend team
2. **Monitoring**: Set up error tracking (Sentry, LogRocket)
3. **Analytics**: Add usage tracking
4. **Security**: Add API authentication
5. **Optimization**: Enable caching for predictions
6. **Scaling**: Configure auto-scaling if needed

---

**Railway Project**: `quantbase-backend`  
**Repository**: `3arii/project-steve`  
**Branch**: `feature/ml-forecasting-models`  
**Last Updated**: 2024-10-25
