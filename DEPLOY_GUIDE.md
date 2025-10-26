# 🚀 QuantBase Deployment Guide

## Deployment Options

You have **TWO options** for deployment:
- **Option A: Web Dashboard** (Recommended - No CLI needed) ⭐
- **Option B: CLI Deployment** (For advanced users)

---

## ⭐ OPTION A: Web Dashboard Deployment (Easiest)

### Step 1: Deploy Backend to Railway

1. **Go to Railway**: https://railway.app
2. **Login with GitHub**
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose: `3arii/project-steve`
6. Select branch: `feature/ml-forecasting-models`
7. **Add Environment Variables**:
   - Click your project → **"Variables"** tab
   - Add these:
     ```
     MONGODB_URI=mongodb+srv://denizlapsekili_db_user:4nUh2jZIkgYEAfWM@solana-data.cx4xp7h.mongodb.net/?appName=solana-data
     DATABASE_NAME=quantbase
     ENVIRONMENT=production
     ALLOWED_ORIGINS=http://localhost:3000,https://*.vercel.app
     ```
8. Railway will **auto-deploy** (takes 2-3 minutes)
9. **Generate Domain**:
   - Click **"Settings"** → **"Generate Domain"**
   - Save your URL: `https://project-steve-production.up.railway.app`

### Step 2: Test Backend

Open in browser:
```
https://your-railway-url.up.railway.app/docs
```

Test health:
```
https://your-railway-url.up.railway.app/health
```

### Step 3: Deploy Frontend to Vercel

1. **Go to Vercel**: https://vercel.com
2. **Login with GitHub**
3. Click **"Add New..." → "Project"**
4. **Import** your Next.js frontend repository
5. Configure:
   - **Framework**: Next.js (auto-detected)
   - **Root Directory**: `./`
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `.next` (default)
6. **Add Environment Variable**:
   ```
   NEXT_PUBLIC_API_URL=https://your-railway-url.up.railway.app
   ```
   *(Use the Railway URL from Step 1)*
7. Click **"Deploy"**
8. Save your URL: `https://your-project.vercel.app`

### Step 4: Update CORS in Railway

1. Go back to **Railway dashboard**
2. Your project → **"Variables"**
3. **Edit** `ALLOWED_ORIGINS`:
   ```
   ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend.vercel.app,https://*.vercel.app
   ```
4. Railway will **auto-redeploy**

### Step 5: Test Integration

1. Open your Vercel frontend URL
2. Check browser console for errors
3. Test API calls from frontend

---

## 🔧 OPTION B: CLI Deployment

### Prerequisites

```bash
# Install Railway CLI
npm install -g @railway/cli

# Install Vercel CLI
npm install -g vercel
```

### Deploy Backend (Railway CLI)

```bash
# Navigate to project
cd /Users/chanyeong/Desktop/Hackathons/project-steve

# Login to Railway
railway login

# Initialize project
railway init
# Choose: "Create new project"
# Name: "quantbase-backend"

# Link to GitHub repo (optional)
railway link

# Set environment variables
railway variables set MONGODB_URI="mongodb+srv://denizlapsekili_db_user:4nUh2jZIkgYEAfWM@solana-data.cx4xp7h.mongodb.net/?appName=solana-data"
railway variables set DATABASE_NAME="quantbase"
railway variables set ENVIRONMENT="production"
railway variables set ALLOWED_ORIGINS="http://localhost:3000,https://*.vercel.app"

# Deploy
railway up

# Get deployment URL
railway status

# Generate public domain
railway domain

# View logs
railway logs
```

### Deploy Frontend (Vercel CLI)

```bash
# Navigate to your Next.js frontend project
cd /path/to/your/frontend

# Login to Vercel
vercel login

# Deploy (first time - will ask questions)
vercel

# Or deploy to production directly
vercel --prod

# Add environment variable
vercel env add NEXT_PUBLIC_API_URL

# When prompted, enter: https://your-railway-url.up.railway.app
```

### Update CORS (Railway CLI)

```bash
# After getting Vercel URL, update CORS
railway variables set ALLOWED_ORIGINS="http://localhost:3000,https://your-frontend.vercel.app,https://*.vercel.app"
```

---

## 📋 Post-Deployment Checklist

### Backend Verification

- [ ] Railway deployment successful
- [ ] Domain generated and accessible
- [ ] `/docs` endpoint shows Swagger UI
- [ ] `/health` returns healthy status
- [ ] `/api/models` lists available models
- [ ] MongoDB connection working (check logs)
- [ ] No errors in Railway logs

### Frontend Verification

- [ ] Vercel deployment successful
- [ ] Custom domain configured (optional)
- [ ] Environment variable `NEXT_PUBLIC_API_URL` set
- [ ] Frontend can call backend API
- [ ] No CORS errors in browser console
- [ ] API data displays correctly

### Integration Testing

```bash
# Test backend from terminal
curl https://your-railway-url.up.railway.app/health
curl https://your-railway-url.up.railway.app/api/models
curl "https://your-railway-url.up.railway.app/api/predict/lightgbm/BTC-USD?days=7"
```

---

## 🔄 Continuous Deployment

### Enable Auto-Deploy on Railway

1. Railway Dashboard → Your Project
2. **Settings** → **GitHub**
3. Connect repository: `3arii/project-steve`
4. Enable **"Auto-deploy"** on branch: `main`
5. Now every push to `main` will auto-deploy!

### Enable Auto-Deploy on Vercel

1. Vercel Dashboard → Your Project
2. **Settings** → **Git**
3. Production Branch: `main`
4. Auto-deploy is **enabled by default**
5. Every push triggers rebuild!

---

## 🐛 Troubleshooting

### Railway Issues

**Problem**: Build fails
- Check `requirements.txt` is present
- Check `Procfile` is present
- View logs: `railway logs`

**Problem**: App crashes on startup
- Check environment variables are set
- Check MongoDB URI is correct
- View startup logs in Railway dashboard

**Problem**: Models not found
- Ensure `ml_models/models/*.pkl` files are committed to git
- Check file size (Railway has limits)
- If files too large, use Git LFS

### Vercel Issues

**Problem**: Environment variable not working
- Ensure `NEXT_PUBLIC_` prefix for client-side vars
- Redeploy after adding env vars
- Check Vercel Dashboard → Settings → Environment Variables

**Problem**: CORS errors
- Verify `ALLOWED_ORIGINS` includes Vercel URL
- Redeploy Railway after updating CORS
- Check exact URL format (no trailing slash)

### MongoDB Issues

**Problem**: Connection timeout
- Check MongoDB Atlas IP whitelist: allow `0.0.0.0/0`
- Verify connection string is correct
- Test connection locally first

---

## 📞 URLs to Save

After deployment, save these:

```
Backend (Railway): https://_____________________.up.railway.app
Frontend (Vercel): https://_____________________.vercel.app
API Docs: https://_____________________.up.railway.app/docs
MongoDB: mongodb+srv://solana-data.cx4xp7h.mongodb.net
```

---

## 🎯 Quick Deploy Commands

```bash
# Backend (Railway - CLI)
cd project-steve
railway login
railway up

# Frontend (Vercel - CLI)
cd your-frontend
vercel login
vercel --prod

# Or use web dashboards - much easier! 🌐
```

---

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app)
- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Next.js Deployment](https://nextjs.org/docs/deployment)

---

**Recommendation**: Use **Option A (Web Dashboard)** for fastest deployment! 🚀
