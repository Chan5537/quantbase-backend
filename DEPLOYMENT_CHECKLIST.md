# QuantBase API - Deployment Checklist

Track your progress deploying the QuantBase FastAPI backend to Railway.app.

## 📋 Pre-Deployment Checklist

### Local Setup & Testing

- [ ] **Python environment setup**
  - [ ] Python 3.8+ installed
  - [ ] Virtual environment created (`python3 -m venv venv`)
  - [ ] Virtual environment activated

- [ ] **Dependencies installed**
  - [ ] `pip install -r requirements.txt` completed successfully
  - [ ] No installation errors

- [ ] **Environment variables configured**
  - [ ] `.env` file created with MongoDB URI
  - [ ] `MONGODB_URI` set correctly
  - [ ] `DATABASE_NAME` set to "quantbase"
  - [ ] `ALLOWED_ORIGINS` configured

- [ ] **Local server runs**
  - [ ] `uvicorn api.main:app --reload --port 8000` starts successfully
  - [ ] No import errors
  - [ ] Server shows "QuantBase API is ready!" message

- [ ] **ML Models loaded**
  - [ ] Model files exist in `ml_models/models/*.pkl`
  - [ ] Server displays model count (should be 8)
  - [ ] No "models not found" errors

- [ ] **MongoDB connection** (Optional but recommended)
  - [ ] Server connects to MongoDB Atlas
  - [ ] "Connected to MongoDB" message appears
  - [ ] Database ping successful

- [ ] **Endpoints tested locally**
  - [ ] `curl http://localhost:8000/` returns welcome message
  - [ ] `curl http://localhost:8000/health` returns healthy status
  - [ ] `curl http://localhost:8000/api/models` lists models
  - [ ] `curl http://localhost:8000/api/predict/lightgbm/BTC-USD?days=7` returns predictions
  - [ ] POST `/api/predict` works
  - [ ] `/api/compare/BTC-USD` works

- [ ] **Interactive docs accessible**
  - [ ] http://localhost:8000/docs loads Swagger UI
  - [ ] All endpoints visible in docs
  - [ ] Can test endpoints interactively

- [ ] **Test script works**
  - [ ] `./test_api.sh` runs without errors
  - [ ] All test endpoints return valid JSON

### Code Quality

- [ ] **Files created**
  - [ ] `api/__init__.py`
  - [ ] `api/main.py`
  - [ ] `api/models_api.py`
  - [ ] `api/database.py`
  - [ ] `ml_models/predict.py`
  - [ ] `requirements.txt`
  - [ ] `.env` (git-ignored)
  - [ ] `.env.example`
  - [ ] `Procfile`
  - [ ] `railway.json`
  - [ ] `README_API.md`
  - [ ] `DEPLOYMENT.md`
  - [ ] `QUICKSTART.md`

- [ ] **Git configuration**
  - [ ] `.gitignore` updated
  - [ ] `.env` is git-ignored
  - [ ] `.pkl` model files exception added
  - [ ] Railway directory ignored

- [ ] **Code committed**
  - [ ] All files staged: `git add .`
  - [ ] Committed: `git commit -m "Add FastAPI backend"`
  - [ ] Pushed: `git push origin feature/ml-forecasting-models`

## 🚀 Railway Deployment Checklist

### Railway Setup

- [ ] **Railway account**
  - [ ] Account created at railway.app
  - [ ] Email verified

- [ ] **Railway CLI installed**
  - [ ] `npm install -g @railway/cli` or `brew install railway`
  - [ ] `railway --version` works

- [ ] **Railway login**
  - [ ] `railway login` completed
  - [ ] Browser authentication successful

### Project Initialization

- [ ] **Railway project created**
  - [ ] `railway init` executed
  - [ ] Project named: "quantbase-backend"
  - [ ] Project appears in Railway dashboard

- [ ] **GitHub connection** (Recommended)
  - [ ] Railway connected to GitHub
  - [ ] Repository `3arii/project-steve` linked
  - [ ] Auto-deploy enabled on branch

### Environment Variables

- [ ] **Production environment variables set**
  - [ ] `MONGODB_URI` set
  - [ ] `DATABASE_NAME` set to "quantbase"
  - [ ] `ENVIRONMENT` set to "production"
  - [ ] `API_TITLE` set
  - [ ] `API_VERSION` set
  - [ ] `ALLOWED_ORIGINS` set (initial)

Commands used:
```bash
railway variables set MONGODB_URI="[your-uri]"
railway variables set DATABASE_NAME="quantbase"
railway variables set ENVIRONMENT="production"
railway variables set ALLOWED_ORIGINS="http://localhost:3000,https://*.vercel.app"
```

### Deployment

- [ ] **Initial deployment**
  - [ ] `railway up` executed
  - [ ] Build completed successfully
  - [ ] No build errors
  - [ ] Deployment successful

- [ ] **Deployment URL obtained**
  - [ ] `railway status` shows URL
  - [ ] URL noted: `https://_____________.railway.app`
  - [ ] Custom domain generated (optional)

- [ ] **Logs checked**
  - [ ] `railway logs` shows startup messages
  - [ ] "QuantBase API is ready!" visible
  - [ ] MongoDB connection successful
  - [ ] Models loaded successfully
  - [ ] No critical errors

## ✅ Post-Deployment Verification

### Production API Testing

- [ ] **Health check**
  - [ ] `curl https://[your-url].railway.app/health` returns healthy
  - [ ] Database status: connected
  - [ ] Models status: available
  - [ ] Model count: 8 (or expected number)

- [ ] **Basic endpoints**
  - [ ] Root `/` returns welcome message
  - [ ] `/api/models` lists all models
  - [ ] `/api/predict/lightgbm/BTC-USD?days=7` returns predictions
  - [ ] POST `/api/predict` works
  - [ ] `/api/compare/BTC-USD` works

- [ ] **Interactive documentation**
  - [ ] https://[your-url].railway.app/docs loads
  - [ ] All endpoints listed
  - [ ] Can test endpoints from browser

- [ ] **Response times acceptable**
  - [ ] Health check < 1s
  - [ ] Model listing < 2s
  - [ ] Predictions < 5s

### Database Verification

- [ ] **MongoDB connection in production**
  - [ ] Logs show "Connected to MongoDB"
  - [ ] Database ping successful
  - [ ] No connection timeout errors

- [ ] **MongoDB Atlas configuration**
  - [ ] IP whitelist set to `0.0.0.0/0` (allow all)
  - [ ] Connection string correct
  - [ ] Database user has proper permissions

### Error Handling

- [ ] **Error responses work**
  - [ ] 404 for invalid endpoints
  - [ ] 400 for invalid parameters (e.g., `days=100`)
  - [ ] 404 for non-existent models
  - [ ] Proper error JSON format

- [ ] **CORS configured**
  - [ ] No CORS errors from localhost:3000
  - [ ] Preflight requests working
  - [ ] Headers properly set

## 🎨 Frontend Integration

- [ ] **Documentation shared**
  - [ ] `README_API.md` shared with frontend team
  - [ ] Railway URL provided
  - [ ] Example code provided

- [ ] **Frontend environment configured**
  - [ ] Frontend `.env.local` has API URL
  - [ ] `NEXT_PUBLIC_API_URL=https://[your-url].railway.app`

- [ ] **Frontend can connect**
  - [ ] Frontend successfully fetches from `/api/models`
  - [ ] Frontend successfully gets predictions
  - [ ] No CORS errors in browser console

- [ ] **CORS updated for production**
  - [ ] Frontend Vercel URL added to `ALLOWED_ORIGINS`
  - [ ] `railway variables set ALLOWED_ORIGINS="..."`
  - [ ] Railway redeployed with new CORS

## 📊 Monitoring & Maintenance

- [ ] **Logging**
  - [ ] Can view logs: `railway logs`
  - [ ] Can follow logs: `railway logs --follow`
  - [ ] Logs are readable and helpful

- [ ] **Error tracking** (Optional)
  - [ ] Sentry or similar tool configured
  - [ ] Error notifications set up

- [ ] **Uptime monitoring** (Optional)
  - [ ] UptimeRobot or similar configured
  - [ ] Health check monitored

- [ ] **Performance**
  - [ ] API response times monitored
  - [ ] No memory leaks
  - [ ] CPU usage acceptable

## 📝 Documentation

- [ ] **README updated**
  - [ ] Production URL added to main README
  - [ ] Deployment status noted

- [ ] **API documentation complete**
  - [ ] All endpoints documented
  - [ ] Example requests/responses provided
  - [ ] Frontend integration guide complete

- [ ] **Team notified**
  - [ ] Frontend team has Railway URL
  - [ ] Backend deployment confirmed
  - [ ] Integration instructions provided

## 🔄 Continuous Deployment

- [ ] **Auto-deploy configured**
  - [ ] GitHub connected to Railway
  - [ ] Push to main branch auto-deploys
  - [ ] Deployment notifications enabled

- [ ] **Tested auto-deploy**
  - [ ] Made a small change
  - [ ] Committed and pushed
  - [ ] Railway auto-deployed
  - [ ] New version live

## 🎉 Launch Checklist

### Pre-Launch

- [ ] All endpoints tested in production
- [ ] Frontend successfully integrated
- [ ] CORS configured for production domain
- [ ] MongoDB connection stable
- [ ] All models loading correctly
- [ ] Error handling tested
- [ ] Documentation complete
- [ ] Team trained on API usage

### Launch Day

- [ ] Monitor logs for errors
- [ ] Check response times
- [ ] Verify database connections
- [ ] Test from multiple locations
- [ ] Frontend team confirmed integration works

### Post-Launch

- [ ] Monitor error rates
- [ ] Check API usage patterns
- [ ] Optimize slow endpoints
- [ ] Gather user feedback
- [ ] Plan next features

## 📞 Emergency Contacts

**If something breaks:**

1. Check Railway logs: `railway logs`
2. Check Railway status: [status.railway.app](https://status.railway.app)
3. Rollback if needed: Railway dashboard → Deployments → Previous version
4. Contact team leads

## 🏆 Success Criteria

Your deployment is successful when:

- ✅ Health endpoint returns "healthy"
- ✅ All 8 models loaded
- ✅ MongoDB connected
- ✅ Predictions returning correct data
- ✅ Frontend can successfully fetch predictions
- ✅ No CORS errors
- ✅ Response times < 5 seconds
- ✅ Documentation complete
- ✅ Team can use the API

---

## 📊 Progress Summary

**Completed:** _____ / 100+ items

**Status:** 
- [ ] Not Started
- [ ] In Progress
- [ ] Testing
- [ ] Deployed
- [ ] Live in Production

**Last Updated:** _____________

**Deployed By:** _____________

**Production URL:** _____________

---

**Good luck with your deployment! 🚀**

Once all items are checked, your QuantBase API will be production-ready!
