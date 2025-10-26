#!/bin/bash

# QuantBase Railway Deployment Script

echo "🚀 Deploying QuantBase API to Railway..."
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null
then
    echo "❌ Railway CLI is not installed"
    echo "📦 Installing Railway CLI..."
    npm install -g @railway/cli
fi

echo "✓ Railway CLI is installed"
echo ""

# Check if logged in
if ! railway whoami &> /dev/null
then
    echo "🔐 Please login to Railway..."
    railway login
fi

echo "✓ Logged in to Railway"
echo ""

# Link to project (if not already linked)
if [ ! -f ".railway.json" ]; then
    echo "🔗 Linking to Railway project..."
    railway link
    echo ""
fi

# Deploy to Railway
echo "📦 Starting deployment..."
echo ""
railway up

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Set environment variables:"
echo "   railway variables set MONGODB_URI='your-mongodb-uri'"
echo "   railway variables set CLAUDE_API_KEY='your-claude-key'"
echo "   railway variables set ALLOWED_ORIGINS='https://your-frontend.vercel.app'"
echo ""
echo "2. View logs:"
echo "   railway logs"
echo ""
echo "3. Open your app:"
echo "   railway open"
echo ""
echo "📚 For more help, see: DEPLOY_TO_RAILWAY.md"
