#!/bin/bash

# QuantBase API - Local Testing Script
# This script helps test the API endpoints locally before deployment

echo "🧪 QuantBase API Testing Script"
echo "================================"
echo ""

# Check if server is running
echo "📡 Checking if API server is running..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API server is running!"
else
    echo "❌ API server is not running."
    echo "   Please start it first with: uvicorn api.main:app --reload --port 8000"
    exit 1
fi

echo ""
echo "Running API tests..."
echo "===================="
echo ""

# Test 1: Root endpoint
echo "1️⃣  Testing root endpoint..."
curl -s http://localhost:8000/ | python3 -m json.tool
echo ""

# Test 2: Health check
echo "2️⃣  Testing health check..."
curl -s http://localhost:8000/health | python3 -m json.tool
echo ""

# Test 3: List models
echo "3️⃣  Testing list models..."
curl -s http://localhost:8000/api/models | python3 -m json.tool
echo ""

# Test 4: Get prediction (GET)
echo "4️⃣  Testing GET prediction..."
curl -s "http://localhost:8000/api/predict/lightgbm/BTC-USD?days=7" | python3 -m json.tool
echo ""

# Test 5: Post prediction
echo "5️⃣  Testing POST prediction..."
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name":"lightgbm","crypto":"BTC-USD","days":7}' | python3 -m json.tool
echo ""

# Test 6: Compare models
echo "6️⃣  Testing model comparison..."
curl -s "http://localhost:8000/api/compare/BTC-USD?days=7" | python3 -m json.tool
echo ""

echo "✅ All tests completed!"
echo ""
echo "📚 View interactive docs at: http://localhost:8000/docs"
