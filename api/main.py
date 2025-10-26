"""
QuantBase API - FastAPI Application Entry Point.

This is the main FastAPI application that serves ML model predictions
for cryptocurrency trading algorithms.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.models_api import router as models_router
from api.database import connect_to_mongo, close_mongo_connection, ping_database
from ml_models.predict import ModelPredictor

# Load environment variables
load_dotenv()


# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██████╗  ██████╗║
║  ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔══██╗██╔════╝║
║  ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██████╔╝██║     ║
║  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██╔══██╗██║     ║
║  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ██████╔╝╚██████╗║
║   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═════╝  ╚═════╝║
║                                                               ║
║              Cryptocurrency Trading ML API                   ║
║                    Powered by FastAPI                        ║
╚═══════════════════════════════════════════════════════════════╝
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    print(BANNER)
    print(f"🚀 QuantBase API Starting...")
    print(f"📝 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"🔧 API Version: {os.getenv('API_VERSION', '1.0.0')}")
    print("="*60)
    
    # Connect to MongoDB
    print("\n🗄️  Initializing Database Connection...")
    await connect_to_mongo()
    
    # Initialize and check ML models
    print("\n🤖 Initializing ML Models...")
    try:
        models_dir = Path(__file__).parent.parent / 'ml_models' / 'models'
        predictor = ModelPredictor(models_dir=str(models_dir))
        models = predictor.list_available_models()
        
        if models:
            print(f"✓ Found {len(models)} trained models:")
            for model in models:
                print(f"  • {model['name']} ({model['size_mb']:.2f} MB)")
        else:
            print("⚠️  No trained models found in ml_models/models/")
            print("   Please ensure model .pkl files are present")
    except Exception as e:
        print(f"⚠️  Error loading models: {str(e)}")
    
    print("\n" + "="*60)
    print("✅ QuantBase API is ready!")
    print(f"📚 Documentation: http://localhost:{os.getenv('PORT', '8000')}/docs")
    print(f"🔍 Health Check: http://localhost:{os.getenv('PORT', '8000')}/health")
    print("="*60 + "\n")
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down QuantBase API...")
    await close_mongo_connection()
    print("👋 Goodbye!\n")


# Initialize FastAPI app
app = FastAPI(
    title=os.getenv('API_TITLE', 'QuantBase API'),
    description="""
    🚀 **QuantBase Cryptocurrency Trading ML API**
    
    This API provides machine learning-powered price predictions for cryptocurrencies
    using multiple trained forecasting models.
    
    ## Features
    
    * 📊 Multiple ML models (LightGBM, XGBoost, Random Forest, Neural Networks)
    * 🔮 Short to medium-term price forecasting (1-30 days)
    * 📈 Confidence intervals for predictions
    * 🔄 Model comparison capabilities
    * 💾 Prediction history tracking (with MongoDB)
    
    ## Available Endpoints
    
    * `/api/models` - List all available models
    * `/api/predict` - Generate predictions (POST)
    * `/api/predict/{model}/{crypto}` - Generate predictions (GET)
    * `/api/compare/{crypto}` - Compare all models
    * `/health` - API health check
    
    ## Support
    
    For questions or issues, please contact the QuantBase team.
    """,
    version=os.getenv('API_VERSION', '1.0.0'),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# CORS Configuration
origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
origins = [origin.strip() for origin in origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests with timing.
    """
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request
    print(
        f"📥 {request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Time: {process_time:.3f}s"
    )
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors with detailed messages.
    """
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "body": exc.body,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with consistent format.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """
    Handle internal server errors.
    """
    print(f"❌ Internal Server Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "path": str(request.url.path),
        },
    )


# Include routers
app.include_router(models_router)


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint - basic health check.
    
    Returns:
        Welcome message with API status
    """
    return {
        "message": "Welcome to QuantBase API",
        "status": "online",
        "version": os.getenv('API_VERSION', '1.0.0'),
        "docs": "/docs",
        "health": "/health",
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint.
    
    Returns:
        Health status of API and its dependencies
    """
    # Check database connection
    db_status = "disconnected"
    db_healthy = False
    
    try:
        db_healthy = await ping_database()
        db_status = "connected" if db_healthy else "disconnected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Check models availability
    models_status = "unavailable"
    models_count = 0
    
    try:
        models_dir = Path(__file__).parent.parent / 'ml_models' / 'models'
        predictor = ModelPredictor(models_dir=str(models_dir))
        models = predictor.list_available_models()
        models_count = len(models)
        models_status = "available" if models_count > 0 else "no models found"
    except Exception as e:
        models_status = f"error: {str(e)}"
    
    # Overall health
    healthy = db_healthy or models_count > 0  # API is healthy if either works
    
    return {
        "status": "healthy" if healthy else "degraded",
        "timestamp": time.time(),
        "environment": os.getenv('ENVIRONMENT', 'development'),
        "version": os.getenv('API_VERSION', '1.0.0'),
        "components": {
            "database": {
                "status": db_status,
                "healthy": db_healthy,
                "required": False,  # Database is optional
            },
            "ml_models": {
                "status": models_status,
                "count": models_count,
                "healthy": models_count > 0,
                "required": True,  # Models are required
            },
        },
    }


# 404 handler
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    """
    Catch-all route for undefined endpoints.
    """
    raise HTTPException(
        status_code=404,
        detail=f"Endpoint '/{full_path}' not found. Visit /docs for API documentation."
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('PORT', '8000'))
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
