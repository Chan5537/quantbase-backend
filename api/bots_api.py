"""
Bot Management API Endpoints for QuantBase.

This module implements REST API endpoints for bot creation, customization,
and Claude AI-powered parameter personalization.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator

# Add parent directory to path to import personalize_model_parameters
sys.path.append(str(Path(__file__).parent.parent))

from personalize_model_parameters.claude_processor import PersonalizationProcessor
from personalize_model_parameters.config import PersonalizationConfig
from api.database import get_database, db_manager


# Pydantic models for request/response validation
class BotParameters(BaseModel):
    """Bot trading parameters - matches personalize_model_parameters backend format."""
    window: int = Field(..., ge=5, le=30, description="Rolling lookback period (5-30, typical 10-20)")
    k_sigma: float = Field(..., ge=0.5, le=3.0, description="Standard deviation multiplier (0.5-3.0, typical 1.0-2.0)")
    risk_factor: float = Field(..., ge=0.0, le=1.0, description="Risk appetite (0.0-1.0, 0.0=conservative, 1.0=aggressive)")
    base_trade_size: float = Field(..., ge=0.0001, le=0.01, description="Base trade size (0.0001-0.01, typical 0.0005-0.002)")


class CreateBotRequest(BaseModel):
    """Request model for creating a new bot."""
    name: str = Field(..., min_length=1, max_length=100, description="Bot name")
    description: str = Field(..., min_length=1, max_length=500, description="Bot description")
    parameters: BotParameters = Field(..., description="Bot trading parameters")
    creator_username: str = Field(..., description="Username of bot creator")


class BotResponse(BaseModel):
    """Response model for bot data."""
    id: str
    name: str
    description: str
    image: str
    creator: Dict[str, str]
    parameters: BotParameters
    created_at: str
    updated_at: str
    is_active: bool = True


class PersonalizeBotRequest(BaseModel):
    """Request model for bot personalization."""
    bot_id: str = Field(..., description="Bot ID to personalize")
    user_preferences: Dict[str, Any] = Field(..., description="User preferences for personalization")
    natural_language_input: Optional[str] = Field(None, description="Natural language description")


class PersonalizeBotResponse(BaseModel):
    """Response model for bot personalization."""
    bot_id: str
    original_parameters: BotParameters
    personalized_parameters: BotParameters
    changes_summary: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    expected_performance: Dict[str, Any]
    warnings: List[str]
    recommendations: List[str]
    processing_time_ms: int


class StandalonePersonalizeRequest(BaseModel):
    """Request model for standalone parameter personalization (no bot required)."""
    user_preferences: Dict[str, Any] = Field(..., description="User preferences for personalization")
    natural_language_input: Optional[str] = Field(None, description="Natural language description")


class StandalonePersonalizeResponse(BaseModel):
    """Response model for standalone parameter personalization."""
    personalized_parameters: BotParameters
    changes_summary: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    expected_performance: Dict[str, Any]
    warnings: List[str]
    recommendations: List[str]
    processing_time_ms: int


class ModifyBotRequest(BaseModel):
    """Request model for interactive bot modification."""
    bot_id: str = Field(..., description="Bot ID to modify")
    user_request: str = Field(..., min_length=1, description="Natural language modification request")


class ModifyBotResponse(BaseModel):
    """Response model for bot modification."""
    bot_id: str
    modified_parameters: BotParameters
    changes_applied: List[Dict[str, Any]]
    warnings: List[str]
    processing_time_ms: int


# Initialize router
router = APIRouter(prefix="/api/bots", tags=["Bot Management"])

# Initialize Claude processor (singleton pattern)
_claude_processor: Optional[PersonalizationProcessor] = None


def get_claude_processor() -> PersonalizationProcessor:
    """
    Get or create PersonalizationProcessor instance.
    
    Returns:
        PersonalizationProcessor instance
    """
    global _claude_processor
    if _claude_processor is None:
        config = PersonalizationConfig()
        config.validate()
        _claude_processor = PersonalizationProcessor(config)
    return _claude_processor


def convert_to_backend_params(frontend_params: BotParameters) -> Dict[str, Any]:
    """Convert BotParameters to backend format (returns dict of the same values)."""
    return {
        "window": frontend_params.window,
        "k_sigma": frontend_params.k_sigma,
        "risk_factor": frontend_params.risk_factor,
        "base_trade_size": frontend_params.base_trade_size
    }


def convert_from_backend_params(backend_params: Dict[str, Any]) -> BotParameters:
    """Convert backend parameters to BotParameters."""
    return BotParameters(
        window=backend_params.get("window", 15),
        k_sigma=backend_params.get("k_sigma", 1.5),
        risk_factor=backend_params.get("risk_factor", 0.5),
        base_trade_size=backend_params.get("base_trade_size", 0.002)
    )


@router.post("/create", response_model=BotResponse)
async def create_bot(request: CreateBotRequest, db=Depends(get_database)):
    """
    Create a new trading bot.
    
    Args:
        request: Bot creation request
        db: Database dependency
        
    Returns:
        Created bot information
    """
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot creation requires database connection."
        )
    
    try:
        # Generate bot ID
        bot_id = f"bot-{int(time.time() * 1000)}"
        
        # Create bot document
        bot_doc = {
            "id": bot_id,
            "name": request.name,
            "description": request.description,
            "image": f"https://api.dicebear.com/7.x/shapes/svg?seed={bot_id}",
            "creator": {
                "username": request.creator_username,
                "avatar": f"https://api.dicebear.com/7.x/avataaars/svg?seed={request.creator_username}"
            },
            "parameters": request.parameters.dict(),
            "backend_parameters": convert_to_backend_params(request.parameters),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Save to database
        collection = db['bots']
        result = await collection.insert_one(bot_doc)
        
        if not result.inserted_id:
            raise HTTPException(status_code=500, detail="Failed to create bot")
        
        # Return created bot
        return BotResponse(
            id=bot_id,
            name=request.name,
            description=request.description,
            image=bot_doc["image"],
            creator=bot_doc["creator"],
            parameters=request.parameters,
            created_at=bot_doc["created_at"].isoformat(),
            updated_at=bot_doc["updated_at"].isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create bot: {str(e)}"
        )


@router.get("/", response_model=List[BotResponse])
async def list_bots(
    creator_username: Optional[str] = None,
    db=Depends(get_database)
):
    """
    List all bots, optionally filtered by creator.
    
    Args:
        creator_username: Optional filter by creator username
        db: Database dependency
        
    Returns:
        List of bots
    """
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot listing requires database connection."
        )
    
    try:
        collection = db['bots']
        
        # Build query filter
        query = {"is_active": True}
        if creator_username:
            query["creator.username"] = creator_username
        
        # Get bots
        cursor = collection.find(query).sort("created_at", -1)
        bots = await cursor.to_list(length=None)
        
        # Convert to response format
        bot_responses = []
        for bot in bots:
            bot_responses.append(BotResponse(
                id=bot["id"],
                name=bot["name"],
                description=bot["description"],
                image=bot["image"],
                creator=bot["creator"],
                parameters=BotParameters(**bot["parameters"]),
                created_at=bot["created_at"].isoformat(),
                updated_at=bot["updated_at"].isoformat(),
                is_active=bot["is_active"]
            ))
        
        return bot_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list bots: {str(e)}"
        )


@router.get("/{bot_id}", response_model=BotResponse)
async def get_bot(bot_id: str, db=Depends(get_database)):
    """
    Get a specific bot by ID.
    
    Args:
        bot_id: Bot ID
        db: Database dependency
        
    Returns:
        Bot information
    """
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot retrieval requires database connection."
        )
    
    try:
        collection = db['bots']
        bot = await collection.find_one({"id": bot_id, "is_active": True})
        
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        return BotResponse(
            id=bot["id"],
            name=bot["name"],
            description=bot["description"],
            image=bot["image"],
            creator=bot["creator"],
            parameters=BotParameters(**bot["parameters"]),
            created_at=bot["created_at"].isoformat(),
            updated_at=bot["updated_at"].isoformat(),
            is_active=bot["is_active"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get bot: {str(e)}"
        )


@router.post("/{bot_id}/personalize", response_model=PersonalizeBotResponse)
async def personalize_bot(
    bot_id: str,
    request: PersonalizeBotRequest,
    db=Depends(get_database)
):
    """
    Personalize bot parameters using Claude AI.
    
    Args:
        bot_id: Bot ID to personalize
        request: Personalization request
        db: Database dependency
        
    Returns:
        Personalization results
    """
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot personalization requires database connection."
        )
    
    start_time = time.time()
    
    try:
        # Get bot from database
        collection = db['bots']
        bot = await collection.find_one({"id": bot_id, "is_active": True})
        
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Get Claude processor
        processor = get_claude_processor()
        
        # Prepare user preferences
        user_preferences = request.user_preferences.copy()
        if request.natural_language_input:
            user_preferences["user_request"] = request.natural_language_input
            user_preferences["preferences_description"] = request.natural_language_input
        
        # Get current backend parameters
        current_backend_params = bot.get("backend_parameters", {
            "window": 15,
            "k_sigma": 1.5,
            "risk_factor": 0.5,
            "base_trade_size": 0.01
        })
        
        # Personalize with Claude
        personalization_result = processor.personalize_bot(
            default_parameters=current_backend_params,
            user_preferences=user_preferences
        )
        
        # Extract personalized parameters
        personalized_backend_params = personalization_result.get("modified_parameters", current_backend_params)
        
        # Convert to frontend format
        personalized_frontend_params = convert_from_backend_params(personalized_backend_params)
        
        # Update bot in database
        update_data = {
            "parameters": personalized_frontend_params.dict(),
            "backend_parameters": personalized_backend_params,
            "updated_at": datetime.utcnow()
        }
        
        await collection.update_one(
            {"id": bot_id},
            {"$set": update_data}
        )
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return PersonalizeBotResponse(
            bot_id=bot_id,
            original_parameters=BotParameters(**bot["parameters"]),
            personalized_parameters=personalized_frontend_params,
            changes_summary=personalization_result.get("parameter_changes_summary", []),
            risk_assessment=personalization_result.get("risk_assessment", {}),
            expected_performance=personalization_result.get("expected_performance", {}),
            warnings=personalization_result.get("warnings", []),
            recommendations=personalization_result.get("recommendations", []),
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to personalize bot: {str(e)}"
        )


@router.post("/personalize-standalone", response_model=StandalonePersonalizeResponse)
async def personalize_standalone(
    request: StandalonePersonalizeRequest,
    db=Depends(get_database)
):
    """
    Personalize bot parameters using Claude AI without requiring an existing bot.
    This endpoint is used for pre-creation parameter customization.
    
    Args:
        request: Personalization request
        db: Database dependency (optional)
        
    Returns:
        Personalization results with parameters ready for bot creation
    """
    start_time = time.time()
    
    try:
        # Get Claude processor
        processor = get_claude_processor()
        
        # Prepare user preferences
        user_preferences = request.user_preferences.copy()
        if request.natural_language_input:
            user_preferences["user_request"] = request.natural_language_input
            user_preferences["preferences_description"] = request.natural_language_input
        
        # Use default parameters as starting point
        default_backend_params = {
            "window": 15,
            "k_sigma": 1.5,
            "risk_factor": 0.5,
            "base_trade_size": 0.01
        }
        
        # Personalize with Claude
        personalization_result = processor.personalize_bot(
            default_parameters=default_backend_params,
            user_preferences=user_preferences
        )
        
        # Extract personalized parameters
        personalized_backend_params = personalization_result.get("modified_parameters", default_backend_params)
        
        # Convert to frontend format
        personalized_frontend_params = convert_from_backend_params(personalized_backend_params)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return StandalonePersonalizeResponse(
            personalized_parameters=personalized_frontend_params,
            changes_summary=personalization_result.get("parameter_changes_summary", []),
            risk_assessment=personalization_result.get("risk_assessment", {}),
            expected_performance=personalization_result.get("expected_performance", {}),
            warnings=personalization_result.get("warnings", []),
            recommendations=personalization_result.get("recommendations", []),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to personalize parameters: {str(e)}"
        )


@router.post("/{bot_id}/modify", response_model=ModifyBotResponse)
async def modify_bot(
    bot_id: str,
    request: ModifyBotRequest,
    db=Depends(get_database)
):
    """
    Modify bot parameters interactively using Claude AI.
    
    Args:
        bot_id: Bot ID to modify
        request: Modification request
        db: Database dependency
        
    Returns:
        Modification results
    """
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot modification requires database connection."
        )
    
    start_time = time.time()
    
    try:
        # Get bot from database
        collection = db['bots']
        bot = await collection.find_one({"id": bot_id, "is_active": True})
        
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Get Claude processor
        processor = get_claude_processor()
        
        # Get current backend parameters
        current_backend_params = bot.get("backend_parameters", {
            "window": 15,
            "k_sigma": 1.5,
            "risk_factor": 0.5,
            "base_trade_size": 0.002
        })
        
        # Modify with Claude
        modification_result = processor.modify_parameters_interactively(
            current_parameters=current_backend_params,
            user_request=request.user_request
        )
        
        # Extract modified parameters
        modified_backend_params = modification_result.get("modified_parameters", current_backend_params)
        
        # Convert to frontend format
        modified_frontend_params = convert_from_backend_params(modified_backend_params)
        
        # Update bot in database
        update_data = {
            "parameters": modified_frontend_params.dict(),
            "backend_parameters": modified_backend_params,
            "updated_at": datetime.utcnow()
        }
        
        await collection.update_one(
            {"id": bot_id},
            {"$set": update_data}
        )
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return ModifyBotResponse(
            bot_id=bot_id,
            modified_parameters=modified_frontend_params,
            changes_applied=modification_result.get("changes_applied", []),
            warnings=modification_result.get("warnings", []),
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to modify bot: {str(e)}"
        )


@router.delete("/{bot_id}")
async def delete_bot(bot_id: str, db=Depends(get_database)):
    """
    Soft delete a bot (mark as inactive).
    
    Args:
        bot_id: Bot ID to delete
        db: Database dependency
        
    Returns:
        Deletion confirmation
    """
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot deletion requires database connection."
        )
    
    try:
        collection = db['bots']
        
        # Soft delete by marking as inactive
        result = await collection.update_one(
            {"id": bot_id},
            {
                "$set": {
                    "is_active": False,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        return {"message": "Bot deleted successfully", "bot_id": bot_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete bot: {str(e)}"
        )
