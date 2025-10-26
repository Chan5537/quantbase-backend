"""
Bot Management API Endpoints for QuantBase.

This module implements REST API endpoints for bot creation, customization,
and Claude AI-powered parameter personalization.
"""

import sys
import logging
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
from api.database import (
    get_database, 
    db_manager,
    get_marketplace_database,
    get_custom_database
)


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
    description: str = Field("", max_length=500, description="Bot description (optional)")
    parameters: BotParameters = Field(..., description="Bot trading parameters")
    creator_username: str = Field(..., description="Username of bot creator")
    model_type: Optional[str] = Field("momentum", description="Trading model type (momentum or mean_reversion) - hidden from user")


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


def validate_and_clamp_parameters(params: Dict[str, Any]) -> BotParameters:
    """Validate and clamp parameters to ensure they're within valid ranges."""
    # Clamp values to ensure they're within valid ranges
    window = params.get("window", 15)
    window_original = window
    window = max(5, min(30, int(window)))  # Clamp between 5 and 30
    
    k_sigma = params.get("k_sigma", 1.5)
    k_sigma_original = k_sigma
    k_sigma = max(0.5, min(3.0, float(k_sigma)))  # Clamp between 0.5 and 3.0
    
    risk_factor = params.get("risk_factor", 0.5)
    risk_factor_original = risk_factor
    risk_factor = max(0.0, min(1.0, float(risk_factor)))  # Clamp between 0.0 and 1.0
    
    base_trade_size = params.get("base_trade_size", 0.002)
    base_trade_size_original = base_trade_size
    base_trade_size = max(0.0001, min(0.01, float(base_trade_size)))  # Clamp between 0.0001 and 0.01
    
    # Log any clamping that occurred
    if window_original != window or k_sigma_original != k_sigma or risk_factor_original != risk_factor or base_trade_size_original != base_trade_size:
        logging.warning(
            f"Clamped parameter values: window={window_original}->{window}, "
            f"k_sigma={k_sigma_original}->{k_sigma}, "
            f"risk_factor={risk_factor_original}->{risk_factor}, "
            f"base_trade_size={base_trade_size_original}->{base_trade_size}"
        )
    
    return BotParameters(
        window=window,
        k_sigma=k_sigma,
        risk_factor=risk_factor,
        base_trade_size=base_trade_size
    )


@router.post("/create", response_model=BotResponse)
async def create_bot(request: CreateBotRequest, db=Depends(get_custom_database)):
    """
    Create a new trading bot in custom_bots_db (user-created bots).
    
    Args:
        request: Bot creation request
        db: Database dependency (custom_bots_db)
        
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
            "parameters": request.parameters.dict(),  # Single source of truth - no need for backend_parameters
            "model_type": request.model_type,  # Use provided model_type or default to "momentum"
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Save to custom_bots_db database
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
    marketplace_db=Depends(get_marketplace_database),
    custom_db=Depends(get_custom_database)
):
    """
    List all bots from both marketplace (bots_db) and custom (custom_bots_db) databases.
    Optionally filtered by creator.
    
    Args:
        creator_username: Optional filter by creator username
        marketplace_db: Marketplace database (bots_db)
        custom_db: Custom bots database (custom_bots_db)
        
    Returns:
        List of bots from both databases
    """
    if marketplace_db is None and custom_db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot listing requires database connection."
        )
    
    try:
        bot_responses = []
        
        # Build query filter
        query = {"is_active": True}
        if creator_username:
            query["creator.username"] = creator_username
        
        # Query marketplace bots (bots_db)
        if marketplace_db:
            marketplace_collection = marketplace_db['bots']
            cursor = marketplace_collection.find(query).sort("created_at", -1)
            marketplace_bots = await cursor.to_list(length=None)
            
            for bot in marketplace_bots:
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
        
        # Query custom bots (custom_bots_db)
        if custom_db:
            custom_collection = custom_db['bots']
            cursor = custom_collection.find(query).sort("created_at", -1)
            custom_bots = await cursor.to_list(length=None)
            
            for bot in custom_bots:
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
        
        # Sort by created_at descending
        bot_responses.sort(key=lambda x: x.created_at, reverse=True)
        
        return bot_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list bots: {str(e)}"
        )


@router.get("/{bot_id}", response_model=BotResponse)
async def get_bot(
    bot_id: str,
    marketplace_db=Depends(get_marketplace_database),
    custom_db=Depends(get_custom_database)
):
    """
    Get a specific bot by ID from either marketplace or custom database.
    
    Args:
        bot_id: Bot ID
        marketplace_db: Marketplace database (bots_db)
        custom_db: Custom bots database (custom_bots_db)
        
    Returns:
        Bot information
    """
    if marketplace_db is None and custom_db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot retrieval requires database connection."
        )
    
    try:
        bot = None
        
        # First try marketplace database
        if marketplace_db:
            collection = marketplace_db['bots']
            bot = await collection.find_one({"id": bot_id, "is_active": True})
        
        # If not found, try custom database
        if not bot and custom_db:
            collection = custom_db['bots']
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
    marketplace_db=Depends(get_marketplace_database),
    custom_db=Depends(get_custom_database)
):
    """
    Personalize bot parameters using Claude AI.
    
    Args:
        bot_id: Bot ID to personalize
        request: Personalization request
        marketplace_db: Marketplace database (bots_db)
        custom_db: Custom bots database (custom_bots_db)
        
    Returns:
        Personalization results
    """
    if marketplace_db is None and custom_db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot personalization requires database connection."
        )
    
    start_time = time.time()
    
    try:
        # Get bot from database - try custom first, then marketplace
        bot = None
        collection = None
        db = None
        
        if custom_db:
            collection = custom_db['bots']
            bot = await collection.find_one({"id": bot_id, "is_active": True})
            if bot:
                db = custom_db
        
        if not bot and marketplace_db:
            collection = marketplace_db['bots']
            bot = await collection.find_one({"id": bot_id, "is_active": True})
            if bot:
                db = marketplace_db
        
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Get Claude processor
        processor = get_claude_processor()
        
        # Prepare user preferences
        user_preferences = request.user_preferences.copy()
        if request.natural_language_input:
            user_preferences["user_request"] = request.natural_language_input
            user_preferences["preferences_description"] = request.natural_language_input
        
        # Get current parameters from bot
        current_params = bot.get("parameters", {
            "window": 15,
            "k_sigma": 1.5,
            "risk_factor": 0.5,
            "base_trade_size": 0.01
        })
        
        # Personalize with Claude
        personalization_result = processor.personalize_bot(
            default_parameters=current_params,
            user_preferences=user_preferences
        )
        
        # Extract personalized parameters
        personalized_params = personalization_result.get("modified_parameters", current_params)
        
        # Extract model type (if provided by Claude)
        model_type = personalization_result.get("model_type", "momentum")  # Default to momentum
        
        # Convert to validated frontend format (clamping applied)
        personalized_frontend_params = validate_and_clamp_parameters(personalized_params)
        
        # Update bot in the correct database
        update_data = {
            "parameters": personalized_frontend_params.dict(),  # Clamped/validated values
            "model_type": model_type,
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
        default_params = {
            "window": 15,
            "k_sigma": 1.5,
            "risk_factor": 0.5,
            "base_trade_size": 0.01
        }
        
        # Personalize with Claude
        personalization_result = processor.personalize_bot(
            default_parameters=default_params,
            user_preferences=user_preferences
        )
        
        # Extract personalized parameters
        personalized_params = personalization_result.get("modified_parameters", default_params)
        
        # Extract model type (if provided by Claude)
        model_type = personalization_result.get("model_type", "momentum")  # Default to momentum
        
        # Convert to validated frontend format (clamping applied)
        personalized_frontend_params = validate_and_clamp_parameters(personalized_params)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Note: model_type is stored internally but not exposed in the response
        # This is intentional to keep it hidden from the user
        
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
    marketplace_db=Depends(get_marketplace_database),
    custom_db=Depends(get_custom_database)
):
    """
    Modify bot parameters interactively using Claude AI.
    
    Args:
        bot_id: Bot ID to modify
        request: Modification request
        marketplace_db: Marketplace database (bots_db)
        custom_db: Custom bots database (custom_bots_db)
        
    Returns:
        Modification results
    """
    if marketplace_db is None and custom_db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot modification requires database connection."
        )
    
    start_time = time.time()
    
    try:
        # Get bot from database - try custom first, then marketplace
        bot = None
        collection = None
        
        if custom_db:
            collection = custom_db['bots']
            bot = await collection.find_one({"id": bot_id, "is_active": True})
        
        if not bot and marketplace_db:
            collection = marketplace_db['bots']
            bot = await collection.find_one({"id": bot_id, "is_active": True})
        
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Get Claude processor
        processor = get_claude_processor()
        
        # Get current parameters from bot
        current_params = bot.get("parameters", {
            "window": 15,
            "k_sigma": 1.5,
            "risk_factor": 0.5,
            "base_trade_size": 0.002
        })
        
        # Modify with Claude
        modification_result = processor.modify_parameters_interactively(
            current_parameters=current_params,
            user_request=request.user_request
        )
        
        # Extract modified parameters
        modified_params = modification_result.get("modified_parameters", current_params)
        
        # Extract model type (if changed by Claude, otherwise keep existing)
        if "model_type" in modification_result and modification_result["model_type"] != "unchanged":
            model_type = modification_result.get("model_type", bot.get("model_type", "momentum"))
        else:
            model_type = bot.get("model_type", "momentum")  # Keep existing or default to momentum
        
        # Convert to validated frontend format (clamping applied)
        modified_frontend_params = validate_and_clamp_parameters(modified_params)
        
        # Update bot in database
        update_data = {
            "parameters": modified_frontend_params.dict(),  # Clamped/validated values
            "model_type": model_type,
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
async def delete_bot(
    bot_id: str,
    marketplace_db=Depends(get_marketplace_database),
    custom_db=Depends(get_custom_database)
):
    """
    Soft delete a bot (mark as inactive).
    Checks both custom and marketplace databases.
    
    Args:
        bot_id: Bot ID to delete
        marketplace_db: Marketplace database (bots_db)
        custom_db: Custom bots database (custom_bots_db)
        
    Returns:
        Deletion confirmation
    """
    if marketplace_db is None and custom_db is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available. Bot deletion requires database connection."
        )
    
    try:
        result = None
        
        # Try custom database first
        if custom_db:
            collection = custom_db['bots']
            result = await collection.update_one(
                {"id": bot_id},
                {
                    "$set": {
                        "is_active": False,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        # If not found, try marketplace database
        if (not result or result.modified_count == 0) and marketplace_db:
            collection = marketplace_db['bots']
            result = await collection.update_one(
                {"id": bot_id},
                {
                    "$set": {
                        "is_active": False,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        if not result or result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        return {"message": "Bot deleted successfully", "bot_id": bot_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete bot: {str(e)}"
        )
