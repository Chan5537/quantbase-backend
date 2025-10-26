"""
MongoDB Database Connection and Utilities for QuantBase API.

This module provides async database connection management using Motor
for FastAPI compatibility.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseManager:
    """
    Manages MongoDB connections and operations for the QuantBase API.
    """
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.connected: bool = False
        
        # Get configuration from environment
        self.mongodb_uri = os.getenv('MONGODB_URI')
        # Hardcoded database names - never use env for these
        self.marketplace_database_name = "bots_db"  # Marketplace bots
        self.custom_database_name = "custom_bots_db"  # User-created custom bots
        self.solana_database_name = "solana_db"  # Trading data/ticks
        self.user_database_name = "user_management"  # User data
        
        # Default database for backward compatibility
        self.database_name = self.marketplace_database_name
        
        if not self.mongodb_uri:
            print("⚠️  MONGODB_URI not found in environment variables")
            print("   Database features will be unavailable")
    
    async def connect(self, max_retries: int = 3) -> bool:
        """
        Connect to MongoDB with retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts
            
        Returns:
            True if connection successful, False otherwise
        """
        if not self.mongodb_uri:
            print("⚠️  Cannot connect: MONGODB_URI not configured")
            return False
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"🔌 Connecting to MongoDB (attempt {attempt}/{max_retries})...")
                
                # Create client with timeout settings
                self.client = AsyncIOMotorClient(
                    self.mongodb_uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                )
                
                # Get database
                self.db = self.client[self.database_name]
                
                # Test connection
                await self.client.admin.command('ping')
                
                self.connected = True
                print(f"✓ Connected to MongoDB")
                print(f"  - Marketplace DB: {self.marketplace_database_name}")
                print(f"  - Custom Bots DB: {self.custom_database_name}")
                print(f"  - Solana DB: {self.solana_database_name}")
                print(f"  - User DB: {self.user_database_name}")
                return True
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                print(f"⚠️  Connection attempt {attempt} failed: {str(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print("❌ Failed to connect to MongoDB after all retries")
                    self.connected = False
                    return False
            except Exception as e:
                print(f"❌ Unexpected error connecting to MongoDB: {str(e)}")
                self.connected = False
                return False
        
        return False
    
    async def disconnect(self):
        """
        Close MongoDB connection.
        """
        if self.client:
            self.client.close()
            self.connected = False
            print("🔌 Disconnected from MongoDB")
    
    async def ping(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if database is reachable, False otherwise
        """
        if not self.connected or not self.client:
            return False
        
        try:
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            print(f"⚠️  Database ping failed: {str(e)}")
            return False
    
    def get_database(self) -> Optional[AsyncIOMotorDatabase]:
        """
        Get the database instance.
        
        Returns:
            Database instance or None if not connected
        """
        return self.db if self.connected else None
    
    def get_marketplace_database(self) -> Optional[AsyncIOMotorDatabase]:
        """
        Get the marketplace database (bots_db) for marketplace bots.
        
        Returns:
            Database instance or None if not connected
        """
        return self.client[self.marketplace_database_name] if self.connected else None
    
    def get_custom_database(self) -> Optional[AsyncIOMotorDatabase]:
        """
        Get the custom bots database (custom_bots_db) for user-created bots.
        
        Returns:
            Database instance or None if not connected
        """
        return self.client[self.custom_database_name] if self.connected else None
    
    def get_solana_database(self) -> Optional[AsyncIOMotorDatabase]:
        """
        Get the solana database (solana_db) for trading data/ticks.
        
        Returns:
            Database instance or None if not connected
        """
        return self.client[self.solana_database_name] if self.connected else None
    
    def get_user_database(self) -> Optional[AsyncIOMotorDatabase]:
        """
        Get the user management database (user_management) for user data.
        
        Returns:
            Database instance or None if not connected
        """
        return self.client[self.user_database_name] if self.connected else None
    
    async def save_prediction(
        self,
        model_name: str,
        crypto: str,
        predictions: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Save prediction results to database.
        
        Args:
            model_name: Name of the model used
            crypto: Cryptocurrency ticker
            predictions: Prediction data
            metadata: Additional metadata
            
        Returns:
            Inserted document ID or None if save failed
        """
        if not self.connected or not self.db:
            print("⚠️  Cannot save prediction: Database not connected")
            return None
        
        try:
            collection = self.db['predictions']
            
            document = {
                'model_name': model_name,
                'crypto': crypto,
                'predictions': predictions,
                'metadata': metadata or {},
                'created_at': datetime.utcnow(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            result = await collection.insert_one(document)
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"⚠️  Failed to save prediction: {str(e)}")
            return None
    
    async def get_prediction_history(
        self,
        model_name: Optional[str] = None,
        crypto: Optional[str] = None,
        limit: int = 10
    ) -> list:
        """
        Retrieve prediction history from database.
        
        Args:
            model_name: Filter by model name (optional)
            crypto: Filter by cryptocurrency (optional)
            limit: Maximum number of results
            
        Returns:
            List of prediction documents
        """
        if not self.connected or not self.db:
            print("⚠️  Cannot get history: Database not connected")
            return []
        
        try:
            collection = self.db['predictions']
            
            # Build query filter
            query = {}
            if model_name:
                query['model_name'] = model_name
            if crypto:
                query['crypto'] = crypto
            
            # Query with sorting and limit
            cursor = collection.find(query).sort('created_at', -1).limit(limit)
            results = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string for JSON serialization
            for result in results:
                result['_id'] = str(result['_id'])
            
            return results
            
        except Exception as e:
            print(f"⚠️  Failed to get prediction history: {str(e)}")
            return []
    
    async def save_model_metadata(
        self,
        model_name: str,
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Save model metadata (performance metrics, etc.).
        
        Args:
            model_name: Name of the model
            metadata: Model metadata
            
        Returns:
            Updated document ID or None if save failed
        """
        if not self.connected or not self.db:
            return None
        
        try:
            collection = self.db['models_metadata']
            
            document = {
                'model_name': model_name,
                'metadata': metadata,
                'updated_at': datetime.utcnow(),
            }
            
            # Upsert: update if exists, insert if not
            result = await collection.update_one(
                {'model_name': model_name},
                {'$set': document},
                upsert=True
            )
            
            return str(result.upserted_id) if result.upserted_id else "updated"
            
        except Exception as e:
            print(f"⚠️  Failed to save model metadata: {str(e)}")
            return None
    
    async def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve model metadata.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metadata or None if not found
        """
        if not self.connected or not self.db:
            return None
        
        try:
            collection = self.db['models_metadata']
            result = await collection.find_one({'model_name': model_name})
            
            if result:
                result['_id'] = str(result['_id'])
            
            return result
            
        except Exception as e:
            print(f"⚠️  Failed to get model metadata: {str(e)}")
            return None


# Global database manager instance
db_manager = DatabaseManager()


async def get_database() -> Optional[AsyncIOMotorDatabase]:
    """
    Dependency injection function for FastAPI routes.
    Returns the default database (backward compatibility).
    
    Returns:
        Database instance or None
    """
    return db_manager.get_database()


async def get_marketplace_database() -> Optional[AsyncIOMotorDatabase]:
    """
    Dependency injection function for marketplace database (bots_db).
    
    Returns:
        Database instance or None
    """
    return db_manager.get_marketplace_database()


async def get_custom_database() -> Optional[AsyncIOMotorDatabase]:
    """
    Dependency injection function for custom bots database (custom_bots_db).
    
    Returns:
        Database instance or None
    """
    return db_manager.get_custom_database()


async def get_solana_database() -> Optional[AsyncIOMotorDatabase]:
    """
    Dependency injection function for solana database (solana_db).
    
    Returns:
        Database instance or None
    """
    return db_manager.get_solana_database()


async def get_user_database() -> Optional[AsyncIOMotorDatabase]:
    """
    Dependency injection function for user database (user_management).
    
    Returns:
        Database instance or None
    """
    return db_manager.get_user_database()


async def ping_database() -> bool:
    """
    Test database connection (for health checks).
    
    Returns:
        True if database is healthy, False otherwise
    """
    return await db_manager.ping()


# Startup and shutdown functions for FastAPI lifespan
async def connect_to_mongo():
    """
    Connect to MongoDB on application startup.
    """
    await db_manager.connect()


async def close_mongo_connection():
    """
    Close MongoDB connection on application shutdown.
    """
    await db_manager.disconnect()
