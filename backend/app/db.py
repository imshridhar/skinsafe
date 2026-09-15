"""SkinSafe AI - MongoDB Database Connection & Resilient Pooling Manager

Provides thread-safe connection pooling, automatic index creation,
fast-failover caching, and seamless memory fallbacks when MongoDB is offline.
"""

import logging
import os
import time
from typing import Optional
from flask import current_app
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database

logger = logging.getLogger("backend.db")

_mongo_client: Optional[MongoClient] = None
_last_connection_attempt: float = 0.0
_is_known_offline: bool = False
_COOLDOWN_SECONDS: float = 15.0  # Time to wait before retrying offline MongoDB


def get_mongo_client() -> Optional[MongoClient]:
    """Returns a pooled MongoClient singleton instance with non-blocking failover."""
    global _mongo_client, _last_connection_attempt, _is_known_offline
    now = time.time()

    # Return healthy existing connection
    if _mongo_client is not None:
        try:
            _mongo_client.admin.command("ping")
            _is_known_offline = False
            return _mongo_client
        except Exception:
            _mongo_client = None
            _is_known_offline = True
            _last_connection_attempt = now

    # If known to be offline, respect cooldown to prevent blocking requests with socket timeouts
    if _is_known_offline and (now - _last_connection_attempt < _COOLDOWN_SECONDS):
        return None

    _last_connection_attempt = now

    uri = None
    try:
        uri = current_app.config.get("MONGODB_URI")
    except RuntimeError:
        pass

    if not uri:
        uri = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/skin_lesion_ai")

    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=400,
            connectTimeoutMS=400,
            socketTimeoutMS=1000,
            maxPoolSize=50,
            minPoolSize=2,
            retryWrites=True
        )
        client.admin.command("ping")
        _mongo_client = client
        _is_known_offline = False
        logger.info("Successfully connected to MongoDB at %s", uri)
        _init_indexes(_mongo_client.get_default_database())
        return _mongo_client
    except Exception as e:
        if not _is_known_offline:
            logger.info("MongoDB service is currently offline on %s. Operating in resilient in-memory fallback mode.", uri)
        _is_known_offline = True
        _mongo_client = None
        return None


def get_database() -> Optional[Database]:
    """Returns the active MongoDB database instance or None if offline."""
    client = get_mongo_client()
    if client is not None:
        try:
            return client.get_default_database()
        except Exception as e:
            logger.error("Failed to retrieve default database: %s", e)
            return None
    return None


def _init_indexes(db: Database) -> None:
    """Ensures optimized unique and compound indexes for fast user-isolated queries."""
    try:
        # Users Collection: unique email
        db.users.create_index([("email", ASCENDING)], unique=True)

        # Scans Collection: compound index on user_email + timestamp for isolated audit queries
        db.scans.create_index([("user_email", ASCENDING), ("timestamp", DESCENDING)])
        db.scans.create_index([("id", ASCENDING)], unique=True)
        db.scans.create_index([("patientId", ASCENDING)])
        logger.info("MongoDB indexes verified successfully.")
    except Exception as e:
        logger.warning("Index initialization note: %s", e)
