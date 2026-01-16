import threading
import time
import pickle
import logging
from typing import Optional, Union
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np

try:
    import redis
except ImportError:
    redis = None

from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    vector: np.ndarray
    expires_at: float


class TTLCache:
    """
    Hybrid Thread-safe TTL + LRU cache.
    Supports local memory (OrderedDict) and Redis backend.
    
    Performance: O(1) for get/put/touch operations (Local).
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        
        # Local Cache
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Redis Client
        self._redis_client = None
        if settings.redis.enabled:
            if redis:
                try:
                    self._redis_client = redis.Redis(
                        host=settings.redis.host,
                        port=settings.redis.port,
                        db=settings.redis.db,
                        password=settings.redis.password,
                        socket_timeout=0.2 # Fast fail
                    )
                    self._redis_client.ping()
                    logger.info("Redis cache initialized successfully.")
                except Exception as e:
                    logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory cache.")
                    self._redis_client = None
            else:
                logger.warning("Redis enabled in config but 'redis' package not installed.")

        # Metrics
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached vector. O(1). Checks Local -> Redis."""
        # 1. Check Local Cache
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                # Check expiration
                if time.time() > entry.expires_at:
                    del self._cache[key]
                else:
                    # Move to end (LRU touch)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return entry.vector

        # 2. Check Redis (if enabled and miss in local)
        if self._redis_client:
            try:
                redis_key = f"{settings.redis.key_prefix}{key}"
                data = self._redis_client.get(redis_key)
                if data:
                    vector = pickle.loads(data)
                    # Populate local cache for next time (L1 cache)
                    self.put_local(key, vector) 
                    self._hits += 1
                    return vector
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        self._misses += 1
        return None
    
    def put(self, key: str, vector: np.ndarray):
        """Put vector into cache. Updates Local + Redis."""
        # 1. Update Local
        self.put_local(key, vector)
        
        # 2. Update Redis
        if self._redis_client:
            try:
                redis_key = f"{settings.redis.key_prefix}{key}"
                data = pickle.dumps(vector)
                # Set with TTL
                self._redis_client.setex(
                    redis_key, 
                    settings.redis.ttl_seconds, 
                    data
                )
            except Exception as e:
                logger.warning(f"Redis put error: {e}")

    def put_local(self, key: str, vector: np.ndarray):
        """Helper to update local cache only."""
        with self._lock:
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]
            
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Remove oldest (first)
            
            # Add new entry at end
            self._cache[key] = CacheEntry(
                vector=vector,
                expires_at=time.time() + self._ttl
            )
    
    def invalidate(self, key: str):
        """Remove specific key."""
        with self._lock:
            self._cache.pop(key, None)
            
        if self._redis_client:
            try:
                redis_key = f"{settings.redis.key_prefix}{key}"
                self._redis_client.delete(redis_key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
    
    def clear(self):
        """Clear entire cache (Local Only for safety, or flushdb?). 
        For safety, we only clear local context unless explicitly requested."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def cleanup_expired(self):
        """Remove expired entries (call periodically). Local only."""
        with self._lock:
            now = time.time()
            # Collect expired keys (can't modify during iteration)
            expired = [k for k, v in self._cache.items() if now > v.expires_at]
            for key in expired:
                del self._cache[key]
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "redis_enabled": self._redis_client is not None
            }
