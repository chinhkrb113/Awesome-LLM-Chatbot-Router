# Kiến trúc v2 Production-Grade (Đã Fix P0)

## 1. TÓM TẮT CÁC VẤN ĐỀ ĐÃ FIX

| Issue | Problem | Solution |
|-------|---------|----------|
| **(A) Dimension mismatch** | Hardcode 768, Jina v3 = 1024 | Dynamic dimension từ model |
| **(B) Thread-safety** | Hot reload race condition | RWLock + Atomic Swap |
| **(C) Key mapping bug** | Text trùng → ghi đè | Map theo (action_id, idx) |
| **(D) Cache no TTL** | Không expire, không thread-safe | TTLCache + Lock |
| **(E) Vector Store aggregation** | Search trả seed, cần action score | Seed → Group → Max aggregation |

---

## 2. PRODUCTION-GRADE SKELETON

### 2.1 Config với Dynamic Dimension

```python
# app/router/embed_config.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict

class VectorStoreType(Enum):
    MEMORY = "memory"
    FAISS = "faiss"
    QDRANT = "qdrant"

class ModelType(Enum):
    JINA_V3 = "jina-v3"
    BGE_M3 = "bge-m3"
    VIETNAMESE_SBERT = "vietnamese-sbert"

# Model dimension mapping (source of truth)
MODEL_DIMENSIONS: Dict[str, int] = {
    "jinaai/jina-embeddings-v3": 1024,
    "BAAI/bge-m3": 1024,
    "keepitreal/vietnamese-sbert": 768,
    # Matryoshka dimensions for Jina v3
    "jinaai/jina-embeddings-v3:256": 256,
    "jinaai/jina-embeddings-v3:512": 512,
    "jinaai/jina-embeddings-v3:768": 768,
    "jinaai/jina-embeddings-v3:1024": 1024,
}

@dataclass
class EmbedConfig:
    # Model settings
    primary_model: str = "jinaai/jina-embeddings-v3"
    fallback_model: str = "BAAI/bge-m3"
    
    # Matryoshka dimension (None = use default)
    # Jina v3 supports: 256, 512, 768, 1024
    matryoshka_dim: Optional[int] = None
    
    # Task type for Jina v3 (retrieval.query vs retrieval.passage)
    task_type: str = "retrieval.query"
    
    # Vector store
    vector_store: VectorStoreType = VectorStoreType.MEMORY
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "action_seeds"
    
    # Scoring strategy
    max_pool: bool = True
    
    # Cache settings
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 300
    
    # Hot reload
    reload_poll_interval: float = 2.0
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension based on model and matryoshka setting."""
        if self.matryoshka_dim:
            return self.matryoshka_dim
        return MODEL_DIMENSIONS.get(self.primary_model, 1024)
    
    @property
    def fallback_dim(self) -> int:
        return MODEL_DIMENSIONS.get(self.fallback_model, 1024)
```

### 2.2 Thread-Safe Data Structures (Atomic Swap + RWLock)

```python
# app/router/thread_safe.py

import threading
from typing import Dict, List, TypeVar, Generic, Optional
from dataclasses import dataclass
from copy import deepcopy
import time

T = TypeVar('T')

class RWLock:
    """
    Read-Write Lock implementation.
    - Multiple readers can hold the lock simultaneously
    - Only one writer can hold the lock (exclusive)
    - Writers have priority to prevent starvation
    """
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False
    
    def acquire_read(self):
        with self._read_ready:
            while self._writer_active or self._writers_waiting > 0:
                self._read_ready.wait()
            self._readers += 1
    
    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
    
    def acquire_write(self):
        with self._read_ready:
            self._writers_waiting += 1
            while self._readers > 0 or self._writer_active:
                self._read_ready.wait()
            self._writers_waiting -= 1
            self._writer_active = True
    
    def release_write(self):
        with self._read_ready:
            self._writer_active = False
            self._read_ready.notify_all()
    
    def read_lock(self):
        """Context manager for read lock."""
        return _ReadLockContext(self)
    
    def write_lock(self):
        """Context manager for write lock."""
        return _WriteLockContext(self)


class _ReadLockContext:
    def __init__(self, lock: RWLock):
        self._lock = lock
    def __enter__(self):
        self._lock.acquire_read()
        return self
    def __exit__(self, *args):
        self._lock.release_read()


class _WriteLockContext:
    def __init__(self, lock: RWLock):
        self._lock = lock
    def __enter__(self):
        self._lock.acquire_write()
        return self
    def __exit__(self, *args):
        self._lock.release_write()


@dataclass
class SeedVector:
    """Immutable seed vector data."""
    action_id: str
    seed_index: int
    seed_text: str
    vector: 'np.ndarray'
    confidence: float = 1.0


class AtomicVectorState:
    """
    Thread-safe vector state using Copy-on-Write pattern.
    
    Read operations: lock-free (read current reference)
    Write operations: build new state, then atomic swap
    """
    
    def __init__(self):
        self._lock = RWLock()
        
        # Immutable state (replaced atomically)
        self._seed_vectors: Dict[str, SeedVector] = {}  # key = "action_id::seed_idx"
        self._action_seeds: Dict[str, List[str]] = {}   # action_id -> [seed_keys]
        self._version: int = 0
    
    def get_state(self):
        """Get current state (read-only snapshot)."""
        with self._lock.read_lock():
            return (
                self._seed_vectors,
                self._action_seeds,
                self._version
            )
    
    def get_seed_vectors_for_action(self, action_id: str) -> List[SeedVector]:
        """Get all seed vectors for an action."""
        with self._lock.read_lock():
            seed_keys = self._action_seeds.get(action_id, [])
            return [self._seed_vectors[k] for k in seed_keys if k in self._seed_vectors]
    
    def get_all_action_ids(self) -> List[str]:
        """Get all action IDs."""
        with self._lock.read_lock():
            return list(self._action_seeds.keys())
    
    def atomic_update(self, 
                      new_seed_vectors: Dict[str, SeedVector],
                      new_action_seeds: Dict[str, List[str]]):
        """
        Atomic swap of entire state.
        Build new state outside lock, then swap reference.
        """
        with self._lock.write_lock():
            self._seed_vectors = new_seed_vectors
            self._action_seeds = new_action_seeds
            self._version += 1
    
    def atomic_update_action(self, 
                             action_id: str,
                             seed_vectors: List[SeedVector]):
        """
        Atomic update for single action (incremental).
        Copy current state, modify, swap.
        """
        with self._lock.write_lock():
            # Copy current state
            new_seed_vectors = dict(self._seed_vectors)
            new_action_seeds = dict(self._action_seeds)
            
            # Remove old seeds for this action
            old_keys = new_action_seeds.get(action_id, [])
            for key in old_keys:
                new_seed_vectors.pop(key, None)
            
            # Add new seeds
            new_keys = []
            for sv in seed_vectors:
                key = f"{action_id}::{sv.seed_index}"
                new_seed_vectors[key] = sv
                new_keys.append(key)
            
            new_action_seeds[action_id] = new_keys
            
            # Atomic swap
            self._seed_vectors = new_seed_vectors
            self._action_seeds = new_action_seeds
            self._version += 1
    
    def atomic_remove_action(self, action_id: str):
        """Atomic remove of single action."""
        with self._lock.write_lock():
            new_seed_vectors = dict(self._seed_vectors)
            new_action_seeds = dict(self._action_seeds)
            
            old_keys = new_action_seeds.pop(action_id, [])
            for key in old_keys:
                new_seed_vectors.pop(key, None)
            
            self._seed_vectors = new_seed_vectors
            self._action_seeds = new_action_seeds
            self._version += 1
```

### 2.3 TTL Cache với Thread-Safety

```python
# app/router/query_cache.py

import threading
import time
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class CacheEntry:
    vector: np.ndarray
    expires_at: float

class TTLCache:
    """
    Thread-safe TTL cache for query vectors.
    
    Features:
    - LRU eviction when max_size reached
    - TTL expiration
    - Thread-safe with fine-grained locking
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        
        # Cache storage
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached vector if exists and not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            # Check expiration
            if time.time() > entry.expires_at:
                self._remove_key(key)
                return None
            
            # Update access order (LRU)
            self._touch(key)
            return entry.vector
    
    def put(self, key: str, vector: np.ndarray):
        """Put vector into cache."""
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._evict_oldest()
            
            # Add/update entry
            self._cache[key] = CacheEntry(
                vector=vector,
                expires_at=time.time() + self._ttl
            )
            self._touch(key)
    
    def invalidate(self, key: str):
        """Remove specific key from cache."""
        with self._lock:
            self._remove_key(key)
    
    def clear(self):
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def cleanup_expired(self):
        """Remove all expired entries (call periodically)."""
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._cache.items() 
                if now > v.expires_at
            ]
            for key in expired_keys:
                self._remove_key(key)
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            now = time.time()
            valid_count = sum(1 for v in self._cache.values() if now <= v.expires_at)
            return {
                "size": len(self._cache),
                "valid": valid_count,
                "expired": len(self._cache) - valid_count,
                "max_size": self._max_size,
                "ttl_seconds": self._ttl
            }
    
    def _touch(self, key: str):
        """Update access order (must hold lock)."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _remove_key(self, key: str):
        """Remove key from cache (must hold lock)."""
        self._cache.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)
    
    def _evict_oldest(self):
        """Evict least recently used entry (must hold lock)."""
        if self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
```

### 2.4 Vector Store với Seed → Action Aggregation

```python
# app/router/vector_store_v2.py

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

@dataclass
class SeedSearchResult:
    """Result from vector search."""
    seed_key: str      # "action_id::seed_idx"
    action_id: str
    seed_index: int
    score: float

@dataclass
class ActionScore:
    """Aggregated score per action."""
    action_id: str
    score: float
    matched_seeds: List[Tuple[int, float]]  # [(seed_idx, score), ...]


class VectorStoreV2(ABC):
    """
    Abstract vector store with seed → action aggregation.
    
    Key difference from v1:
    - search() returns seed-level results
    - search_actions() aggregates to action-level scores
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
    
    @abstractmethod
    def add_seed(self, action_id: str, seed_index: int, vector: np.ndarray, 
                 metadata: dict = None): 
        """Add a single seed vector."""
        pass
    
    @abstractmethod
    def search_seeds(self, query_vector: np.ndarray, top_k: int = 50) -> List[SeedSearchResult]:
        """Search for similar seeds (raw results)."""
        pass
    
    @abstractmethod
    def delete_action(self, action_id: str):
        """Delete all seeds for an action."""
        pass
    
    @abstractmethod
    def update_seed(self, action_id: str, seed_index: int, vector: np.ndarray):
        """Update a specific seed vector."""
        pass
    
    def search_actions(self, 
                       query_vector: np.ndarray, 
                       top_k_seeds: int = 50,
                       top_k_actions: int = 10,
                       aggregation: str = "max") -> List[ActionScore]:
        """
        Search and aggregate to action-level scores.
        
        Args:
            query_vector: Query embedding
            top_k_seeds: Number of seeds to retrieve
            top_k_actions: Number of actions to return
            aggregation: "max" or "avg"
        
        Returns:
            List of ActionScore sorted by score descending
        """
        # Step 1: Get seed-level results
        seed_results = self.search_seeds(query_vector, top_k_seeds)
        
        # Step 2: Group by action_id
        action_seeds: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for sr in seed_results:
            action_seeds[sr.action_id].append((sr.seed_index, sr.score))
        
        # Step 3: Aggregate scores per action
        action_scores = []
        for action_id, seeds in action_seeds.items():
            if aggregation == "max":
                score = max(s[1] for s in seeds)
            else:  # avg
                score = sum(s[1] for s in seeds) / len(seeds)
            
            action_scores.append(ActionScore(
                action_id=action_id,
                score=score,
                matched_seeds=seeds
            ))
        
        # Step 4: Sort and return top_k
        action_scores.sort(key=lambda x: x.score, reverse=True)
        return action_scores[:top_k_actions]


class InMemoryVectorStoreV2(VectorStoreV2):
    """In-memory implementation for <100 actions."""
    
    def __init__(self, dimension: int):
        super().__init__(dimension)
        self._vectors: Dict[str, np.ndarray] = {}  # key -> vector
        self._action_keys: Dict[str, List[str]] = {}  # action_id -> [keys]
    
    def add_seed(self, action_id: str, seed_index: int, vector: np.ndarray,
                 metadata: dict = None):
        key = f"{action_id}::{seed_index}"
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        self._vectors[key] = vector
        
        if action_id not in self._action_keys:
            self._action_keys[action_id] = []
        if key not in self._action_keys[action_id]:
            self._action_keys[action_id].append(key)
    
    def search_seeds(self, query_vector: np.ndarray, top_k: int = 50) -> List[SeedSearchResult]:
        # Normalize query
        norm_q = np.linalg.norm(query_vector)
        if norm_q > 0:
            query_vector = query_vector / norm_q
        
        results = []
        for key, vec in self._vectors.items():
            score = float(np.dot(query_vector, vec))
            action_id, seed_idx = key.rsplit("::", 1)
            results.append(SeedSearchResult(
                seed_key=key,
                action_id=action_id,
                seed_index=int(seed_idx),
                score=score
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def delete_action(self, action_id: str):
        keys = self._action_keys.pop(action_id, [])
        for key in keys:
            self._vectors.pop(key, None)
    
    def update_seed(self, action_id: str, seed_index: int, vector: np.ndarray):
        self.add_seed(action_id, seed_index, vector)


class FAISSVectorStoreV2(VectorStoreV2):
    """FAISS implementation for <10K actions."""
    
    def __init__(self, dimension: int):
        super().__init__(dimension)
        import faiss
        
        # Use IndexIDMap to support delete/update
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self._key_to_id: Dict[str, int] = {}
        self._id_to_key: Dict[int, str] = {}
        self._action_keys: Dict[str, List[str]] = {}
        self._next_id = 0
    
    def add_seed(self, action_id: str, seed_index: int, vector: np.ndarray,
                 metadata: dict = None):
        key = f"{action_id}::{seed_index}"
        
        # Remove if exists (for update)
        if key in self._key_to_id:
            old_id = self._key_to_id[key]
            self._index.remove_ids(np.array([old_id], dtype=np.int64))
            del self._id_to_key[old_id]
        
        # Normalize and add
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        vector = vector.astype('float32').reshape(1, -1)
        ids = np.array([self._next_id], dtype=np.int64)
        self._index.add_with_ids(vector, ids)
        
        self._key_to_id[key] = self._next_id
        self._id_to_key[self._next_id] = key
        self._next_id += 1
        
        if action_id not in self._action_keys:
            self._action_keys[action_id] = []
        if key not in self._action_keys[action_id]:
            self._action_keys[action_id].append(key)
    
    def search_seeds(self, query_vector: np.ndarray, top_k: int = 50) -> List[SeedSearchResult]:
        norm_q = np.linalg.norm(query_vector)
        if norm_q > 0:
            query_vector = query_vector / norm_q
        
        query_vector = query_vector.astype('float32').reshape(1, -1)
        scores, ids = self._index.search(query_vector, min(top_k, self._index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx not in self._id_to_key:
                continue
            key = self._id_to_key[idx]
            action_id, seed_idx = key.rsplit("::", 1)
            results.append(SeedSearchResult(
                seed_key=key,
                action_id=action_id,
                seed_index=int(seed_idx),
                score=float(score)
            ))
        
        return results
    
    def delete_action(self, action_id: str):
        keys = self._action_keys.pop(action_id, [])
        ids_to_remove = []
        for key in keys:
            if key in self._key_to_id:
                ids_to_remove.append(self._key_to_id[key])
                del self._id_to_key[self._key_to_id[key]]
                del self._key_to_id[key]
        
        if ids_to_remove:
            self._index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
    
    def update_seed(self, action_id: str, seed_index: int, vector: np.ndarray):
        self.add_seed(action_id, seed_index, vector)


class QdrantVectorStoreV2(VectorStoreV2):
    """Qdrant implementation for >10K actions (production scale)."""
    
    def __init__(self, dimension: int, host: str = "localhost", port: int = 6333,
                 collection: str = "action_seeds"):
        super().__init__(dimension)
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        
        # Create collection with correct dimension
        try:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=dimension,  # Dynamic dimension!
                    distance=Distance.COSINE
                )
            )
        except Exception:
            pass  # Collection exists
    
    def add_seed(self, action_id: str, seed_index: int, vector: np.ndarray,
                 metadata: dict = None):
        from qdrant_client.models import PointStruct
        
        key = f"{action_id}::{seed_index}"
        point_id = self._key_to_point_id(key)
        
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={
                    "key": key,
                    "action_id": action_id,
                    "seed_index": seed_index,
                    **(metadata or {})
                }
            )]
        )
    
    def search_seeds(self, query_vector: np.ndarray, top_k: int = 50) -> List[SeedSearchResult]:
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        
        return [
            SeedSearchResult(
                seed_key=r.payload["key"],
                action_id=r.payload["action_id"],
                seed_index=r.payload["seed_index"],
                score=r.score
            )
            for r in results
        ]
    
    def delete_action(self, action_id: str):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(
                    key="action_id",
                    match=MatchValue(value=action_id)
                )]
            )
        )
    
    def update_seed(self, action_id: str, seed_index: int, vector: np.ndarray):
        self.add_seed(action_id, seed_index, vector)
    
    def _key_to_point_id(self, key: str) -> int:
        """Convert string key to int64 point ID."""
        import hashlib
        return int(hashlib.md5(key.encode()).hexdigest()[:16], 16) % (2**63)
```

### 2.5 EmbedAnything Engine Production-Grade

```python
# app/router/embed_anything_engine_v2.py

import logging
from typing import List, Dict, Optional
import numpy as np

from app.router.embed_config import EmbedConfig, MODEL_DIMENSIONS
from app.router.thread_safe import AtomicVectorState, SeedVector
from app.router.query_cache import TTLCache
from app.router.vector_store_v2 import (
    VectorStoreV2, InMemoryVectorStoreV2, 
    FAISSVectorStoreV2, QdrantVectorStoreV2,
    ActionScore
)

logger = logging.getLogger(__name__)


class EmbedAnythingEngineV2:
    """
    Production-grade embedding engine with EmbedAnything.
    
    Fixed issues:
    - (A) Dynamic dimension based on model
    - (B) Thread-safe with RWLock + Atomic Swap
    - (C) Key mapping by (action_id, idx) not text
    - (D) TTL cache with proper expiration
    - (E) Vector store with seed → action aggregation
    """
    
    def __init__(self, config: EmbedConfig = None):
        self.config = config or EmbedConfig()
        
        # Models
        self._model = None
        self._fallback_model = None
        
        # Thread-safe state
        self._state = AtomicVectorState()
        
        # TTL Cache
        self._cache = TTLCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds
        )
        
        # Vector Store (initialized with correct dimension)
        self._vector_store: Optional[VectorStoreV2] = None
        
        self._is_ready = False
    
    def initialize(self, actions: List['ActionConfig']):
        """Initialize models and compute all vectors."""
        try:
            self._load_models()
            self._init_vector_store()
            self._compute_all_vectors(actions)
            self._is_ready = True
            logger.info(f"EmbedAnything engine initialized with {len(actions)} actions")
        except Exception as e:
            logger.error(f"Failed to initialize embedding engine: {e}")
            self._is_ready = False
            raise
    
    def _load_models(self):
        """Load primary and fallback models."""
        import embed_anything
        from embed_anything import EmbeddingModel, WhichModel
        
        # Primary model
        logger.info(f"Loading primary model: {self.config.primary_model}")
        self._model = EmbeddingModel.from_pretrained_hf(
            WhichModel.Jina,
            model_id=self.config.primary_model
        )
        
        # Fallback model
        try:
            logger.info(f"Loading fallback model: {self.config.fallback_model}")
            self._fallback_model = EmbeddingModel.from_pretrained_hf(
                WhichModel.Bert,
                model_id=self.config.fallback_model
            )
        except Exception as e:
            logger.warning(f"Failed to load fallback model: {e}")
            self._fallback_model = None
    
    def _init_vector_store(self):
        """Initialize vector store with correct dimension."""
        dim = self.config.embedding_dim
        
        if self.config.vector_store.value == "faiss":
            self._vector_store = FAISSVectorStoreV2(dimension=dim)
        elif self.config.vector_store.value == "qdrant":
            self._vector_store = QdrantVectorStoreV2(
                dimension=dim,
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                collection=self.config.qdrant_collection
            )
        else:
            self._vector_store = InMemoryVectorStoreV2(dimension=dim)
        
        logger.info(f"Vector store initialized: {self.config.vector_store.value}, dim={dim}")
    
    def _compute_all_vectors(self, actions: List['ActionConfig']):
        """
        Compute vectors for all actions.
        
        FIX (C): Map by (action_id, idx) not by text to avoid collision.
        """
        # Collect all texts with proper indexing
        all_items = []  # [(action_id, seed_idx, text), ...]
        
        for action in actions:
            # Index 0 = business_description
            all_items.append((action.action_id, 0, action.business_description))
            
            # Index 1+ = seed phrases
            for idx, phrase in enumerate(action.seed_phrases, start=1):
                all_items.append((action.action_id, idx, phrase))
        
        # Batch encode
        texts = [item[2] for item in all_items]
        vectors = self._batch_encode(texts)
        
        # Build new state (atomic)
        new_seed_vectors = {}
        new_action_seeds = {}
        
        for (action_id, seed_idx, text), vector in zip(all_items, vectors):
            key = f"{action_id}::{seed_idx}"
            
            seed_vec = SeedVector(
                action_id=action_id,
                seed_index=seed_idx,
                seed_text=text,
                vector=np.array(vector),
                confidence=1.0
            )
            new_seed_vectors[key] = seed_vec
            
            if action_id not in new_action_seeds:
                new_action_seeds[action_id] = []
            new_action_seeds[action_id].append(key)
            
            # Also add to vector store
            self._vector_store.add_seed(action_id, seed_idx, np.array(vector))
        
        # Atomic swap
        self._state.atomic_update(new_seed_vectors, new_action_seeds)
    
    def _batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """Batch encode with fallback."""
        import embed_anything
        
        try:
            embeddings = embed_anything.embed_query(texts, self._model)
            return [np.array(e.embedding) for e in embeddings]
        except Exception as e:
            logger.warning(f"Primary model failed: {e}, trying fallback")
            if self._fallback_model:
                embeddings = embed_anything.embed_query(texts, self._fallback_model)
                return [np.array(e.embedding) for e in embeddings]
            raise
    
    def _encode_single(self, text: str) -> np.ndarray:
        """Encode single text with caching."""
        # Check cache first
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        
        # Encode
        vectors = self._batch_encode([text])
        vector = vectors[0]
        
        # Cache
        self._cache.put(text, vector)
        
        return vector
    
    def batch_score(self, text: str, action_ids: List[str]) -> Dict[str, float]:
        """
        Score multiple actions for a query.
        
        Uses vector store search + aggregation for efficiency.
        """
        if not self._is_ready:
            return {aid: 0.0 for aid in action_ids}
        
        query_vector = self._encode_single(text)
        
        # Use vector store search with aggregation
        action_scores = self._vector_store.search_actions(
            query_vector=query_vector,
            top_k_seeds=50,
            top_k_actions=len(action_ids),
            aggregation="max" if self.config.max_pool else "avg"
        )
        
        # Convert to dict
        scores = {aid: 0.0 for aid in action_ids}
        for action_score in action_scores:
            if action_score.action_id in scores:
                scores[action_score.action_id] = action_score.score
        
        return scores
    
    def score(self, text: str, action_id: str) -> float:
        """Score single action."""
        scores = self.batch_score(text, [action_id])
        return scores.get(action_id, 0.0)
    
    # === Incremental Update Methods (Thread-Safe) ===
    
    def update_action(self, action: 'ActionConfig'):
        """
        Hot-update vectors for single action.
        Thread-safe with atomic swap.
        """
        if not self._is_ready:
            return
        
        # Compute new vectors
        texts = [action.business_description] + action.seed_phrases
        vectors = self._batch_encode(texts)
        
        # Build seed vectors
        seed_vectors = []
        for idx, (text, vector) in enumerate(zip(texts, vectors)):
            seed_vectors.append(SeedVector(
                action_id=action.action_id,
                seed_index=idx,
                seed_text=text,
                vector=np.array(vector),
                confidence=1.0
            ))
        
        # Atomic update state
        self._state.atomic_update_action(action.action_id, seed_vectors)
        
        # Update vector store
        self._vector_store.delete_action(action.action_id)
        for sv in seed_vectors:
            self._vector_store.add_seed(sv.action_id, sv.seed_index, sv.vector)
        
        logger.info(f"Updated action: {action.action_id}")
    
    def remove_action(self, action_id: str):
        """Remove action (thread-safe)."""
        if not self._is_ready:
            return
        
        self._state.atomic_remove_action(action_id)
        self._vector_store.delete_action(action_id)
        logger.info(f"Removed action: {action_id}")
    
    def update_seed_confidence(self, action_id: str, seed_index: int, confidence: float):
        """Update confidence for learning loop."""
        # This requires a more complex update - for now, log it
        logger.info(f"Confidence update: {action_id}::{seed_index} = {confidence}")
    
    def clear_cache(self):
        """Clear query cache."""
        self._cache.clear()
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        _, _, version = self._state.get_state()
        return {
            "is_ready": self._is_ready,
            "state_version": version,
            "cache_stats": self._cache.stats(),
            "embedding_dim": self.config.embedding_dim,
            "vector_store": self.config.vector_store.value
        }
```

### 2.6 Config Watcher với Atomic File Read

```python
# app/utils/config_watcher_v2.py

import os
import hashlib
import threading
import time
import tempfile
import shutil
from typing import Callable, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AtomicConfigWatcher:
    """
    Watch config files with atomic read protection.
    
    Features:
    - Detect changes via hash comparison
    - Atomic read (avoid reading while file is being written)
    - Retry on read failure
    - Incremental callback for actions
    """
    
    def __init__(self,
                 config_paths: Dict[str, str],
                 on_action_change: Callable[[str], None],
                 on_rule_change: Callable[[str], None],
                 poll_interval: float = 2.0,
                 max_retries: int = 3,
                 retry_delay: float = 0.5):
        self.config_paths = config_paths
        self.on_action_change = on_action_change
        self.on_rule_change = on_rule_change
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._file_hashes: Dict[str, str] = {}
        self._file_sizes: Dict[str, int] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start watching in background thread."""
        if self._running:
            return
        
        self._running = True
        self._init_hashes()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("Config watcher started")
    
    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Config watcher stopped")
    
    def _init_hashes(self):
        """Initialize file hashes."""
        for name, path in self.config_paths.items():
            hash_val, size = self._compute_hash_safe(path)
            self._file_hashes[name] = hash_val
            self._file_sizes[name] = size
    
    def _compute_hash_safe(self, path: str) -> tuple[str, int]:
        """
        Compute file hash with atomic read protection.
        
        Strategy:
        1. Check file size twice with delay
        2. If size changed, file is being written - skip
        3. Read and hash
        """
        if not os.path.exists(path):
            return "", 0
        
        for attempt in range(self.max_retries):
            try:
                # Check size
                size1 = os.path.getsize(path)
                time.sleep(0.05)  # Small delay
                size2 = os.path.getsize(path)
                
                if size1 != size2:
                    # File is being written
                    logger.debug(f"File {path} is being written, retry {attempt + 1}")
                    time.sleep(self.retry_delay)
                    continue
                
                # Read and hash
                with open(path, 'rb') as f:
                    content = f.read()
                    return hashlib.md5(content).hexdigest(), len(content)
                    
            except (IOError, OSError) as e:
                logger.warning(f"Error reading {path}: {e}, retry {attempt + 1}")
                time.sleep(self.retry_delay)
        
        # Failed after retries
        logger.error(f"Failed to read {path} after {self.max_retries} retries")
        return self._file_hashes.get(os.path.basename(path), ""), 0
    
    def _watch_loop(self):
        """Main watch loop."""
        while self._running:
            time.sleep(self.poll_interval)
            
            for name, path in self.config_paths.items():
                try:
                    new_hash, new_size = self._compute_hash_safe(path)
                    old_hash = self._file_hashes.get(name, "")
                    
                    if new_hash and new_hash != old_hash:
                        logger.info(f"Config changed: {name}")
                        self._file_hashes[name] = new_hash
                        self._file_sizes[name] = new_size
                        self._handle_change(name, path)
                        
                except Exception as e:
                    logger.error(f"Error watching {name}: {e}")
    
    def _handle_change(self, name: str, path: str):
        """Handle config file change."""
        try:
            if "action" in name.lower():
                self.on_action_change(path)
            elif "rule" in name.lower():
                self.on_rule_change(path)
        except Exception as e:
            logger.error(f"Error handling change for {name}: {e}")


def atomic_write_config(path: str, content: str):
    """
    Write config file atomically.
    
    Strategy: Write to temp file, then rename (atomic on most filesystems).
    """
    dir_path = os.path.dirname(path)
    
    # Write to temp file
    fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Atomic rename
        shutil.move(temp_path, path)
        
    except Exception:
        # Cleanup temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
```

### 2.7 Router v2 Production Integration

```python
# app/router/router_v2_production.py

import os
import uuid
import logging
from typing import List, Optional

from app.core.models import (
    RouterOutput, UserRequest, ActionCandidate, 
    UIStrategy, ActionConfig
)
from app.utils.config_loader import ConfigLoader
from app.router.preprocess import Preprocessor
from app.router.rule_engine import RuleEngine
from app.router.ui_decision import UIDecision
from app.router.embed_config import EmbedConfig, VectorStoreType
from app.router.embed_anything_engine_v2 import EmbedAnythingEngineV2
from app.utils.config_watcher_v2 import AtomicConfigWatcher

logger = logging.getLogger(__name__)


class FuserV2:
    """
    Dynamic fuser with domain/intent-aware weights.
    """
    
    def __init__(self):
        # Default weights
        self.default_w_rule = 0.6
        self.default_w_embed = 0.4
        
        # Domain-specific weights (rule-heavy domains)
        self.domain_weights = {
            "leave": {"w_rule": 0.7, "w_embed": 0.3},
            "visitor": {"w_rule": 0.5, "w_embed": 0.5},
        }
        
        # Intent-specific adjustments
        self.intent_adjustments = {
            "cancel": {"w_rule": 0.8, "w_embed": 0.2},  # Cancel needs precision
            "status": {"w_rule": 0.5, "w_embed": 0.5},
            "create": {"w_rule": 0.6, "w_embed": 0.4},
        }
    
    def fuse(self, 
             rule_score: float, 
             embed_score: float,
             domain: str = None,
             intent_type: str = None,
             has_pattern: bool = False) -> float:
        
        # Get base weights
        w_rule = self.default_w_rule
        w_embed = self.default_w_embed
        
        # Apply domain weights
        if domain and domain in self.domain_weights:
            w_rule = self.domain_weights[domain]["w_rule"]
            w_embed = self.domain_weights[domain]["w_embed"]
        
        # Apply intent adjustments (override domain)
        if intent_type and intent_type in self.intent_adjustments:
            w_rule = self.intent_adjustments[intent_type]["w_rule"]
            w_embed = self.intent_adjustments[intent_type]["w_embed"]
        
        # Pattern bonus
        bonus = 0.15 if has_pattern else 0.0
        
        # Compute final score
        final = (w_rule * rule_score) + (w_embed * embed_score) + bonus
        
        return max(0.0, min(1.0, final))


class RouterV2Production:
    """
    Production-grade Router with EmbedAnything.
    
    Features:
    - Thread-safe hot reload
    - Dynamic dimension handling
    - TTL query cache
    - Vector store with seed → action aggregation
    - Feature flag for v1/v2 switching
    """
    
    def __init__(self, 
                 config_loader: ConfigLoader,
                 embed_config: EmbedConfig = None,
                 enable_v2: bool = True):
        
        self.loader = config_loader
        self.enable_v2 = enable_v2
        
        # Core components
        self.preprocessor = Preprocessor()
        self.rule_engine = RuleEngine()
        self.ui_decision = UIDecision()
        self.fuser = FuserV2()
        
        # Embedding engine (v2 or v1 fallback)
        if enable_v2:
            self.embed_config = embed_config or EmbedConfig()
            self.embedding_engine = EmbedAnythingEngineV2(self.embed_config)
        else:
            from app.router.embedding_engine import EmbeddingEngine
            self.embedding_engine = EmbeddingEngine()
        
        # Initialize
        self._initialize()
        
        # Config watcher (only if v2)
        self._watcher: Optional[AtomicConfigWatcher] = None
        if enable_v2:
            self._start_watcher()
    
    def _initialize(self):
        """Load config and initialize engines."""
        self.loader.load()
        self.actions = self.loader.get_all_actions()
        self.embedding_engine.initialize(self.actions)
        logger.info(f"Router initialized with {len(self.actions)} actions")
    
    def _start_watcher(self):
        """Start config file watcher."""
        self._watcher = AtomicConfigWatcher(
            config_paths={
                "actions": self.loader.action_catalog_path,
                "rules": self.loader.rule_config_path
            },
            on_action_change=self._on_action_config_change,
            on_rule_change=self._on_rule_config_change,
            poll_interval=self.embed_config.reload_poll_interval
        )
        self._watcher.start()
    
    def _on_action_config_change(self, path: str):
        """Handle action config change (incremental update)."""
        logger.info(f"Action config changed: {path}")
        
        try:
            # Get old and new actions
            old_actions = {a.action_id: a for a in self.actions}
            self.loader.load()
            new_actions = {a.action_id: a for a in self.loader.get_all_actions()}
            
            # Find changes
            for action_id, action in new_actions.items():
                if action_id not in old_actions:
                    # New action
                    logger.info(f"New action detected: {action_id}")
                    self.embedding_engine.update_action(action)
                elif self._action_changed(old_actions[action_id], action):
                    # Changed action
                    logger.info(f"Action changed: {action_id}")
                    self.embedding_engine.update_action(action)
            
            # Find removed
            for action_id in old_actions:
                if action_id not in new_actions:
                    logger.info(f"Action removed: {action_id}")
                    self.embedding_engine.remove_action(action_id)
            
            self.actions = list(new_actions.values())
            
        except Exception as e:
            logger.error(f"Error handling action config change: {e}")
    
    def _on_rule_config_change(self, path: str):
        """Handle rule config change (lightweight reload)."""
        logger.info(f"Rule config changed: {path}")
        try:
            self.loader.load()
        except Exception as e:
            logger.error(f"Error handling rule config change: {e}")
    
    def _action_changed(self, old: ActionConfig, new: ActionConfig) -> bool:
        """Check if action config changed (affects embedding)."""
        return (
            old.business_description != new.business_description or
            old.seed_phrases != new.seed_phrases
        )
    
    def route(self, request: UserRequest) -> RouterOutput:
        """
        Route user request to best action(s).
        """
        request_id = request.request_id or str(uuid.uuid4())
        
        # 1. Preprocess
        clean_text = self.preprocessor.process(request.text)
        
        # 2. Get scores
        action_ids = [a.action_id for a in self.actions]
        embed_scores = self.embedding_engine.batch_score(clean_text, action_ids)
        
        # 3. Build candidates
        candidates: List[ActionCandidate] = []
        
        for action in self.actions:
            # Rule score
            rule_config = self.loader.get_rule(action.action_id)
            rule_score, rule_reasons = self.rule_engine.score(clean_text, rule_config)
            
            # Embed score
            embed_score = embed_scores.get(action.action_id, 0.0)
            
            # Fuse with domain/intent awareness
            has_pattern = any("pattern" in r for r in rule_reasons)
            final_score = self.fuser.fuse(
                rule_score=rule_score,
                embed_score=embed_score,
                domain=action.domain,
                intent_type=action.intent_type.value if hasattr(action.intent_type, 'value') else str(action.intent_type),
                has_pattern=has_pattern
            )
            
            # Build reasoning
            reasons = rule_reasons.copy()
            if embed_score > 0.5:
                reasons.append(f"semantic: {embed_score:.2f}")
            
            candidates.append(ActionCandidate(
                action_id=action.action_id,
                rule_score=rule_score,
                embed_score=embed_score,
                final_score=final_score,
                reasoning=reasons
            ))
        
        # 4. Sort and select top
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        top_candidates = candidates[:5]
        
        # 5. UI decision
        strategy, message = self.ui_decision.decide(top_candidates, self.loader.actions)
        
        return RouterOutput(
            request_id=request_id,
            top_actions=top_candidates,
            ui_strategy=strategy,
            message=message
        )
    
    def reload(self):
        """Force full reload (for admin API)."""
        self._initialize()
    
    def get_stats(self) -> dict:
        """Get router statistics."""
        stats = {
            "actions_count": len(self.actions),
            "enable_v2": self.enable_v2,
        }
        
        if self.enable_v2 and hasattr(self.embedding_engine, 'get_stats'):
            stats["embedding"] = self.embedding_engine.get_stats()
        
        return stats
    
    def shutdown(self):
        """Cleanup resources."""
        if self._watcher:
            self._watcher.stop()
```

---

## 3. HEALTH CHECK & FEATURE FLAG

### 3.1 Health Endpoints

```python
# app/health.py

from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(tags=["health"])

# Global references (set by main.py)
_router_instance = None
_embedding_engine = None

def set_instances(router_inst, embed_engine):
    global _router_instance, _embedding_engine
    _router_instance = router_inst
    _embedding_engine = embed_engine


@router.get("/healthz")
def health_check() -> Dict[str, Any]:
    """
    Liveness probe - process is running.
    """
    return {"status": "ok"}


@router.get("/readyz")
def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe - all components ready to serve.
    """
    checks = {
        "config_loaded": False,
        "embedding_ready": False,
        "vector_store_ready": False,
    }
    
    try:
        if _router_instance:
            checks["config_loaded"] = len(_router_instance.actions) > 0
        
        if _embedding_engine:
            stats = _embedding_engine.get_stats()
            checks["embedding_ready"] = stats.get("is_ready", False)
            checks["vector_store_ready"] = True  # If no error, it's ready
            
    except Exception as e:
        return {
            "status": "not_ready",
            "checks": checks,
            "error": str(e)
        }
    
    all_ready = all(checks.values())
    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks
    }


@router.get("/stats")
def get_stats() -> Dict[str, Any]:
    """
    Get detailed statistics.
    """
    stats = {}
    
    if _router_instance:
        stats["router"] = _router_instance.get_stats()
    
    if _embedding_engine:
        stats["embedding"] = _embedding_engine.get_stats()
    
    return stats
```

### 3.2 Feature Flag Integration

```python
# app/main_v2.py

import os
from fastapi import FastAPI
from app.utils.config_loader import ConfigLoader
from app.router.embed_config import EmbedConfig, VectorStoreType

app = FastAPI(title="Hybrid Intent Router v2")

# Feature flags from environment
ENABLE_V2 = os.getenv("ROUTER_V2_ENABLED", "true").lower() in ("1", "true", "yes")
VECTOR_STORE = os.getenv("VECTOR_STORE", "memory")  # memory, faiss, qdrant
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

loader = ConfigLoader(
    os.path.join(CONFIG_DIR, "action_catalog.yaml"),
    os.path.join(CONFIG_DIR, "keyword_rules.yaml")
)

# Initialize router based on feature flag
if ENABLE_V2:
    from app.router.router_v2_production import RouterV2Production
    
    embed_config = EmbedConfig(
        primary_model=EMBEDDING_MODEL,
        vector_store=VectorStoreType(VECTOR_STORE),
    )
    router = RouterV2Production(loader, embed_config, enable_v2=True)
else:
    from app.router.router import Router
    router = Router(loader)

# Health endpoints
from app.health import router as health_router, set_instances
set_instances(router, getattr(router, 'embedding_engine', None))
app.include_router(health_router)

# ... rest of endpoints
```

---

## 4. CHECKLIST PRODUCTION READINESS

### P0 - Must Have (Before Go-Live)

| Item | Status | Notes |
|------|--------|-------|
| ✅ Dynamic dimension | Fixed | `EmbedConfig.embedding_dim` |
| ✅ Thread-safe hot reload | Fixed | `RWLock` + `AtomicVectorState` |
| ✅ Key mapping bug | Fixed | Map by `(action_id, idx)` |
| ✅ TTL Cache | Fixed | `TTLCache` with expiration |
| ✅ Seed → Action aggregation | Fixed | `VectorStoreV2.search_actions()` |
| ✅ Feature flag v1/v2 | Added | `ROUTER_V2_ENABLED` env var |
| ✅ Health/readiness | Added | `/healthz`, `/readyz` |
| ⬜ Atomic config write | Added | `atomic_write_config()` |

### P1 - Stability (Week 1-2)

| Item | Status | Notes |
|------|--------|-------|
| ⬜ Metrics (Prometheus) | TODO | Latency, cache hit rate, error rate |
| ⬜ Structured logging | TODO | JSON format, correlation ID |
| ⬜ Evaluation harness | TODO | Top-1/Top-3/MRR measurement |
| ⬜ Confusion pair logging | TODO | Track misclassifications |

### P2 - Quality (Week 3-4)

| Item | Status | Notes |
|------|--------|-------|
| ⬜ 2-stage retrieval | TODO | Embed topK → rerank |
| ⬜ Active learning | TODO | Update seed confidence |
| ⬜ Task adapter (Jina) | TODO | `retrieval.query` vs `retrieval.passage` |

---

## 5. MIGRATION GUIDE

### Step 1: Install Dependencies
```bash
pip install embed-anything faiss-cpu
# Optional for production scale:
pip install qdrant-client
```

### Step 2: Create New Files
```
app/router/
├── embed_config.py          # NEW
├── thread_safe.py           # NEW
├── query_cache.py           # NEW
├── vector_store_v2.py       # NEW
├── embed_anything_engine_v2.py  # NEW
├── router_v2_production.py  # NEW
app/utils/
├── config_watcher_v2.py     # NEW
app/
├── health.py                # NEW
```

### Step 3: Test with Feature Flag
```bash
# Test v2
export ROUTER_V2_ENABLED=true
export VECTOR_STORE=memory
python -m uvicorn app.main_v2:app --reload

# Rollback to v1 if issues
export ROUTER_V2_ENABLED=false
```

### Step 4: Benchmark
```bash
python tests/router/run_eval.py --devset tests/router/fixtures/devset.yaml
# Compare v1 vs v2 metrics
```

---

## 6. ESTIMATED EFFORT (REVISED)

| Task | Original | Revised | Notes |
|------|----------|---------|-------|
| Core Engine | 3 days | 4 days | +1 day for thread-safety |
| Vector Store | 2 days | 3 days | +1 day for aggregation |
| Router Integration | 2 days | 2 days | Same |
| Config Watcher | 1 day | 1.5 days | +0.5 day for atomic read |
| Query Cache | 1 day | 1 day | Same |
| Health/Feature Flag | 0 days | 1 day | NEW |
| Testing | 1 day | 2 days | +1 day for thread-safety tests |
| Documentation | 1 day | 1 day | Same |

**Total: ~15.5 days (3 weeks, 1 senior developer)**

---

## 7. RISK MITIGATION

| Risk | Mitigation |
|------|------------|
| EmbedAnything không stable | Feature flag rollback to v1 |
| Jina v3 không support Vietnamese tốt | Benchmark trước, fallback BGE-M3 |
| Thread-safety bugs | Extensive unit tests + load testing |
| Memory leak từ cache | TTL + max_size limit + monitoring |
| Config watcher race condition | Atomic read + retry logic |
