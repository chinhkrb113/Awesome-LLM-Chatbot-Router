from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import threading
import hashlib
import numpy as np


@dataclass
class SeedSearchResult:
    seed_key: str
    action_id: str
    seed_index: int
    score: float


@dataclass
class ActionScore:
    action_id: str
    score: float
    matched_seeds: List[Tuple[int, float]]


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length for cosine similarity."""
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v


def stable_hash_id(key: str) -> int:
    """Convert string key to stable int64 ID."""
    return int(hashlib.md5(key.encode()).hexdigest()[:16], 16) % (2**63)


class VectorStoreFinal(ABC):
    """
    Thread-safe vector store with consistent normalization.
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._lock = threading.RLock()
    
    @abstractmethod
    def _add_seed_internal(self, action_id: str, seed_index: int, 
                           vector: np.ndarray, metadata: dict = None): pass
    
    @abstractmethod
    def _search_seeds_internal(self, query_vector: np.ndarray, 
                                top_k: int) -> List[SeedSearchResult]: pass
    
    @abstractmethod
    def _delete_action_internal(self, action_id: str): pass
    
    def add_seed(self, action_id: str, seed_index: int, vector: np.ndarray,
                 metadata: dict = None):
        """Thread-safe add with normalization."""
        normalized = normalize_vector(vector)
        with self._lock:
            self._add_seed_internal(action_id, seed_index, normalized, metadata)
    
    def search_seeds(self, query_vector: np.ndarray, top_k: int = 50) -> List[SeedSearchResult]:
        """Thread-safe search with normalization."""
        normalized = normalize_vector(query_vector)
        with self._lock:
            return self._search_seeds_internal(normalized, top_k)
    
    def delete_action(self, action_id: str):
        """Thread-safe delete."""
        with self._lock:
            self._delete_action_internal(action_id)
    
    def update_action_atomic(self, action_id: str, seeds: List[Tuple[int, np.ndarray]]):
        """
        Atomic update: delete old + add new in single lock.
        """
        with self._lock:
            self._delete_action_internal(action_id)
            for seed_index, vector in seeds:
                normalized = normalize_vector(vector)
                self._add_seed_internal(action_id, seed_index, normalized)
    
    def search_actions(self, 
                       query_vector: np.ndarray, 
                       top_k_seeds: int = 50,
                       top_k_actions: int = 10,
                       aggregation: str = "max") -> List[ActionScore]:
        """Search and aggregate to action-level scores."""
        seed_results = self.search_seeds(query_vector, top_k_seeds)
        
        # Group by action_id
        action_seeds: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for sr in seed_results:
            action_seeds[sr.action_id].append((sr.seed_index, sr.score))
        
        # Aggregate
        action_scores = []
        for action_id, seeds in action_seeds.items():
            if aggregation == "max":
                score = max(s[1] for s in seeds)
            else:
                score = sum(s[1] for s in seeds) / len(seeds)
            
            action_scores.append(ActionScore(
                action_id=action_id,
                score=score,
                matched_seeds=seeds
            ))
        
        action_scores.sort(key=lambda x: x.score, reverse=True)
        return action_scores[:top_k_actions]


class InMemoryVectorStoreFinal(VectorStoreFinal):
    """In-memory implementation with consistent normalization."""
    
    def __init__(self, dimension: int):
        super().__init__(dimension)
        self._vectors: Dict[str, np.ndarray] = {}
        self._action_keys: Dict[str, List[str]] = {}
    
    def _add_seed_internal(self, action_id: str, seed_index: int, 
                           vector: np.ndarray, metadata: dict = None):
        key = f"{action_id}::{seed_index}"
        self._vectors[key] = vector  # Already normalized
        
        if action_id not in self._action_keys:
            self._action_keys[action_id] = []
        if key not in self._action_keys[action_id]:
            self._action_keys[action_id].append(key)
    
    def _search_seeds_internal(self, query_vector: np.ndarray, 
                                top_k: int) -> List[SeedSearchResult]:
        results = []
        for key, vec in self._vectors.items():
            # Both normalized -> dot product = cosine similarity
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
    
    def _delete_action_internal(self, action_id: str):
        keys = self._action_keys.pop(action_id, [])
        for key in keys:
            self._vectors.pop(key, None)

# Placeholder for FAISS/Qdrant to avoid import errors if libs missing
class FAISSVectorStoreFinal(VectorStoreFinal):
    def __init__(self, dimension: int):
        super().__init__(dimension)
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed")
        # Implementation omitted for brevity in Phase 2 setup
        # Use InMemory for now
        self._store = InMemoryVectorStoreFinal(dimension)
    
    def _add_seed_internal(self, *args, **kwargs): return self._store._add_seed_internal(*args, **kwargs)
    def _search_seeds_internal(self, *args, **kwargs): return self._store._search_seeds_internal(*args, **kwargs)
    def _delete_action_internal(self, *args, **kwargs): return self._store._delete_action_internal(*args, **kwargs)

class QdrantVectorStoreFinal(VectorStoreFinal):
    def __init__(self, dimension: int, **kwargs):
        super().__init__(dimension)
        raise NotImplementedError("Qdrant not yet enabled")
    def _add_seed_internal(self, *args, **kwargs): pass
    def _search_seeds_internal(self, *args, **kwargs): return []
    def _delete_action_internal(self, *args, **kwargs): pass
