import threading
from typing import Dict, List, TypeVar, Generic, Optional
from dataclasses import dataclass
from copy import deepcopy
import time
import numpy as np

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
