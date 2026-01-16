import unittest
import threading
import time
import numpy as np
from app.router.thread_safe import RWLock, AtomicVectorState, SeedVector
from app.router.query_cache import TTLCache

class TestRWLock(unittest.TestCase):
    def test_read_lock_concurrency(self):
        lock = RWLock()
        counter = 0
        
        def reader():
            with lock.read_lock():
                nonlocal counter
                time.sleep(0.1)
                counter += 1
        
        threads = [threading.Thread(target=reader) for _ in range(5)]
        start = time.time()
        for t in threads: t.start()
        for t in threads: t.join()
        duration = time.time() - start
        
        self.assertEqual(counter, 5)
        # 5 threads sleep 0.1s concurrently -> total time should be close to 0.1s, not 0.5s
        self.assertLess(duration, 0.3)

class TestAtomicVectorState(unittest.TestCase):
    def test_atomic_update(self):
        state = AtomicVectorState()
        
        # Initial update
        seed1 = SeedVector("a1", 0, "test", np.array([1, 2]))
        state.atomic_update_action("a1", [seed1])
        
        vecs = state.get_seed_vectors_for_action("a1")
        self.assertEqual(len(vecs), 1)
        self.assertEqual(vecs[0].seed_text, "test")
        
        # Incremental update
        seed2 = SeedVector("a1", 1, "test2", np.array([3, 4]))
        state.atomic_update_action("a1", [seed1, seed2])
        
        vecs = state.get_seed_vectors_for_action("a1")
        self.assertEqual(len(vecs), 2)

class TestTTLCache(unittest.TestCase):
    def test_ttl_expiration(self):
        cache = TTLCache(ttl_seconds=0.1)
        cache.put("key", np.array([1]))
        
        self.assertIsNotNone(cache.get("key"))
        time.sleep(0.2)
        self.assertIsNone(cache.get("key"))

    def test_lru_eviction(self):
        cache = TTLCache(max_size=2)
        cache.put("a", np.array([1]))
        cache.put("b", np.array([2]))
        cache.put("c", np.array([3])) # Should evict 'a'
        
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))
        self.assertIsNotNone(cache.get("c"))

if __name__ == '__main__':
    unittest.main()
