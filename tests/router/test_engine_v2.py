import unittest
import numpy as np
from dataclasses import dataclass
from typing import List

from app.router.vector_store_final import InMemoryVectorStoreFinal
from app.router.embed_anything_engine_final import EmbedAnythingEngineFinal
from app.router.embed_config import EmbedConfig

@dataclass
class MockActionConfig:
    action_id: str
    business_description: str
    seed_phrases: List[str]

class TestVectorStoreFinal(unittest.TestCase):
    def test_search_aggregation(self):
        store = InMemoryVectorStoreFinal(dimension=2)
        
        # Action A: 2 seeds close to [1, 0]
        store.add_seed("A", 0, np.array([1.0, 0.1]))
        store.add_seed("A", 1, np.array([0.9, 0.2]))
        
        # Action B: 1 seed close to [0, 1]
        store.add_seed("B", 0, np.array([0.1, 1.0]))
        
        # Query close to A
        query = np.array([1.0, 0.0])
        results = store.search_actions(query, top_k_actions=2, aggregation="max")
        
        self.assertEqual(results[0].action_id, "A")
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(len(results[0].matched_seeds), 2)

class TestEmbedAnythingEngineFinal(unittest.TestCase):
    def test_initialization_mock(self):
        config = EmbedConfig()
        engine = EmbedAnythingEngineFinal(config)
        
        actions = [
            MockActionConfig("act1", "desc1", ["seed1", "seed2"]),
            MockActionConfig("act2", "desc2", ["seed3"])
        ]
        
        engine.initialize(actions)
        stats = engine.get_stats()
        
        self.assertTrue(stats["is_ready"])
        # Mock mode depends on environment (lib installed or not) and fallback logic
        # Just verify key exists
        self.assertIn("mock_mode", stats)
        
        # Test scoring
        scores = engine.batch_score("test query", ["act1", "act2"])
        self.assertIn("act1", scores)
        self.assertIn("act2", scores)

if __name__ == '__main__':
    unittest.main()
