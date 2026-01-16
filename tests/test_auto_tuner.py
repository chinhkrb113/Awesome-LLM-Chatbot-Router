import unittest
import os
import sys
import yaml
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.auto_tuner import AutoTuner

class TestAutoTuner(unittest.TestCase):
    def setUp(self):
        # Create dummy catalog
        self.test_catalog = "test_catalog.yaml"
        self.data = {
            "action_catalog": [
                {
                    "action_id": "test.action",
                    "seed_phrases": ["hello", "hi"]
                }
            ]
        }
        with open(self.test_catalog, "w", encoding="utf-8") as f:
            yaml.dump(self.data, f)
            
        self.tuner = AutoTuner(self.test_catalog)

    def tearDown(self):
        if os.path.exists(self.test_catalog):
            os.remove(self.test_catalog)
        if os.path.exists(self.tuner.backup_dir):
            shutil.rmtree(self.tuner.backup_dir)

    def test_tune_add_new(self):
        candidates = {
            "test.action": ["hello", "good morning"] # "hello" is dupe, "good morning" is new
        }
        
        result = self.tuner.tune(candidates)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["added_phrases"], 1)
        
        # Verify file content
        with open(self.test_catalog, "r", encoding="utf-8") as f:
            new_data = yaml.safe_load(f)
            
        seeds = new_data["action_catalog"][0]["seed_phrases"]
        self.assertIn("good morning", seeds)
        self.assertIn("hello", seeds)
        self.assertEqual(len(seeds), 3)

    def test_tune_no_new(self):
        candidates = {
            "test.action": ["hello"] # All dupes
        }
        result = self.tuner.tune(candidates)
        self.assertEqual(result["status"], "skipped")

if __name__ == '__main__':
    unittest.main()
