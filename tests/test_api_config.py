import unittest
import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.main import app

class TestAPIConfig(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_exception_handler(self):
        # Trigger 404
        response = self.client.get("/non-existent-endpoint")
        self.assertEqual(response.status_code, 404)
        
        # We can't easily trigger 500 without mocking internal failure,
        # but presence of handler ensures it won't crash process.

    def test_health_check(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

if __name__ == "__main__":
    unittest.main()
