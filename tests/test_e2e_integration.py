import unittest
import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.main import app

class TestE2EIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def test_full_visitor_flow(self):
        # 1. Route: User says "tạo visitor"
        print("\n1. [E2E] Routing...")
        route_res = self.client.post("/route", json={"text": "tạo visitor"})
        self.assertEqual(route_res.status_code, 200)
        data = route_res.json()
        
        # Expect PRESELECT or TOP_3 with visitor.create at top
        top_action = data["top_actions"][0]["action_id"]
        self.assertEqual(top_action, "visitor.create")
        
        # 2. Start Action
        print(f"2. [E2E] Starting action: {top_action}")
        session_id = "e2e_test_session_123"
        start_res = self.client.post("/action/start", json={
            "session_id": session_id,
            "action_id": top_action,
            "initial_text": "tạo visitor"
        })
        self.assertEqual(start_res.status_code, 200)
        state = start_res.json()
        self.assertEqual(state["status"], "COLLECTING")
        
        # 3. Interact: Provide Name
        print("3. [E2E] User provides name...")
        interact_res = self.client.post("/action/interact", json={
            "session_id": session_id,
            "text": "Nguyễn Văn A"
        })
        state = interact_res.json()
        # Expect collected visitor_name, now asking for visit_date
        self.assertIn("visitor_name", state["slots"])
        self.assertEqual(state["slots"]["visitor_name"]["value"], "Nguyễn Văn A")
        
        # 4. Interact: Provide Date
        print("4. [E2E] User provides date...")
        interact_res = self.client.post("/action/interact", json={
            "session_id": session_id,
            "text": "ngày mai"
        })
        state = interact_res.json()
        # Should be DRAFT now if all required slots filled
        # visitor.create needs: [visitor_name, visit_date]
        self.assertIn("visit_date", state["slots"])
        self.assertEqual(state["status"], "DRAFT")
        
        # 5. Confirm
        print("5. [E2E] User confirms...")
        confirm_res = self.client.post("/action/interact", json={
            "session_id": session_id,
            "text": "xác nhận"
        })
        state = confirm_res.json()
        self.assertEqual(state["status"], "COMMITTED")
        print("✅ [E2E] Flow completed successfully.")

if __name__ == "__main__":
    unittest.main()
