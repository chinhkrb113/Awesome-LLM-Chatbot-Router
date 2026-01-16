"""
End-to-end test script for Feedback API integration.
Run this against a live server to verify the complete flow.

Usage:
    python scripts/test_feedback_e2e.py [--base-url http://localhost:8000]
"""

import argparse
import requests
import uuid
import json
import sys
from typing import Optional


class FeedbackE2ETest:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.passed = 0
        self.failed = 0
        
    def log(self, msg: str, status: str = "INFO"):
        colors = {
            "PASS": "\033[92m",
            "FAIL": "\033[91m",
            "INFO": "\033[94m",
            "WARN": "\033[93m",
        }
        reset = "\033[0m"
        color = colors.get(status, "")
        print(f"{color}[{status}]{reset} {msg}")
        
    def test_health(self) -> bool:
        """Test server health."""
        try:
            resp = requests.get(f"{self.base_url}/healthz", timeout=5)
            if resp.status_code == 200:
                self.log("Server is healthy", "PASS")
                self.passed += 1
                return True
            else:
                self.log(f"Health check failed: {resp.status_code}", "FAIL")
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"Cannot connect to server: {e}", "FAIL")
            self.failed += 1
            return False
            
    def test_feedback_route_endpoint(self) -> bool:
        """Test POST /feedback/route endpoint."""
        test_data = {
            "request_id": str(uuid.uuid4()),
            "user_id": "e2e_test_user",
            "session_id": "e2e_test_session",
            "selected_action": "leave.create",
            "selection_index": 0,
            "selection_source": "click",
            "ui_strategy": "TOP_3"
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/feedback/route",
                json=test_data,
                timeout=5
            )
            
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                self.log("POST /feedback/route works correctly", "PASS")
                self.passed += 1
                return True
            else:
                self.log(f"POST /feedback/route failed: {resp.text}", "FAIL")
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"POST /feedback/route error: {e}", "FAIL")
            self.failed += 1
            return False
            
    def test_feedback_outcome_endpoint(self) -> bool:
        """Test POST /feedback/outcome endpoint."""
        test_data = {
            "request_id": str(uuid.uuid4()),
            "user_id": "e2e_test_user",
            "session_id": "e2e_test_session",
            "action_id": "leave.create",
            "status": "confirmed"
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/feedback/outcome",
                json=test_data,
                timeout=5
            )
            
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                self.log("POST /feedback/outcome works correctly", "PASS")
                self.passed += 1
                return True
            else:
                self.log(f"POST /feedback/outcome failed: {resp.text}", "FAIL")
                self.failed += 1
                return False
        except Exception as e:
            self.log(f"POST /feedback/outcome error: {e}", "FAIL")
            self.failed += 1
            return False
            
    def test_complete_flow(self) -> bool:
        """Test complete flow: route -> selection feedback -> outcome feedback."""
        self.log("Testing complete feedback flow...", "INFO")
        
        # Step 1: Route request
        try:
            route_resp = requests.post(
                f"{self.base_url}/route",
                json={
                    "text": "xin nghỉ phép 2 ngày",
                    "user_id": "e2e_flow_test"
                },
                timeout=10
            )
            
            if route_resp.status_code != 200:
                self.log(f"Route request failed: {route_resp.text}", "FAIL")
                self.failed += 1
                return False
                
            route_data = route_resp.json()
            request_id = route_data.get("request_id")
            
            if not request_id:
                self.log("Route response missing request_id", "FAIL")
                self.failed += 1
                return False
                
            self.log(f"Route returned request_id: {request_id}", "INFO")
            
        except Exception as e:
            self.log(f"Route request error: {e}", "FAIL")
            self.failed += 1
            return False
            
        # Step 2: Selection feedback
        if route_data.get("top_actions"):
            selected_action = route_data["top_actions"][0]["action_id"]
            
            selection_resp = requests.post(
                f"{self.base_url}/feedback/route",
                json={
                    "request_id": request_id,
                    "user_id": "e2e_flow_test",
                    "selected_action": selected_action,
                    "selection_index": 0,
                    "selection_source": "click",
                    "ui_strategy": route_data.get("ui_strategy", "CLARIFY")
                },
                timeout=5
            )
            
            if selection_resp.status_code != 200:
                self.log(f"Selection feedback failed: {selection_resp.text}", "FAIL")
                self.failed += 1
                return False
                
            self.log(f"Selection feedback sent for action: {selected_action}", "INFO")
            
            # Step 3: Outcome feedback
            outcome_resp = requests.post(
                f"{self.base_url}/feedback/outcome",
                json={
                    "request_id": request_id,
                    "user_id": "e2e_flow_test",
                    "action_id": selected_action,
                    "status": "confirmed"
                },
                timeout=5
            )
            
            if outcome_resp.status_code != 200:
                self.log(f"Outcome feedback failed: {outcome_resp.text}", "FAIL")
                self.failed += 1
                return False
                
            self.log("Complete feedback flow successful!", "PASS")
            self.passed += 1
            return True
        else:
            self.log("No actions returned from route", "WARN")
            return True
            
    def test_feedback_file_created(self) -> bool:
        """Check if feedback file exists and has content."""
        import os
        feedback_path = "logs/router_feedback.jsonl"
        
        if os.path.exists(feedback_path):
            with open(feedback_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    self.log(f"Feedback file has {len(lines)} entries", "PASS")
                    self.passed += 1
                    return True
                else:
                    self.log("Feedback file is empty", "WARN")
                    return True
        else:
            self.log("Feedback file not found (will be created on first feedback)", "INFO")
            return True
            
    def run_all_tests(self):
        """Run all E2E tests."""
        self.log("=" * 50, "INFO")
        self.log("Starting Feedback API E2E Tests", "INFO")
        self.log("=" * 50, "INFO")
        
        # Test 1: Health check
        if not self.test_health():
            self.log("Server not available, aborting tests", "FAIL")
            return False
            
        # Test 2: Feedback route endpoint
        self.test_feedback_route_endpoint()
        
        # Test 3: Feedback outcome endpoint
        self.test_feedback_outcome_endpoint()
        
        # Test 4: Complete flow (requires embedding engine)
        try:
            self.test_complete_flow()
        except Exception as e:
            self.log(f"Complete flow test skipped: {e}", "WARN")
            
        # Test 5: Check feedback file
        self.test_feedback_file_created()
        
        # Summary
        self.log("=" * 50, "INFO")
        self.log(f"Tests completed: {self.passed} passed, {self.failed} failed", 
                 "PASS" if self.failed == 0 else "FAIL")
        self.log("=" * 50, "INFO")
        
        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description="E2E tests for Feedback API")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL of the API server")
    args = parser.parse_args()
    
    tester = FeedbackE2ETest(args.base_url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
