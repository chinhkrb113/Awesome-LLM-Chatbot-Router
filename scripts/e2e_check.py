import requests
import json
import sys
import time

API_URL = "http://localhost:8000"

def test_chat(text, expected_action=None):
    print(f"\nUser: {text}")
    try:
        # 1. Route
        t0 = time.time()
        resp = requests.post(f"{API_URL}/route", json={"text": text, "user_id": "test_user"})
        lat = (time.time() - t0) * 1000
        
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} - {resp.text}")
            return False
            
        data = resp.json()
        top_action = data['top_actions'][0]['action_id'] if data['top_actions'] else "None"
        confidence = data['top_actions'][0]['final_score'] if data['top_actions'] else 0.0
        
        print(f"Bot: Strategy={data['ui_strategy']}, Action={top_action} ({confidence:.2f}), Latency={lat:.0f}ms")
        
        if expected_action:
            if top_action == expected_action:
                print("✅ PASS")
                return True
            else:
                print(f"❌ FAIL (Expected {expected_action})")
                return False
        return True
        
    except Exception as e:
        print(f"Exception: {e}")
        return False

def check_health():
    try:
        resp = requests.get(f"{API_URL}/healthz")
        print(f"Health: {resp.status_code} - {resp.json()}")
        return resp.status_code == 200
    except:
        print("Server not reachable")
        return False

if __name__ == "__main__":
    if not check_health():
        sys.exit(1)
        
    print("\n--- Starting E2E Check ---")
    
    # Test cases
    cases = [
        ("xin nghỉ phép ngày mai", "leave.create"),
        ("đăng ký khách đến thăm", "visitor.create"),
        ("nhà xe ở đâu", "facilities.parking"),
        ("tôi muốn hủy lịch họp", "meeting.cancel"), # Intent cancel
    ]
    
    passed = 0
    for text, action in cases:
        if test_chat(text, action):
            passed += 1
            
    print(f"\nResult: {passed}/{len(cases)} passed")
