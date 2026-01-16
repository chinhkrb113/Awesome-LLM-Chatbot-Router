import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_api_flow():
    print("Waiting for server to start...")
    for i in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/")
            if resp.status_code == 200:
                print("Server is UP!")
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            print(".", end="", flush=True)
    else:
        print("\nServer failed to start.")
        return

    # 1. Test Routing
    print("\n--- Testing Router ---")
    payload = {"text": "Mai cho anh A vào cổng", "user_id": "tester"}
    resp = requests.post(f"{BASE_URL}/route", json=payload)
    data = resp.json()
    print(f"Input: {payload['text']}")
    print(f"UI Message: {data['message']}")
    top1 = data['top_actions'][0]
    print(f"Top 1 Action: {top1['action_id']} (Score: {top1['final_score']:.2f})")
    
    # 2. Test Action Start
    print("\n--- Testing Action Start ---")
    session_id = "api_test_session_1"
    action_id = top1['action_id']
    
    resp = requests.post(f"{BASE_URL}/action/start", json={
        "session_id": session_id,
        "action_id": action_id
    })
    state = resp.json()
    print(f"Action Started: {state['action_id']}")
    print(f"Status: {state['status']}")
    
    # 3. Test Interaction (Slot Filling)
    print("\n--- Testing Interaction ---")
    # Visitor create needs: visitor_name, visit_date
    
    # Mock filling slots
    slots_to_fill = ["Nguyen Van A", "2026-10-20"]
    
    for val in slots_to_fill:
        resp = requests.post(f"{BASE_URL}/action/interact", json={
            "session_id": session_id,
            "text": val
        })
        state = resp.json()
        print(f"Sent: {val} -> Status: {state['status']}")
        if state['status'] == "DRAFT":
            print("Draft reached!")
            break
            
    # 4. Confirm
    if state['status'] == "DRAFT":
        print("\n--- Confirming ---")
        resp = requests.post(f"{BASE_URL}/action/interact", json={
            "session_id": session_id,
            "text": "confirm"
        })
        state = resp.json()
        print(f"Final Status: {state['status']}")
        assert state['status'] == "COMMITTED"
        print("SUCCESS: Full flow verified.")

if __name__ == "__main__":
    test_api_flow()
