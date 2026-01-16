import requests
import json
import time

BASE_URL = "http://localhost:8000"

def log(msg, data=None):
    print(f"\n[TEST] {msg}")
    if data:
        print(json.dumps(data, indent=2, ensure_ascii=False))

def test_admin_config():
    log("Testing Admin: Get Action Config")
    resp = requests.get(f"{BASE_URL}/admin/config/actions")
    assert resp.status_code == 200
    assert "content" in resp.json()
    log("Admin Config OK")

def test_chat_flow():
    # 1. User says "kiểm tra số dư"
    text = "tôi muốn kiểm tra số dư"
    log(f"Testing Chat: Route '{text}'")
    resp = requests.post(f"{BASE_URL}/route", json={"text": text, "user_id": "test_user"})
    assert resp.status_code == 200
    data = resp.json()
    log("Route Response", data)
    
    assert len(data["top_actions"]) > 0
    top_action = data["top_actions"][0]
    action_id = top_action["action_id"]
    log(f"Top Action Identified: {action_id}")
    
    # 2. Start Action (simulating UI click or auto-start)
    session_id = "test_session_" + str(int(time.time()))
    log(f"Testing Chat: Start Action {action_id} (Session: {session_id})")
    resp = requests.post(f"{BASE_URL}/action/start", json={
        "session_id": session_id,
        "action_id": action_id
    })
    assert resp.status_code == 200
    state = resp.json()
    log("Start Action Response", state)
    
    # 3. Interact (if needed)
    if state["status"] == "COLLECTING":
        log("Action is COLLECTING, sending dummy input...")
        resp = requests.post(f"{BASE_URL}/action/interact", json={
            "session_id": session_id,
            "text": "123456" # Dummy account number
        })
        state = resp.json()
        log("Interact Response (Step 1)", state)

    # 4. Confirm
    if state["status"] == "DRAFT":
        log("Action is DRAFT, sending confirmation...")
        resp = requests.post(f"{BASE_URL}/action/interact", json={
            "session_id": session_id,
            "text": "ok"
        })
        state = resp.json()
        log("Interact Response (Confirm)", state)
        assert state["status"] == "CONFIRMED" or state["status"] == "COMMITTED"

    log("Chat Flow OK")

if __name__ == "__main__":
    try:
        test_admin_config()
        test_chat_flow()
        print("\n✅ ALL E2E TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
