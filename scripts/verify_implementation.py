import sys
import os
import time
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from app.router.router_final import RouterFinal
from app.core.models import UserRequest
from app.utils.config_loader import ConfigLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_router_features():
    print("=== STARTING ROUTER VERIFICATION ===")
    
    # 1. Initialize
    try:
        loader = ConfigLoader("config/action_catalog.yaml", "config/keyword_rules.yaml")
        # Mock embedding engine to avoid heavy load if needed, or use real one
        # For this test, we assume real one works or use v2=False if no model
        # NOTE: embedding_engine.py does not exist, so we MUST use v2
        router = RouterFinal(loader, enable_v2=True) 
        print("✅ Router Initialized")
    except Exception as e:
        print(f"❌ Router Init Failed: {e}")
        return

    # 2. Test Context Logic
    print("\n--- Testing Context & Domain Boost ---")
    req1 = UserRequest(text="xin nghỉ phép", session_id="test_session_1")
    res1 = router.route(req1)
    top1 = res1.top_actions[0].action_id
    print(f"Turn 1: {req1.text} -> {top1}")
    
    # Check context update
    ctx = router._get_context("test_session_1")
    if ctx and ctx.last_domain == "leave":
        print("✅ Context updated correctly (domain=leave)")
    else:
        print(f"❌ Context update failed: {ctx}")

    # Turn 2: Context Boost
    req2 = UserRequest(text="tạo đơn", session_id="test_session_1")
    res2 = router.route(req2)
    top2 = res2.top_actions[0].action_id
    print(f"Turn 2: {req2.text} -> {top2}")
    
    # Check if context boost is in reasoning
    if any("context:" in r for r in res2.top_actions[0].reasoning):
         print("✅ Context boost applied")
    else:
         print("❌ Context boost missing")

    # 3. Test Negative Reset
    print("\n--- Testing Negative Reset ---")
    req3 = UserRequest(text="thôi không làm nữa", session_id="test_session_1")
    router.route(req3)
    ctx = router._get_context("test_session_1")
    if ctx.last_action is None:
        print("✅ Context reset successful")
    else:
        print("❌ Context reset failed")

    # 4. Test Entity Boost
    print("\n--- Testing Entity Signal Boost ---")
    # "ngày mai" should boost .create intents
    req4 = UserRequest(text="ngày mai", session_id="session_new") 
    res4 = router.route(req4)
    
    # Find a create action
    create_cand = next((c for c in res4.top_actions if ".create" in c.action_id), None)
    if create_cand and any("entity_signal:" in r for r in create_cand.reasoning):
        print(f"✅ Entity boost detected for {create_cand.action_id}")
    else:
        print("❌ Entity boost not found")

    # 5. Test Pairwise (Mocking score gap if needed, or relying on real logic)
    print("\n--- Testing Pairwise Disambiguation ---")
    # "trạng thái" -> should prefer status over create
    req5 = UserRequest(text="trạng thái nghỉ phép", session_id="session_pair")
    res5 = router.route(req5)
    top5 = res5.top_actions[0]
    print(f"Input: {req5.text} -> Selected: {top5.action_id}")
    
    if "status" in top5.action_id:
        print("✅ Pairwise preference correct (Status)")
    else:
        print(f"⚠️ Might need tuning, got {top5.action_id}")

    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    test_router_features()
