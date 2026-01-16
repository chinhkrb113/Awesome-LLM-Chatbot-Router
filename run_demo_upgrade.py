import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from app.utils.config_loader import ConfigLoader
from app.action_flow.engine import ActionFlowEngine

def print_header(text):
    print(f"\n{'='*50}")
    print(f"  {text}")
    print(f"{'='*50}")

def run_demo():
    # 1. Setup Environment
    config_dir = os.path.join(os.getcwd(), "config")
    loader = ConfigLoader(
        os.path.join(config_dir, "action_catalog.yaml"),
        os.path.join(config_dir, "keyword_rules.yaml")
    )
    loader.load()
    engine = ActionFlowEngine(loader)

    # 2. Scenario: "Xin nghỉ ngày mai"
    user_text = "tôi muốn xin nghỉ ngày mai"
    action_id = "leave.create"
    session_id = "demo_session_01"

    print_header("KỊCH BẢN: 'tôi muốn xin nghỉ ngày mai'")
    print(f"Action: {action_id}")
    print(f"Required Slots: {loader.get_action(action_id).required_slots}")
    
    # --- CÁCH CŨ (Mô phỏng) ---
    print("\n--- [BEFORE] CÁCH CŨ (Chưa có Extractor) ---")
    print("Bot: (Nhận diện intent leave.create)")
    print("Bot: Vui lòng nhập thông tin cho: start_date")
    print(f"User: {user_text}")
    print(f"Bot: (Lưu slot 'start_date' = '{user_text}') -> SAI! Cần là ngày dạng YYYY-MM-DD")
    
    # --- CÁCH MỚI (Thực tế) ---
    print("\n--- [AFTER] CÁCH MỚI (Với Extractor & Validator) ---")
    
    # Step 1: Start Action with Context
    print(f"\n1. Router chuyển intent + text sang ActionFlow...")
    state = engine.start_action(session_id, action_id, initial_text=user_text)
    
    print(f"   -> Trạng thái: {state.status}")
    print(f"   -> Slots đã điền tự động: {[f'{k}: {v.value}' for k,v in state.slots.items()]}")
    
    if "start_date" in state.slots:
        print("   ✅ SUCCESS: Đã bắt được 'ngày mai' và chuyển thành ngày cụ thể!")
    else:
        print("   ❌ FAIL: Chưa bắt được ngày.")

    # Step 2: Next Interaction
    print(f"\n2. Bot phản hồi: {state.message}")
    
    # Step 3: User inputs missing info
    next_input = "nghỉ ốm"
    print(f"\n3. User nhập tiếp: '{next_input}' (cho slot leave_type)")
    state = engine.handle_input(session_id, next_input)
    print(f"   -> Bot phản hồi: {state.message}")
    print(f"   -> Slots hiện tại: {[f'{k}: {v.value}' for k,v in state.slots.items()]}")

    # Step 4: Invalid Input Test
    print("\n4. Test Validate ngày tháng sai:")
    print("   User nhập: 'ngày 35 tháng 13'")
    # Force clear date to test validation input
    del state.slots['start_date']
    # Reset to collecting start_date
    engine._check_slots_and_transition(state, loader.get_action(action_id))
    
    state = engine.handle_input(session_id, "35/13/2025")
    print(f"   -> Bot phản hồi: {state.message}")

if __name__ == "__main__":
    run_demo()
