import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.config_loader import ConfigLoader
from app.router.router import Router
from app.action_flow.engine import ActionFlowEngine
from app.core.models import UserRequest, ActionStatus

def main():
    # Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")
    
    print("Loading Config...")
    loader = ConfigLoader(
        os.path.join(config_dir, "action_catalog.yaml"),
        os.path.join(config_dir, "keyword_rules.yaml")
    )
    loader.load()
    
    router = Router(loader)
    afe = ActionFlowEngine(loader)
    
    print(f"Loaded {len(loader.actions)} actions.")
    print("-" * 50)

    # Scenario 1: Routing
    # user_text = "Mai cho anh A vào cổng"
    user_text = "xin nghỉ phép"
    print(f"User Input: '{user_text}'")
    
    req = UserRequest(text=user_text)
    result = router.route(req)
    
    print(f"UI Message: {result.message}")
    print("Top Actions:")
    for idx, cand in enumerate(result.top_actions):
        print(f"  {idx+1}. {cand.action_id} (Score: {cand.final_score:.2f})")
        print(f"     Reasoning: {cand.reasoning}")

    if not result.top_actions:
        print("No actions found.")
        return

    # Scenario 2: User selects Top 1
    selected_action = result.top_actions[0].action_id
    session_id = "test_session_1"
    print("-" * 50)
    print(f"User selects: {selected_action}")
    
    state = afe.start_action(session_id, selected_action)
    print(f"Action Status: {state.status}")
    print(f"Slots filled: {state.slots.keys()}")
    
    # Interaction Loop (Mock)
    while state.status == ActionStatus.COLLECTING:
        missing = afe._get_missing_slots(state, loader.get_action(selected_action))
        question = f"Please provide {missing[0]}: "
        print(f"Bot: {question}")
        
        # Mock answer
        answer = "Test Value"
        print(f"User: {answer}")
        
        state = afe.handle_input(session_id, answer)
        print(f"Action Status: {state.status}")
    
    if state.status == ActionStatus.DRAFT:
        print("Bot: Draft is ready. Confirm? (yes/no)")
        print("User: yes")
        state = afe.handle_input(session_id, "confirm") # "yes" logic depends on AFE implementation, I used "confirm" keyword in code
        print(f"Action Status: {state.status}")

if __name__ == "__main__":
    main()
