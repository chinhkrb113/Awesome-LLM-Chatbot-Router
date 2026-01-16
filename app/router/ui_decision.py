from typing import List, Tuple
from app.core.models import ActionCandidate, UIStrategy, IntentType, ActionConfig

class UIDecision:
    def decide(self, sorted_actions: List[ActionCandidate], action_map: dict) -> Tuple[UIStrategy, str]:
        # Convert action_map values to dict lookup if it's a list
        if isinstance(action_map, list):
            action_map = {a.action_id: a for a in action_map}

        if not sorted_actions:
            return UIStrategy.CLARIFY, "Mình chưa hiểu ý bạn lắm, bạn chọn thao tác bên dưới nhé:"

        top1 = sorted_actions[0]
        
        # Get intent type of top1
        top1_config: ActionConfig = action_map.get(top1.action_id)
        intent_type = top1_config.intent_type if top1_config else IntentType.OTHER

        # Calculate gap
        gap = 0.0
        if len(sorted_actions) > 1:
            gap = top1.final_score - sorted_actions[1].final_score
        else:
            gap = 1.0 # Only one action, gap is huge

        # Strategy Logic
        # Case 1: Preselect (Very confident)
        if top1.final_score >= 0.85 and gap >= 0.15 and intent_type != IntentType.CANCEL:
            return UIStrategy.PRESELECT, f"Mình hiểu bạn đang muốn thực hiện thao tác sau:"
        
        # Case 2: Top-3 (Fairly confident)
        elif top1.final_score >= 0.70:
            return UIStrategy.TOP_3, "Bạn muốn thực hiện thao tác nào?"
        
        # Case 3: Clarify (Not confident)
        else:
            return UIStrategy.CLARIFY, "Mình chưa chắc bạn muốn làm nội dung nào, bạn chọn giúp nhé:"
