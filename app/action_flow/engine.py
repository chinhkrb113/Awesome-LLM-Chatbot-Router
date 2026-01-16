from typing import Dict, Any, Optional
import threading
from app.core.models import ActionState, ActionStatus, ActionConfig, SlotValue, ActionButton
from app.utils.config_loader import ConfigLoader
from app.action_flow.entity_extractor import EntityExtractor
from app.action_flow.validator import Validator

class ActionFlowEngine:
    def __init__(self, config_loader: ConfigLoader):
        self.loader = config_loader
        # In-memory storage for MVP: {session_id: ActionState}
        self.states: Dict[str, ActionState] = {}
        self._lock = threading.RLock() # Reentrant lock for thread safety
        self.extractor = EntityExtractor()
        self.validator = Validator()

    def start_action(self, session_id: str, action_id: str, initial_text: str = "") -> ActionState:
        action_config = self.loader.get_action(action_id)
        if not action_config:
            raise ValueError(f"Action {action_id} not found")

        state = ActionState(
            session_id=session_id,
            action_id=action_id,
            status=ActionStatus.INIT
        )
        
        with self._lock:
            self.states[session_id] = state
        
        # 0. Auto-fill slots from initial text (Context Handover)
        if initial_text:
            extracted = self.extractor.extract(initial_text, action_config.required_slots)
            for slot, value in extracted.items():
                is_valid, _, normalized = self.validator.validate(slot, value)
                if is_valid:
                     state.slots[slot] = SlotValue(name=slot, value=normalized)

        # Immediate transition to COLLECTING or DRAFT if no slots needed
        self._check_slots_and_transition(state, action_config)
        return state

    def get_state(self, session_id: str) -> Optional[ActionState]:
        with self._lock:
            return self.states.get(session_id)

    def handle_input(self, session_id: str, user_input: str, payload: Dict[str, Any] = None) -> ActionState:
        with self._lock:
            state = self.states.get(session_id)
        
        if not state:
            return None # Or raise error

        action_config = self.loader.get_action(state.action_id)
        
        # Clear buttons from previous state
        state.buttons = []

        # 1. Handle Cancel (Global)
        if user_input.lower() in ["huỷ", "hủy", "cancel", "bỏ"]:
            state.status = ActionStatus.CANCELED
            state.message = "Hành động đã bị hủy."
            return state

        # 2. State Machine Logic
        if state.status == ActionStatus.INIT:
            # Should have auto-transitioned, but just in case
            state.status = ActionStatus.COLLECTING
            
        if state.status == ActionStatus.COLLECTING:
            missing = self._get_missing_slots(state, action_config)
            if missing:
                current_slot = missing[0]
                
                # Validation Logic
                is_valid, err_msg, normalized_val = self.validator.validate(current_slot, user_input)
                
                if is_valid:
                    state.slots[current_slot] = SlotValue(name=current_slot, value=normalized_val)
                    # Sau khi điền xong 1 slot, check tiếp
                    self._check_slots_and_transition(state, action_config)
                else:
                    # Validation fail: giữ nguyên state, yêu cầu nhập lại
                    state.message = f"{err_msg} Vui lòng nhập lại {current_slot}:"
            else:
                 self._check_slots_and_transition(state, action_config)

        elif state.status == ActionStatus.DRAFT:
            if user_input.lower() in ["xác nhận", "confirm", "ok", "đúng"]:
                state.status = ActionStatus.CONFIRMED
                # Here we would trigger the actual backend API commit
                self._commit_action(state)
            elif user_input.lower() in ["sửa", "edit"]:
                # Simple logic: clear slots and go back to collecting
                state.slots = {} 
                self._check_slots_and_transition(state, action_config)
            elif user_input.lower() in ["huỷ", "cancel"]: # Handle cancel button specifically if missed by global
                state.status = ActionStatus.CANCELED
                state.message = "Hành động đã bị hủy."
            else:
                 # Unknown command in draft, re-display draft
                 self._check_slots_and_transition(state, action_config)

        return state

    def _get_missing_slots(self, state: ActionState, config: ActionConfig):
        return [slot for slot in config.required_slots if slot not in state.slots]

    def _check_slots_and_transition(self, state: ActionState, config: ActionConfig):
        missing = self._get_missing_slots(state, config)
        if not missing:
            state.status = ActionStatus.DRAFT
            
            # Use friendly action name if available
            action_name = config.friendly_name or state.action_id
            
            # Build friendly summary
            summary_lines = []
            for k, v in state.slots.items():
                slot_cfg = config.slot_config.get(k, {})
                friendly_slot_name = slot_cfg.get("friendly_name", k)
                summary_lines.append(f"- {friendly_slot_name}: {v.value}")
            
            slot_summary = "\n".join(summary_lines)
            
            # Empathetic confirmation message
            state.message = (
                f"Mình đã ghi nhận yêu cầu {action_name} của bạn với các thông tin sau:\n\n"
                f"{slot_summary}\n\n"
                f"Bạn có muốn xác nhận gửi đơn này không?"
            )
            
            state.buttons = [
                ActionButton(label="Xác nhận", value="ok", style="primary"),
                ActionButton(label="Sửa lại", value="edit", style="default"),
                ActionButton(label="Huỷ", value="cancel", style="danger")
            ]
        else:
            state.status = ActionStatus.COLLECTING
            next_slot = missing[0]
            
            # Use custom prompt from config if available
            slot_cfg = config.slot_config.get(next_slot, {})
            custom_prompt = slot_cfg.get("prompt")
            
            if custom_prompt:
                state.message = custom_prompt
            elif "date" in next_slot:
                state.message = f"Vui lòng nhập ngày cho {next_slot} (VD: hôm nay, 15/01/2026):"
            else:
                state.message = f"Vui lòng nhập thông tin cho: {next_slot}"
            state.buttons = []

    def _commit_action(self, state: ActionState):
        state.status = ActionStatus.COMMITTED
        state.message = "Cảm ơn bạn đã cung cấp thông tin. Đơn xin nghỉ của bạn sẽ được xử lý sớm nhất! Mình hiểu việc sắp xếp thời gian nghỉ đôi khi khá căng thẳng. Hãy cho mình biết nếu bạn cần giúp đỡ thêm nhé!"
        state.buttons = []
        # Log or Call API
        pass
