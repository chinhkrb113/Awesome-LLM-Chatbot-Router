from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

from datetime import datetime

class IntentType(str, Enum):
    CREATE = "create"
    STATUS = "status"
    CANCEL = "cancel"
    BALANCE = "balance"
    OTHER = "other"

class ConversationContext(BaseModel):
    session_id: str
    last_action: Optional[str] = None
    last_domain: Optional[str] = None
    recent_intents: List[str] = []
    last_updated_at: datetime = Field(default_factory=datetime.now)
    ttl_seconds: int = 1800  # 30 minutes

    def is_expired(self) -> bool:
        return (datetime.now() - self.last_updated_at).total_seconds() > self.ttl_seconds

class RouterTrace(BaseModel):
    request_id: str
    user_text: str
    semantic_score: float = 0.0
    rule_score: float = 0.0
    context_boost: float = 0.0
    entity_boost: float = 0.0
    pairwise_adjustment: float = 0.0
    final_score: float = 0.0
    selected_action: str
    ui_strategy: str
    timestamp: datetime = Field(default_factory=datetime.now)
    explain: Optional[str] = None

class SlotDefinition(BaseModel):
    name: str
    required: bool = False
    type: str = "text" # text, date, time, select, etc.

class ActionConfig(BaseModel):
    action_id: str
    friendly_name: Optional[str] = None
    domain: str
    business_description: str
    seed_phrases: List[str]
    required_slots: List[str] = []
    optional_slots: List[str] = []
    typical_entities: List[str] = []
    examples: List[str] = []
    slot_config: Dict[str, Dict[str, str]] = {}
    slot_options: Dict[str, List[str]] = {}

    @property
    def intent_type(self) -> IntentType:
        if ".create" in self.action_id:
            return IntentType.CREATE
        elif ".status" in self.action_id:
            return IntentType.STATUS
        elif ".cancel" in self.action_id:
            return IntentType.CANCEL
        elif ".balance" in self.action_id:
            return IntentType.BALANCE
        return IntentType.OTHER

class RuleConfig(BaseModel):
    strong_keywords: List[str] = []
    weak_keywords: List[str] = []
    negative_keywords: List[str] = []
    special_patterns: List[str] = []

class UIStrategy(str, Enum):
    PRESELECT = "PRESELECT"
    TOP_3 = "TOP_3"
    CLARIFY = "CLARIFY"

class ActionCandidate(BaseModel):
    action_id: str
    friendly_name: Optional[str] = None
    rule_score: float = 0.0
    embed_score: float = 0.0
    final_score: float = 0.0
    reasoning: List[str] = []

class RouterOutput(BaseModel):
    request_id: str
    top_actions: List[ActionCandidate]
    ui_strategy: UIStrategy
    message: str # The text to display to the user

class UserRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}

class ActionStatus(str, Enum):
    INIT = "INIT"
    COLLECTING = "COLLECTING"
    DRAFT = "DRAFT"
    CONFIRMED = "CONFIRMED"
    CANCELED = "CANCELED"
    COMMITTED = "COMMITTED"

class SlotValue(BaseModel):
    name: str
    value: Any
    confidence: float = 1.0
    source: str = "user_input" # user_input, extracted, default, history

class ActionButton(BaseModel):
    label: str
    value: str
    style: str = "default" # primary, default, danger

class ActionState(BaseModel):
    session_id: str
    action_id: str
    status: ActionStatus = ActionStatus.INIT
    slots: Dict[str, SlotValue] = {}
    history: List[str] = [] # log of transitions
    message: Optional[str] = None
    buttons: List[ActionButton] = []

