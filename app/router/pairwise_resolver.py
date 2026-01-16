import logging
import aiofiles
from typing import List, Dict, Optional
from app.core.models import ActionCandidate
from app.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class PairwiseResolver:
    """
    Resolves ambiguity between top candidates using pairwise rules and default biases.
    Implementation of Architecture Review M2.
    """
    
    def __init__(self, config_loader: ConfigLoader):
        self.loader = config_loader
        self.rules_config = {}
        
    async def initialize(self):
        self.rules_config = await self._load_rules()

    async def _load_rules(self) -> Dict:
        # In a real scenario, this should come from ConfigLoader
        # For now, we'll read the yaml manually or assume ConfigLoader has it
        # Assuming ConfigLoader can load arbitrary yaml or we add a method to it.
        # Let's try to load from the file we just created if ConfigLoader doesn't have it.
        import yaml
        try:
            async with aiofiles.open("config/pairwise_rules.yaml", "r", encoding="utf-8") as f:
                content = await f.read()
                return yaml.safe_load(content) or {}
        except Exception as e:
            logger.error(f"Failed to load pairwise rules: {e}")
            return {}

    def resolve(self, text: str, candidates: List[ActionCandidate]) -> List[ActionCandidate]:
        if not candidates:
            return []
            
        text_lower = text.lower()
        
        # 1. Apply Default Bias (for dangerous intents)
        default_bias = self.rules_config.get("default_bias", {})
        for candidate in candidates:
            bias = default_bias.get(candidate.action_id, 0.0)
            if bias != 0.0:
                candidate.final_score += bias
                candidate.reasoning.append(f"bias: {bias}")
                
        # Sort again after bias
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        if len(candidates) < 2:
            return candidates
            
        # 2. Pairwise Disambiguation
        # Check gap between top 2
        top1 = candidates[0]
        top2 = candidates[1]
        gap = top1.final_score - top2.final_score
        
        if gap < 0.1: # Threshold defined in architecture
            self._apply_pairwise_rules(text_lower, top1, top2)
            
            # Re-sort after pairwise adjustment
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            
        return candidates

    def _apply_pairwise_rules(self, text: str, c1: ActionCandidate, c2: ActionCandidate):
        rules = self.rules_config.get("pairwise_disambiguation", [])
        
        for rule in rules:
            pair = rule.get("pair", [])
            if c1.action_id in pair and c2.action_id in pair:
                signals = rule.get("signals", {})
                
                # Determine which is which
                id1 = c1.action_id
                id2 = c2.action_id
                
                # Check signals for c1
                # (Simplistic logic: if signal matches, boost it)
                boost_amount = 0.15
                
                # Look for specific keys in signals that might match the intent suffix or manual mapping
                # For simplicity, we iterate signals keys
                
                score_adj = 0.0
                
                # Helper to check signals
                def check_signals(intent_key, phrase_list):
                    if any(p in text for p in phrase_list):
                        return True
                    return False

                # We need to map action_id to signal keys (e.g. leave.create -> prefer_create)
                # This is heuristic based on the config structure
                
                # Let's check prefer_create / prefer_status
                if "prefer_create" in signals:
                    if check_signals("prefer_create", signals["prefer_create"]):
                        # Boost the one that is 'create'
                        if ".create" in id1: c1.final_score += boost_amount; c1.reasoning.append("pairwise: prefer_create")
                        if ".create" in id2: c2.final_score += boost_amount; c2.reasoning.append("pairwise: prefer_create")

                if "prefer_status" in signals:
                    if check_signals("prefer_status", signals["prefer_status"]):
                         # Boost the one that is 'status'
                        if ".status" in id1: c1.final_score += boost_amount; c1.reasoning.append("pairwise: prefer_status")
                        if ".status" in id2: c2.final_score += boost_amount; c2.reasoning.append("pairwise: prefer_status")
