from typing import List

class Fuser:
    def __init__(self, w_rule: float = 0.6, w_embed: float = 0.4):
        self.w_rule = w_rule
        self.w_embed = w_embed

    def fuse(self, rule_score: float, embed_score: float, has_pattern: bool = False) -> float:
        bonus = 0.1 if has_pattern else 0.0
        final = (self.w_rule * rule_score) + (self.w_embed * embed_score) + bonus
        return max(0.0, min(1.0, final))
