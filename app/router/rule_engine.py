import re
import unicodedata
from typing import List, Tuple

from app.core.models import RuleConfig


class RuleEngine:
    def __init__(self):
        # Trọng số và giới hạn cộng dồn
        self.strong_weight = 0.30
        self.weak_weight = 0.12
        self.negative_weight = -0.35
        self.pattern_weight = 0.45
        self.max_strong_hits = 3
        self.max_weak_hits = 3
        self.max_negative_hits = 2
        self.max_pattern_hits = 2

    def score(self, text: str, rule: RuleConfig) -> Tuple[float, List[str]]:
        if not text:
            return 0.0, []

        # Chủ động hạ chữ để tránh miss do hoa/thường
        norm_text = text.lower()
        norm_no_accent = self._strip_accents(norm_text)

        reasons: List[str] = []
        score = 0.0

        # 1. Special Patterns
        pattern_score = self._score_patterns(rule.special_patterns, norm_text, norm_no_accent, reasons)
        score += pattern_score

        # 2. Strong keywords
        strong_score = self._score_keywords(
            keywords=rule.strong_keywords,
            weight=self.strong_weight,
            max_hits=self.max_strong_hits,
            text=norm_text,
            text_no_accent=norm_no_accent,
            label="strong",
            reasons=reasons,
        )
        score += strong_score

        # 3. Weak keywords
        weak_score = self._score_keywords(
            keywords=rule.weak_keywords,
            weight=self.weak_weight,
            max_hits=self.max_weak_hits,
            text=norm_text,
            text_no_accent=norm_no_accent,
            label="weak",
            reasons=reasons,
        )
        score += weak_score

        # 4. Negative keywords
        negative_score = self._score_keywords(
            keywords=rule.negative_keywords,
            weight=self.negative_weight,
            max_hits=self.max_negative_hits,
            text=norm_text,
            text_no_accent=norm_no_accent,
            label="negative",
            reasons=reasons,
        )
        score += negative_score

        final_score = max(0.0, min(1.0, score))
        return final_score, reasons

    def _score_patterns(
        self,
        patterns: List[str],
        text: str,
        text_no_accent: str,
        reasons: List[str],
    ) -> float:
        hits = 0
        for pattern in patterns:
            regex = re.escape(pattern).replace(r"\*", ".*")
            if self._search(regex, text) or self._search(regex, text_no_accent):
                hits += 1
                reasons.append(f"pattern: {pattern}")
                if hits >= self.max_pattern_hits:
                    break
        return self.pattern_weight * min(hits, self.max_pattern_hits)

    def _score_keywords(
        self,
        keywords: List[str],
        weight: float,
        max_hits: int,
        text: str,
        text_no_accent: str,
        label: str,
        reasons: List[str],
    ) -> float:
        hits = 0
        for kw in keywords:
            if self._contains_keyword(kw, text, text_no_accent):
                hits += 1
                reasons.append(f"{label}: {kw}")
                if hits >= max_hits:
                    break
        return weight * min(hits, max_hits)

    def _contains_keyword(self, keyword: str, text: str, text_no_accent: str) -> bool:
        if not keyword:
            return False
        kw_norm = keyword.lower()
        kw_no_accent = self._strip_accents(kw_norm)

        # Biên từ để tránh match chuỗi con
        pattern = rf"\b{re.escape(kw_norm)}\b"
        pattern_no = rf"\b{re.escape(kw_no_accent)}\b"

        return self._search(pattern, text) or self._search(pattern_no, text_no_accent)

    def _search(self, regex: str, text: str) -> bool:
        try:
            return re.search(regex, text, flags=re.IGNORECASE) is not None
        except re.error:
            # Nếu pattern lỗi, bỏ qua để không làm hỏng pipeline
            return False

    def _strip_accents(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if not unicodedata.combining(ch))
