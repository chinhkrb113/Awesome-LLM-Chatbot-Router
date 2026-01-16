import re
import unicodedata
from typing import Dict


class Preprocessor:
    def __init__(self):
        # Các thay thế lỗi gõ phổ biến (có thể mở rộng khi thu thập log)
        self.typo_map: Dict[str, str] = {
            "phep": "phép",
            "nghi": "nghỉ",  # thường đi với "xin", "muốn"
            "ngay": "ngày",
            "ktra": "kiểm tra",
        }

    def process(self, text: str) -> str:
        if not text:
            return ""

        # 1. Unicode normalize + lowercase
        text = self._normalize_unicode(text.lower())
        # 2. Thay thế lỗi gõ phổ biến
        text = self._replace_common_typos(text)
        # 3. Chuẩn hoá thời gian (10h, 10h30 -> 10:00, 10:30)
        text = self._normalize_time(text)
        # 4. Chuẩn hoá khoảng trắng
        text = " ".join(text.split())
        return text

    def normalize_no_accent(self, text: str) -> str:
        """Trả về chuỗi đã bỏ dấu để match rule robust hơn."""
        if not text:
            return ""
        text = self._normalize_unicode(text.lower())
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return " ".join(text.split())

    def _normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def _replace_common_typos(self, text: str) -> str:
        for wrong, right in self.typo_map.items():
            # Biên từ để tránh thay thế chuỗi con không mong muốn
            text = re.sub(rf"\b{re.escape(wrong)}\b", right, text)
        return text

    def _normalize_time(self, text: str) -> str:
        # 10h30 -> 10:30
        text = re.sub(r"\b(\d{1,2})h(\d{2})\b", r"\1:\2", text)
        # 10h -> 10:00
        text = re.sub(r"\b(\d{1,2})h\b", r"\1:00", text)
        return text
