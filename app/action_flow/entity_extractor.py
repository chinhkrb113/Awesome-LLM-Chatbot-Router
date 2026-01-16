import re
import datetime
from typing import Dict, Any, List, Optional
import unicodedata

class EntityExtractor:
    
    # Mapping for Leave Types (Normalization)
    LEAVE_TYPE_MAPPING = {
        "Nghỉ phép năm": ["phép năm", "nghỉ phép", "annual leave", "nghỉ mát", "du lịch", "xả hơi"],
        "Nghỉ ốm": ["ốm", "bệnh", "đau", "sốt", "khám bệnh", "nhập viện", "sick", "cảm cúm"],
        "Nghỉ việc riêng": ["việc riêng", "việc nhà", "gia đình", "đám cưới", "đám ma", "về quê", "có việc", "personal", "bận"],
        "Nghỉ không lương": ["không lương", "unpaid", "hết phép"],
        "Nghỉ bù": ["nghỉ bù", "làm bù", "off in lieu"],
        "Nghỉ thai sản": ["thai sản", "đẻ", "sinh con", "maternity", "khám thai"]
    }

    def __init__(self):
        pass

    def extract(self, text: str, required_slots: List[str]) -> Dict[str, Any]:
        """
        Trích xuất các entity từ text dựa trên các slot cần thiết.
        """
        extracted = {}
        text_lower = text.lower()
        
        # 1. Date Extraction (Advanced: Multi-date)
        # Check if we need dates
        date_slots = [s for s in required_slots if s in ['date', 'start_date', 'end_date', 'visit_date']]
        
        if date_slots:
            dates = self._extract_all_dates(text_lower)
            if dates:
                # Strategy: Fill slots sequentially
                # e.g. ["hôm nay", "ngày mai"] -> start_date="today", end_date="tomorrow"
                # If only 1 date found but start/end required -> assume start=end
                if len(dates) == 1 and 'start_date' in date_slots and 'end_date' in date_slots:
                     extracted['start_date'] = dates[0]
                     extracted['end_date'] = dates[0]
                else:
                    for i, slot in enumerate(date_slots):
                        if i < len(dates):
                            extracted[slot] = dates[i]
        
        # 2. Leave Type Extraction
        if 'leave_type' in required_slots:
            l_type = self._extract_leave_type(text_lower)
            if l_type:
                extracted['leave_type'] = l_type

        # 3. Number Extraction (Duration, Quantity)
        if any(s in ['duration', 'amount', 'quantity'] for s in required_slots):
            num_val = self._extract_number(text_lower)
            if num_val:
                for slot in required_slots:
                    if slot in ['duration', 'amount', 'quantity'] and slot not in extracted:
                        extracted[slot] = num_val
                        break
                        
        # 4. Email Extraction
        if 'email' in required_slots:
            email_val = self._extract_email(text)
            if email_val:
                extracted['email'] = email_val
                
        # 5. Visitor Name / Host Name
        if 'visitor_name' in required_slots:
            v_name = self._extract_visitor_name(text)
            if v_name:
                extracted['visitor_name'] = v_name
                
        # 6. Generic Reason/Purpose (Fallback)
        if 'reason' in required_slots or 'visit_purpose' in required_slots:
             # Very simple heuristic: anything after "vì", "lý do", "để"
             # Or simple keyword matching
             pass

        return extracted

    def _extract_leave_type(self, text: str) -> Optional[str]:
        """
        Map keywords to canonical leave types.
        """
        # Prioritize longer matches or specific logic if needed
        for canonical, keywords in self.LEAVE_TYPE_MAPPING.items():
            for kw in keywords:
                # Use word boundary to avoid partial matches inside other words if necessary
                # But for Vietnamese, simple inclusion is often enough and safer for compound words
                if kw in text:
                    return canonical
        
        # Fallback: if user explicitly typed a canonical name (case-insensitive)
        for canonical in self.LEAVE_TYPE_MAPPING.keys():
            if canonical.lower() in text:
                return canonical
                
        return None

    def _extract_all_dates(self, text: str) -> List[str]:
        """
        Extract all possible dates from text, returning a list of ISO strings.
        Preserves order of appearance in text.
        """
        today = datetime.date.today()
        found_dates = []
        
        # Helper to add date if unique
        def add_date(d_str):
            if d_str and d_str not in found_dates:
                found_dates.append(d_str)

        # 1. Regex patterns (dd/mm/yyyy or dd/mm)
        regex_matches = []
        # Pattern: dd/mm or dd-mm or dd.mm, optional year
        for match in re.finditer(r'\b(\d{1,2})[/\-\.](\d{1,2})(?:[/\-\.](\d{2,4}))?\b', text):
            day, month = int(match.group(1)), int(match.group(2))
            year_str = match.group(3)
            
            # Basic validation
            if month > 12 or day > 31: continue
            
            if year_str:
                year = int(year_str)
                if len(year_str) == 2: year += 2000
            else:
                year = today.year
                # Smart year inference: if date is in the past (e.g. asking for 01/01 in Dec), might mean next year?
                # For leave creation, usually future. If date < today, assume next year?
                # Let's keep it simple: current year
                if datetime.date(year, month, day) < today and (month < today.month):
                     year += 1

            try:
                d = datetime.date(year, month, day).isoformat()
                regex_matches.append((match.start(), d))
            except ValueError:
                pass
        
        # 2. Keywords
        keyword_map = {
            "hôm nay": today,
            "bữa nay": today,
            "nay": today,
            "ngày mai": today + datetime.timedelta(days=1),
            "mai": today + datetime.timedelta(days=1),
            "sáng mai": today + datetime.timedelta(days=1),
            "chiều mai": today + datetime.timedelta(days=1),
            "ngày kia": today + datetime.timedelta(days=2),
            "ngày mốt": today + datetime.timedelta(days=2),
            "mốt": today + datetime.timedelta(days=2),
            "hôm qua": today - datetime.timedelta(days=1),
            "tuần sau": today + datetime.timedelta(days=7), # Rough approx
            "thứ 2 tuần sau": self._next_weekday(today, 0, next_week=True),
            "thứ 3 tuần sau": self._next_weekday(today, 1, next_week=True),
            # Add simple weekdays if needed (thứ 2, thứ 3...) - requires context relative to today
        }
        
        keyword_matches = []
        for kw, val in keyword_map.items():
            # Find all occurrences
            for match in re.finditer(r'\b' + re.escape(kw) + r'\b', text):
                keyword_matches.append((match.start(), val.isoformat()))

        # Combine and sort by position
        all_matches = regex_matches + keyword_matches
        all_matches.sort(key=lambda x: x[0])
        
        for _, d_str in all_matches:
            add_date(d_str)
            
        return found_dates

    def _next_weekday(self, d, weekday, next_week=False):
        """Helper to find next Monday (0), Tuesday (1), etc."""
        days_ahead = weekday - d.weekday()
        if days_ahead <= 0: # Target day already happened this week
            days_ahead += 7
        if next_week:
            days_ahead += 7
        return d + datetime.timedelta(days=days_ahead)

    def _extract_number(self, text: str) -> Optional[str]:
        match = re.search(r'\b\d+\b', text)
        if match:
            return match.group(0)
        return None

    def _extract_email(self, text: str) -> Optional[str]:
        match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if match:
            return match.group(0)
        return None
        
    def _extract_visitor_name(self, text: str) -> Optional[str]:
        """
        Improved heuristic for names.
        """
        # Patterns like "cho anh [Name] vào", "mời [Name]", "tên là [Name]"
        patterns = [
            # Pattern 1: Capitalized words (2+ words), allowing single letter (e.g. Nguyen Van A)
            # Starts with Cap, followed by anything letters. Subsequent words also Start with Cap.
            r"(?:cho|mời|gặp|tên|anh|chị|ông|bà|bạn)\s+([A-ZĐ][a-zA-ZĐđà-ỹ]*(?:\s+[A-ZĐ][a-zA-ZĐđà-ỹ]*)+)", 
            # Pattern 2: Fallback for non-capitalized (2-4 words)
            r"(?:cho|mời|gặp|tên|anh|chị|ông|bà|bạn)\s+([a-zA-ZĐđà-ỹ]+(?:\s+[a-zA-ZĐđà-ỹ]+){1,3})" 
        ]
        
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                # Filter out common stop words if they got caught
                name = match.group(1).strip()
                if name.lower() not in ["vào", "cổng", "thăm", "làm", "việc"]:
                    return name.title()
        return None


class EntitySignalExtractor:
    """
    Analyzes text for entity signals and calculates boost scores for actions.
    Implementation of Architecture Review M3 (Entity Boost Clamping).
    """
    MAX_TOTAL_BOOST = 0.15
    
    # Mapping signals to actions they should boost
    SIGNAL_MAP = {
        "start_date": {"boost": [".create"], "weight": 0.05},
        "end_date": {"boost": [".create"], "weight": 0.05},
        "duration": {"boost": [".create"], "weight": 0.05},
        "visitor_name": {"boost": ["visitor.create"], "weight": 0.1},
        "amount": {"boost": [".create", ".balance"], "weight": 0.05},
    }

    def __init__(self):
        self.extractor = EntityExtractor()

    def get_boosts(self, text: str, candidates_actions: List[str]) -> Dict[str, float]:
        """
        Returns a dictionary of action_id -> boost_score
        """
        # For signal extraction, we might just look for slot presence without full extraction
        # But reusing extractor is safer.
        # We try to extract common slots
        common_slots = ["start_date", "end_date", "visitor_name", "amount", "duration"]
        extracted = self.extractor.extract(text, required_slots=common_slots)
        
        boosts = {action: 0.0 for action in candidates_actions}
        
        # Calculate raw boosts
        for slot, value in extracted.items():
            if slot in self.SIGNAL_MAP:
                config = self.SIGNAL_MAP[slot]
                target_pattern = config["boost"]
                weight = config["weight"]
                
                for action in candidates_actions:
                    # Check if action matches any pattern
                    if any(p in action for p in target_pattern):
                         boosts[action] += weight

        # Apply Clamping
        for action in boosts:
            boosts[action] = min(boosts[action], self.MAX_TOTAL_BOOST)
            
        return boosts
