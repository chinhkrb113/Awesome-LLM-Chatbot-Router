import datetime
import re
from typing import Optional, Tuple

class Validator:
    @staticmethod
    def validate(slot_name: str, value: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate giá trị cho slot.
        Returns: (is_valid, error_message, normalized_value)
        """
        value = value.strip()
        
        # 1. Date Validation
        if slot_name in ['date', 'start_date', 'end_date', 'visit_date']:
            return Validator._validate_date(value)
            
        # 2. Number Validation
        if slot_name in ['duration', 'amount', 'quantity']:
            return Validator._validate_number(value)
            
        # 3. Email Validation
        if slot_name == 'email':
            return Validator._validate_email(value)
            
        # 4. Generic non-empty
        if not value:
            return False, "Thông tin không được để trống.", None
            
        return True, "", value

    @staticmethod
    def _validate_date(value: str) -> Tuple[bool, str, Optional[str]]:
        # Thử parse ISO format trước (do Extractor trả về)
        try:
            datetime.date.fromisoformat(value)
            return True, "", value
        except ValueError:
            pass
            
        # Thử parse các format dd/mm/yyyy
        # Đây là logic fallback nếu user nhập tay vào prompt
        match = re.match(r'^(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?$', value)
        if match:
            try:
                day, month = int(match.group(1)), int(match.group(2))
                today = datetime.date.today()
                year = int(match.group(3)) if match.group(3) else today.year
                if match.group(3) and len(match.group(3)) == 2:
                    year += 2000
                
                dt = datetime.date(year, month, day)
                return True, "", dt.isoformat()
            except ValueError:
                return False, "Ngày tháng không hợp lệ (ví dụ 30/02).", None
        
        # Các từ khóa ngày tháng (hỗ trợ nhập tay lại)
        # Code lặp lại logic extractor một chút, nhưng cần thiết cho validation direct input
        # Để đơn giản, ta chỉ chấp nhận format chuẩn hoặc các từ khóa đơn giản
        keywords = {
            "hôm nay": 0, "bữa nay": 0,
            "ngày mai": 1, "mai": 1,
            "ngày kia": 2, "mốt": 2,
            "hôm qua": -1
        }
        val_lower = value.lower()
        if val_lower in keywords:
            dt = datetime.date.today() + datetime.timedelta(days=keywords[val_lower])
            return True, "", dt.isoformat()

        return False, "Vui lòng nhập ngày đúng định dạng dd/mm/yyyy hoặc các từ khóa như 'hôm nay', 'ngày mai'.", None

    @staticmethod
    def _validate_number(value: str) -> Tuple[bool, str, Optional[str]]:
        try:
            val = float(value)
            if val <= 0:
                return False, "Số lượng phải lớn hơn 0.", None
            # Trả về string số nguyên nếu là int
            if val.is_integer():
                return True, "", str(int(val))
            return True, "", str(val)
        except ValueError:
            return False, "Vui lòng nhập một con số hợp lệ.", None

    @staticmethod
    def _validate_email(value: str) -> Tuple[bool, str, Optional[str]]:
        if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value):
            return True, "", value
        return False, "Email không đúng định dạng.", None
