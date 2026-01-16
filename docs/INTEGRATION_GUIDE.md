# Hướng dẫn Tích hợp API mới vào Hybrid Intent Router

## Mục lục
1. [Tổng quan quy trình](#1-tổng-quan-quy-trình)
2. [Bước 1: Định nghĩa Action trong Catalog](#2-bước-1-định-nghĩa-action-trong-catalog)
3. [Bước 2: Cấu hình Keyword Rules](#3-bước-2-cấu-hình-keyword-rules)
4. [Bước 3: Cấu hình Pairwise Disambiguation](#4-bước-3-cấu-hình-pairwise-disambiguation)
5. [Bước 4: Mở rộng Entity Extractor](#5-bước-4-mở-rộng-entity-extractor)
6. [Bước 5: Implement API Connector](#6-bước-5-implement-api-connector)
7. [Bước 6: Testing và Validation](#7-bước-6-testing-và-validation)
8. [Ví dụ hoàn chỉnh: Tích hợp 5 API mẫu](#8-ví-dụ-hoàn-chỉnh-tích-hợp-5-api-mẫu)
9. [Checklist tích hợp](#9-checklist-tích-hợp)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Tổng quan quy trình

### 1.1 Luồng xử lý của hệ thống

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   User       │───▶│   Router     │───▶│ Action Flow  │───▶│   External   │
│   Input      │    │   (Intent)   │    │   Engine     │    │   API        │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      │                    │                   │                    │
      │              Xác định action     Thu thập slots       Gọi API thực
      │              phù hợp nhất        cần thiết            tế của hệ thống
      │                    │                   │                    │
      └────────────────────┴───────────────────┴────────────────────┘
```

### 1.2 Các file cần cập nhật

| File | Mục đích | Bắt buộc |
|------|----------|----------|
| `config/action_catalog.yaml` | Định nghĩa action, slots, seed phrases | ✅ Có |
| `config/keyword_rules.yaml` | Rules cho deterministic matching | ✅ Có |
| `config/pairwise_rules.yaml` | Xử lý conflict giữa các actions | ⚠️ Nếu có conflict |
| `app/action_flow/entity_extractor.py` | Extract entities mới | ⚠️ Nếu có entity type mới |
| `app/connectors/*.py` | Gọi external APIs | ✅ Có |
| `tests/*.py` | Test cases | ✅ Có |

---

## 2. Bước 1: Định nghĩa Action trong Catalog

### 2.1 Cấu trúc một Action

```yaml
- action_id: <domain>.<intent_type>
  domain: <domain_name>
  business_description: >
    Mô tả chi tiết action này làm gì, dùng cho semantic matching.
  seed_phrases:
    - "câu mẫu 1"
    - "câu mẫu 2"
  required_slots:
    - slot_1
    - slot_2
  optional_slots:
    - slot_3
  typical_entities:
    - entity_type_1
  examples:
    - "Ví dụ câu người dùng có thể nói"
```

### 2.2 Quy tắc đặt tên

#### action_id
- Format: `<domain>.<intent_type>`
- Domain: nhóm chức năng (leave, visitor, request, approval, hr, finance...)
- Intent type: loại hành động (create, status, cancel, balance, list, update...)

```yaml
# Ví dụ tốt
action_id: request.late_arrival      # Domain: request, Intent: late_arrival
action_id: hr.salary_increase        # Domain: hr, Intent: salary_increase
action_id: finance.expense_claim     # Domain: finance, Intent: expense_claim

# Ví dụ không tốt
action_id: xindimuon                  # Không có domain, không rõ ràng
action_id: request_late              # Dùng underscore thay vì dot
```

### 2.3 Viết seed_phrases hiệu quả

**Nguyên tắc:**
- Tối thiểu 8-12 phrases cho mỗi action
- Bao gồm cả formal và informal
- Bao gồm các biến thể viết tắt, typo phổ biến
- Không trùng lặp với action khác

```yaml
# Ví dụ: request.late_arrival
seed_phrases:
  # Formal
  - "xin phép đi làm muộn"
  - "đăng ký đi muộn"
  - "báo cáo đi trễ"
  
  # Informal
  - "hôm nay tôi đến muộn"
  - "sáng nay đi muộn"
  - "tôi sẽ đến trễ"
  
  # Viết tắt / Slang
  - "xin đi muộn"
  - "báo đi muộn"
  - "đi trễ nha"
  
  # Có context
  - "kẹt xe nên đi muộn"
  - "con ốm nên đến trễ"
```

### 2.4 Định nghĩa Slots

#### Required Slots
Slots bắt buộc phải có trước khi thực hiện action.

```yaml
required_slots:
  - date              # Ngày áp dụng
  - expected_time     # Thời gian dự kiến
  - reason            # Lý do
```

#### Optional Slots
Slots không bắt buộc, có thể bỏ qua.

```yaml
optional_slots:
  - duration_minutes  # Số phút muộn (có thể tự tính)
  - attachment        # File đính kèm
```

#### Typical Entities
Các loại entity thường xuất hiện trong input của action này.

```yaml
typical_entities:
  - date              # "hôm nay", "ngày mai", "15/01/2026"
  - time              # "9h30", "10 giờ"
  - duration          # "30 phút", "1 tiếng"
  - person_name       # Tên người
  - money             # "10 triệu", "500k"
  - percentage        # "10%", "mười phần trăm"
```

### 2.5 Ví dụ hoàn chỉnh một Action

```yaml
- action_id: request.late_arrival
  domain: request
  business_description: >
    Đăng ký xin đi làm muộn với lý do cụ thể. Nhân viên cần cung cấp
    ngày đi muộn, thời gian dự kiến đến công ty, và lý do. Hệ thống
    sẽ gửi thông báo cho quản lý trực tiếp để phê duyệt.
  seed_phrases:
    - "xin đi muộn"
    - "xin phép đi làm muộn"
    - "đăng ký đi muộn"
    - "báo đi muộn"
    - "hôm nay tôi đến muộn"
    - "sáng nay đi muộn"
    - "tôi sẽ đến trễ"
    - "xin đến trễ"
    - "báo cáo đi trễ"
    - "kẹt xe nên đi muộn"
    - "đi muộn vì việc gia đình"
  required_slots:
    - date
    - expected_arrival_time
    - reason
  optional_slots:
    - duration_minutes
    - notify_manager
  typical_entities:
    - date
    - time
    - duration
  examples:
    - "Sáng nay tôi đi muộn 30 phút vì kẹt xe"
    - "Xin đi muộn ngày mai, dự kiến 10h đến"
    - "Báo đi trễ hôm nay, con ốm phải đưa đi khám"
```

---

## 3. Bước 2: Cấu hình Keyword Rules

### 3.1 Cấu trúc Rule

```yaml
<action_id>:
  strong_keywords:    # Keywords chắc chắn thuộc action này
    - "keyword 1"
  weak_keywords:      # Keywords có thể thuộc action này
    - "keyword 2"
  negative_keywords:  # Keywords KHÔNG thuộc action này
    - "keyword 3"
  special_patterns:   # Patterns đặc biệt (dùng * làm wildcard)
    - "pattern * here"
```

### 3.2 Trọng số mặc định

| Loại | Trọng số | Max hits | Giải thích |
|------|----------|----------|------------|
| strong_keywords | +0.30 | 3 | Tối đa +0.90 |
| weak_keywords | +0.12 | 3 | Tối đa +0.36 |
| negative_keywords | -0.35 | 2 | Tối đa -0.70 |
| special_patterns | +0.45 | 2 | Tối đa +0.90 |

### 3.3 Nguyên tắc viết Keywords

#### Strong Keywords
- Cụm từ đặc trưng, chỉ xuất hiện trong action này
- Thường là 2-3 từ

```yaml
strong_keywords:
  - "đi muộn"           # Cụm từ đặc trưng
  - "đến trễ"           # Biến thể
  - "xin phép trễ"      # Formal
```

#### Weak Keywords
- Từ đơn hoặc cụm từ chung chung hơn
- Có thể xuất hiện ở nhiều actions

```yaml
weak_keywords:
  - "muộn"              # Từ đơn
  - "trễ"               # Từ đơn
  - "sáng nay"          # Context chung
```

#### Negative Keywords
- Keywords của các actions dễ nhầm lẫn
- Giúp phân biệt với actions tương tự

```yaml
# Cho request.late_arrival
negative_keywords:
  - "deadline"          # Thuộc request.late_deadline
  - "task"              # Thuộc request.late_deadline
  - "nộp bài"           # Thuộc request.late_deadline
  - "lương"             # Thuộc hr.salary_increase
```

### 3.4 Special Patterns

Dùng `*` làm wildcard để match các patterns phức tạp.

```yaml
special_patterns:
  - "xin đi muộn *"           # Match: "xin đi muộn hôm nay"
  - "* đến trễ *"             # Match: "tôi sẽ đến trễ 30 phút"
  - "đi muộn vì *"            # Match: "đi muộn vì kẹt xe"
  - "* giờ * mới đến"         # Match: "10 giờ sáng mới đến"
```

### 3.5 Ví dụ hoàn chỉnh Keyword Rules

```yaml
request.late_arrival:
  strong_keywords:
    - "đi muộn"
    - "đến trễ"
    - "đi làm trễ"
    - "xin phép trễ"
    - "báo trễ"
  weak_keywords:
    - "muộn"
    - "trễ"
    - "sáng nay"
    - "kẹt xe"
    - "đến công ty"
  negative_keywords:
    - "deadline"
    - "task"
    - "nộp bài"
    - "dự án"
    - "lương"
    - "nghỉ phép"
  special_patterns:
    - "xin đi muộn *"
    - "* đến trễ *"
    - "đi muộn vì *"
    - "* giờ * mới đến"
    - "sáng nay * muộn"
```

---

## 4. Bước 3: Cấu hình Pairwise Disambiguation

### 4.1 Khi nào cần Pairwise Rules?

Khi 2 actions có:
- Seed phrases tương tự
- Keywords overlap
- Dễ gây nhầm lẫn cho người dùng

### 4.2 Cấu trúc Pairwise Rule

```yaml
pairwise_disambiguation:
  - pair: ["action_1", "action_2"]
    signals:
      prefer_action_1: ["keyword_a", "keyword_b"]
      prefer_action_2: ["keyword_c", "keyword_d"]

default_bias:
  dangerous_action: -0.05  # Giảm score cho actions nguy hiểm
```

### 4.3 Ví dụ: Phân biệt "đi muộn" vs "trễ deadline"

```yaml
pairwise_disambiguation:
  - pair: ["request.late_arrival", "request.late_deadline"]
    signals:
      prefer_late_arrival:
        - "đi muộn"
        - "đến trễ"
        - "đến công ty"
        - "kẹt xe"
        - "giờ làm"
      prefer_late_deadline:
        - "deadline"
        - "task"
        - "nộp bài"
        - "dự án"
        - "hoàn thành"
        - "gia hạn"
```

### 4.4 Default Bias

Dùng để giảm score cho các actions "nguy hiểm" (cancel, delete, reject...).

```yaml
default_bias:
  # Actions hủy/xóa cần user confirm rõ ràng hơn
  leave.cancel: -0.05
  visitor.cancel: -0.05
  request.cancel: -0.05
  
  # Actions có impact lớn
  hr.terminate_employee: -0.10
  finance.large_transfer: -0.08
```

---

## 5. Bước 4: Mở rộng Entity Extractor

### 5.1 Khi nào cần mở rộng?

Khi action mới cần extract các entity types chưa được hỗ trợ:
- `money` - Số tiền
- `percentage` - Phần trăm
- `phone` - Số điện thoại
- `address` - Địa chỉ
- Custom entities

### 5.2 Entity Types hiện có

| Entity Type | Ví dụ | File |
|-------------|-------|------|
| date | "hôm nay", "15/01/2026" | entity_extractor.py |
| time | "9h30", "10 giờ" | Cần thêm |
| duration | "30 phút", "2 tiếng" | Cần thêm |
| number | "10", "một trăm" | entity_extractor.py |
| email | "abc@company.com" | entity_extractor.py |
| person_name | "Nguyễn Văn A" | entity_extractor.py |

### 5.3 Thêm Entity Extractor mới

Tạo file `app/action_flow/entity_extractor_extended.py`:

```python
import re
from typing import Optional, List
from datetime import datetime, timedelta

class ExtendedEntityExtractor:
    """
    Mở rộng EntityExtractor với các entity types mới.
    """
    
    def extract_time(self, text: str) -> Optional[str]:
        """
        Extract thời gian từ text.
        
        Ví dụ:
        - "9h30" -> "09:30"
        - "10 giờ" -> "10:00"
        - "9 giờ 30 phút" -> "09:30"
        - "9:30" -> "09:30"
        """
        text_lower = text.lower()
        
        # Pattern 1: 9h30, 10h, 9h30p
        match = re.search(r'(\d{1,2})h(\d{0,2})p?', text_lower)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            return f"{hour:02d}:{minute:02d}"
        
        # Pattern 2: 9 giờ 30 phút, 10 giờ
        match = re.search(r'(\d{1,2})\s*giờ\s*(\d{0,2})\s*(?:phút)?', text_lower)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            return f"{hour:02d}:{minute:02d}"
        
        # Pattern 3: 9:30, 10:00
        match = re.search(r'(\d{1,2}):(\d{2})', text_lower)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            return f"{hour:02d}:{minute:02d}"
        
        return None
    
    def extract_money(self, text: str) -> Optional[dict]:
        """
        Extract số tiền từ text.
        
        Ví dụ:
        - "10 triệu" -> {"amount": 10000000, "currency": "VND"}
        - "500k" -> {"amount": 500000, "currency": "VND"}
        - "10,000,000" -> {"amount": 10000000, "currency": "VND"}
        - "$100" -> {"amount": 100, "currency": "USD"}
        """
        text_lower = text.lower().replace(',', '').replace('.', '')
        
        # Pattern 1: X triệu, X tr
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:triệu|tr)', text_lower)
        if match:
            amount = float(match.group(1)) * 1_000_000
            return {"amount": int(amount), "currency": "VND"}
        
        # Pattern 2: Xk, X nghìn
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:k|nghìn|ngàn)', text_lower)
        if match:
            amount = float(match.group(1)) * 1_000
            return {"amount": int(amount), "currency": "VND"}
        
        # Pattern 3: X tỷ
        match = re.search(r'(\d+(?:\.\d+)?)\s*tỷ', text_lower)
        if match:
            amount = float(match.group(1)) * 1_000_000_000
            return {"amount": int(amount), "currency": "VND"}
        
        # Pattern 4: $X, X USD
        match = re.search(r'\$(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:usd|đô)', text_lower)
        if match:
            amount = float(match.group(1) or match.group(2))
            return {"amount": int(amount), "currency": "USD"}
        
        # Pattern 5: Số thuần (VND)
        match = re.search(r'(\d{6,})', text.replace(',', '').replace('.', ''))
        if match:
            return {"amount": int(match.group(1)), "currency": "VND"}
        
        return None
    
    def extract_percentage(self, text: str) -> Optional[float]:
        """
        Extract phần trăm từ text.
        
        Ví dụ:
        - "10%" -> 10.0
        - "10 phần trăm" -> 10.0
        - "mười phần trăm" -> 10.0
        """
        text_lower = text.lower()
        
        # Pattern 1: X%
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', text_lower)
        if match:
            return float(match.group(1))
        
        # Pattern 2: X phần trăm
        match = re.search(r'(\d+(?:\.\d+)?)\s*phần\s*trăm', text_lower)
        if match:
            return float(match.group(1))
        
        # Pattern 3: Số chữ
        word_to_num = {
            "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
            "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
            "mười lăm": 15, "hai mươi": 20, "ba mươi": 30
        }
        for word, num in word_to_num.items():
            if f"{word} phần trăm" in text_lower:
                return float(num)
        
        return None
    
    def extract_duration(self, text: str) -> Optional[dict]:
        """
        Extract khoảng thời gian từ text.
        
        Ví dụ:
        - "30 phút" -> {"minutes": 30}
        - "2 tiếng" -> {"hours": 2}
        - "1 tiếng 30 phút" -> {"hours": 1, "minutes": 30}
        """
        text_lower = text.lower()
        result = {}
        
        # Extract giờ
        match = re.search(r'(\d+)\s*(?:tiếng|giờ|h)', text_lower)
        if match:
            result["hours"] = int(match.group(1))
        
        # Extract phút
        match = re.search(r'(\d+)\s*(?:phút|p)', text_lower)
        if match:
            result["minutes"] = int(match.group(1))
        
        return result if result else None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """
        Extract số điện thoại từ text.
        
        Ví dụ:
        - "0901234567" -> "0901234567"
        - "090 123 4567" -> "0901234567"
        - "+84 901234567" -> "+84901234567"
        """
        # Remove spaces and dashes
        cleaned = re.sub(r'[\s\-\.]', '', text)
        
        # Vietnam phone patterns
        patterns = [
            r'(\+84\d{9,10})',           # +84xxxxxxxxx
            r'(0\d{9,10})',               # 0xxxxxxxxx
            r'(84\d{9,10})',              # 84xxxxxxxxx
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                return match.group(1)
        
        return None
```

### 5.4 Tích hợp vào EntityExtractor chính

Cập nhật `app/action_flow/entity_extractor.py`:

```python
from app.action_flow.entity_extractor_extended import ExtendedEntityExtractor

class EntityExtractor:
    def __init__(self):
        self.extended = ExtendedEntityExtractor()

    def extract(self, text: str, required_slots: List[str]) -> Dict[str, Any]:
        extracted = {}
        text_lower = text.lower()
        
        # ... existing code ...
        
        # 5. Time Extraction (MỚI)
        if any(s in ['time', 'expected_arrival_time', 'start_time', 'end_time'] 
               for s in required_slots):
            time_val = self.extended.extract_time(text_lower)
            if time_val:
                for slot in required_slots:
                    if 'time' in slot and slot not in extracted:
                        extracted[slot] = time_val
                        break
        
        # 6. Money Extraction (MỚI)
        if any(s in ['amount', 'salary', 'proposed_amount', 'budget'] 
               for s in required_slots):
            money_val = self.extended.extract_money(text)
            if money_val:
                for slot in required_slots:
                    if slot in ['amount', 'salary', 'proposed_amount', 'budget']:
                        extracted[slot] = money_val
                        break
        
        # 7. Percentage Extraction (MỚI)
        if 'percentage' in required_slots or 'increase_rate' in required_slots:
            pct_val = self.extended.extract_percentage(text)
            if pct_val:
                slot = 'percentage' if 'percentage' in required_slots else 'increase_rate'
                extracted[slot] = pct_val
        
        # 8. Duration Extraction (MỚI)
        if any(s in ['duration', 'duration_minutes'] for s in required_slots):
            dur_val = self.extended.extract_duration(text)
            if dur_val:
                # Convert to minutes
                total_minutes = dur_val.get('hours', 0) * 60 + dur_val.get('minutes', 0)
                extracted['duration_minutes'] = total_minutes
        
        return extracted
```

---

## 6. Bước 5: Implement API Connector

### 6.1 Kiến trúc Connector

```
app/connectors/
├── __init__.py
├── base.py              # Base class
├── registry.py          # Connector registry
├── hr_connector.py      # HR system APIs
├── finance_connector.py # Finance system APIs
└── facility_connector.py # Facility system APIs
```

### 6.2 Base Connector

Tạo `app/connectors/base.py`:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import httpx
import logging

logger = logging.getLogger(__name__)

class ConnectorStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    TIMEOUT = "timeout"

@dataclass
class ConnectorResponse:
    status: ConnectorStatus
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None

class BaseConnector(ABC):
    """
    Base class cho tất cả API connectors.
    """
    
    def __init__(self, base_url: str, api_key: str = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            headers["Content-Type"] = "application/json"
            
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout
            )
        return self._client
    
    @abstractmethod
    def get_endpoint(self, action_id: str) -> str:
        """Map action_id to API endpoint."""
        pass
    
    @abstractmethod
    def transform_slots(self, action_id: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        """Transform slots to API request format."""
        pass
    
    async def execute(self, action_id: str, slots: Dict[str, Any]) -> ConnectorResponse:
        """
        Execute API call.
        """
        try:
            endpoint = self.get_endpoint(action_id)
            payload = self.transform_slots(action_id, slots)
            
            logger.info(f"Calling {endpoint} with payload: {payload}")
            
            response = await self.client.post(endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return ConnectorResponse(
                status=ConnectorStatus.SUCCESS,
                data=data,
                request_id=data.get("request_id")
            )
            
        except httpx.TimeoutException:
            logger.error(f"Timeout calling {action_id}")
            return ConnectorResponse(
                status=ConnectorStatus.TIMEOUT,
                error_message="Request timeout"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            return ConnectorResponse(
                status=ConnectorStatus.FAILED,
                error_message=f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            logger.error(f"Error calling {action_id}: {e}")
            return ConnectorResponse(
                status=ConnectorStatus.FAILED,
                error_message=str(e)
            )
    
    async def close(self):
        if self._client:
            await self._client.aclose()
```

### 6.3 HR System Connector

Tạo `app/connectors/hr_connector.py`:

```python
from typing import Dict, Any
from app.connectors.base import BaseConnector

class HRConnector(BaseConnector):
    """
    Connector cho HR System APIs.
    
    Supported actions:
    - request.late_arrival
    - request.early_leave
    - hr.salary_increase
    - hr.promotion_request
    - hr.training_request
    """
    
    # Mapping action_id -> API endpoint
    ENDPOINT_MAP = {
        "request.late_arrival": "/api/v1/attendance/late-arrival",
        "request.early_leave": "/api/v1/attendance/early-leave",
        "hr.salary_increase": "/api/v1/compensation/salary-review",
        "hr.promotion_request": "/api/v1/career/promotion",
        "hr.training_request": "/api/v1/learning/training-request",
        "hr.certificate_request": "/api/v1/documents/certificate",
    }
    
    def get_endpoint(self, action_id: str) -> str:
        endpoint = self.ENDPOINT_MAP.get(action_id)
        if not endpoint:
            raise ValueError(f"Unknown action: {action_id}")
        return endpoint
    
    def transform_slots(self, action_id: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform chatbot slots to API request format.
        """
        # Extract raw values from SlotValue objects
        raw_slots = {}
        for key, value in slots.items():
            if hasattr(value, 'value'):
                raw_slots[key] = value.value
            else:
                raw_slots[key] = value
        
        # Action-specific transformations
        if action_id == "request.late_arrival":
            return {
                "attendance_date": raw_slots.get("date"),
                "expected_arrival": raw_slots.get("expected_arrival_time"),
                "reason": raw_slots.get("reason"),
                "duration_minutes": raw_slots.get("duration_minutes"),
            }
        
        elif action_id == "hr.salary_increase":
            amount = raw_slots.get("proposed_amount", {})
            return {
                "proposed_salary": amount.get("amount") if isinstance(amount, dict) else amount,
                "currency": amount.get("currency", "VND") if isinstance(amount, dict) else "VND",
                "justification": raw_slots.get("reason"),
                "effective_date": raw_slots.get("effective_date"),
            }
        
        elif action_id == "hr.training_request":
            return {
                "course_name": raw_slots.get("course_name"),
                "training_date": raw_slots.get("date"),
                "duration_days": raw_slots.get("duration_days"),
                "reason": raw_slots.get("reason"),
                "estimated_cost": raw_slots.get("budget"),
            }
        
        # Default: pass through
        return raw_slots
```

### 6.4 Connector Registry

Tạo `app/connectors/registry.py`:

```python
from typing import Dict, Optional
from app.connectors.base import BaseConnector
from app.connectors.hr_connector import HRConnector
import os

class ConnectorRegistry:
    """
    Registry quản lý tất cả connectors.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connectors = {}
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        """Initialize all connectors from environment."""
        if self._initialized:
            return
        
        # HR System
        hr_url = os.getenv("HR_SYSTEM_URL", "http://hr-system:8080")
        hr_key = os.getenv("HR_SYSTEM_API_KEY", "")
        self._connectors["hr"] = HRConnector(hr_url, hr_key)
        
        # Finance System
        # finance_url = os.getenv("FINANCE_SYSTEM_URL", "http://finance:8080")
        # self._connectors["finance"] = FinanceConnector(finance_url)
        
        # Facility System
        # facility_url = os.getenv("FACILITY_SYSTEM_URL", "http://facility:8080")
        # self._connectors["facility"] = FacilityConnector(facility_url)
        
        self._initialized = True
    
    def get_connector(self, action_id: str) -> Optional[BaseConnector]:
        """
        Get connector for action based on domain.
        """
        domain = action_id.split(".")[0] if "." in action_id else action_id
        
        domain_mapping = {
            "request": "hr",
            "hr": "hr",
            "leave": "hr",
            "finance": "finance",
            "expense": "finance",
            "facility": "facility",
            "visitor": "facility",
        }
        
        connector_key = domain_mapping.get(domain)
        return self._connectors.get(connector_key)
    
    async def close_all(self):
        """Close all connector connections."""
        for connector in self._connectors.values():
            await connector.close()

# Global instance
def get_registry() -> ConnectorRegistry:
    registry = ConnectorRegistry()
    registry.initialize()
    return registry
```

### 6.5 Tích hợp vào Action Flow Engine

Cập nhật `app/action_flow/engine.py`:

```python
from app.connectors.registry import get_registry

class ActionFlowEngine:
    def __init__(self, config_loader: ConfigLoader):
        self.loader = config_loader
        self.states: Dict[str, ActionState] = {}
        self.extractor = EntityExtractor()
        self.validator = Validator()
        self.connector_registry = get_registry()  # THÊM MỚI

    async def _commit_action(self, state: ActionState):
        """
        Commit action và gọi external API.
        """
        connector = self.connector_registry.get_connector(state.action_id)
        
        if connector:
            # Gọi external API
            response = await connector.execute(state.action_id, state.slots)
            
            if response.status.value == "success":
                state.status = ActionStatus.COMMITTED
                state.message = f"✅ Yêu cầu đã được gửi thành công!\nMã yêu cầu: {response.request_id}"
            else:
                state.status = ActionStatus.FAILED
                state.message = f"❌ Có lỗi xảy ra: {response.error_message}"
        else:
            # Fallback: Mock commit
            state.status = ActionStatus.COMMITTED
            state.message = "✅ Yêu cầu đã được ghi nhận (demo mode)."
        
        state.buttons = []
```

---

## 7. Bước 6: Testing và Validation

### 7.1 Unit Tests cho Action mới

Tạo `tests/test_new_actions.py`:

```python
import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.utils.config_loader import ConfigLoader
from app.router.router_final import RouterFinal
from app.router.embed_config import EmbedConfig
from app.core.models import UserRequest

class TestNewActions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = ConfigLoader(
            "config/action_catalog.yaml",
            "config/keyword_rules.yaml"
        )
        cls.router = RouterFinal(cls.loader, EmbedConfig(), enable_v2=True)
    
    def test_late_arrival_routing(self):
        """Test routing cho request.late_arrival"""
        test_cases = [
            ("xin đi muộn hôm nay", "request.late_arrival"),
            ("sáng nay tôi đến trễ 30 phút", "request.late_arrival"),
            ("báo đi muộn vì kẹt xe", "request.late_arrival"),
            ("10h mới đến công ty được", "request.late_arrival"),
        ]
        
        for text, expected_action in test_cases:
            with self.subTest(text=text):
                request = UserRequest(text=text)
                result = self.router.route(request)
                
                top_action = result.top_actions[0].action_id
                self.assertEqual(
                    top_action, expected_action,
                    f"Expected {expected_action}, got {top_action} for '{text}'"
                )
    
    def test_late_deadline_vs_late_arrival(self):
        """Test phân biệt late_deadline vs late_arrival"""
        # Should be late_deadline
        deadline_texts = [
            "xin trễ deadline task ABC",
            "gia hạn nộp bài",
            "xin thêm thời gian hoàn thành dự án",
        ]
        
        for text in deadline_texts:
            with self.subTest(text=text):
                request = UserRequest(text=text)
                result = self.router.route(request)
                top_action = result.top_actions[0].action_id
                self.assertEqual(top_action, "request.late_deadline")
        
        # Should be late_arrival
        arrival_texts = [
            "xin đi muộn hôm nay",
            "sáng nay đến trễ",
        ]
        
        for text in arrival_texts:
            with self.subTest(text=text):
                request = UserRequest(text=text)
                result = self.router.route(request)
                top_action = result.top_actions[0].action_id
                self.assertEqual(top_action, "request.late_arrival")
    
    def test_salary_increase_routing(self):
        """Test routing cho hr.salary_increase"""
        test_cases = [
            ("xin tăng lương", "hr.salary_increase"),
            ("đề xuất review lương", "hr.salary_increase"),
            ("muốn tăng lương 10%", "hr.salary_increase"),
        ]
        
        for text, expected_action in test_cases:
            with self.subTest(text=text):
                request = UserRequest(text=text)
                result = self.router.route(request)
                top_action = result.top_actions[0].action_id
                self.assertEqual(top_action, expected_action)

class TestEntityExtraction(unittest.TestCase):
    def setUp(self):
        from app.action_flow.entity_extractor_extended import ExtendedEntityExtractor
        self.extractor = ExtendedEntityExtractor()
    
    def test_time_extraction(self):
        test_cases = [
            ("9h30", "09:30"),
            ("10 giờ", "10:00"),
            ("9 giờ 30 phút", "09:30"),
            ("14:45", "14:45"),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_time(text)
                self.assertEqual(result, expected)
    
    def test_money_extraction(self):
        test_cases = [
            ("10 triệu", {"amount": 10000000, "currency": "VND"}),
            ("500k", {"amount": 500000, "currency": "VND"}),
            ("$100", {"amount": 100, "currency": "USD"}),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_money(text)
                self.assertEqual(result, expected)
    
    def test_percentage_extraction(self):
        test_cases = [
            ("10%", 10.0),
            ("15 phần trăm", 15.0),
            ("mười phần trăm", 10.0),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.extractor.extract_percentage(text)
                self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
```

### 7.2 E2E Test cho Flow hoàn chỉnh

Tạo `tests/test_e2e_new_actions.py`:

```python
import unittest
from fastapi.testclient import TestClient
from app.main import app

class TestE2ELateArrival(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.session_id = "test_late_arrival_001"
    
    def test_full_late_arrival_flow(self):
        """Test complete flow: Route -> Start -> Collect Slots -> Confirm"""
        
        # Step 1: Route
        print("\n[E2E] Step 1: Routing...")
        route_res = self.client.post("/route", json={
            "text": "xin đi muộn hôm nay vì kẹt xe"
        })
        self.assertEqual(route_res.status_code, 200)
        data = route_res.json()
        
        top_action = data["top_actions"][0]["action_id"]
        self.assertEqual(top_action, "request.late_arrival")
        print(f"   Routed to: {top_action}")
        
        # Step 2: Start Action
        print("[E2E] Step 2: Starting action...")
        start_res = self.client.post("/action/start", json={
            "session_id": self.session_id,
            "action_id": top_action,
            "initial_text": "xin đi muộn hôm nay vì kẹt xe"
        })
        self.assertEqual(start_res.status_code, 200)
        state = start_res.json()
        
        # Should have extracted date from "hôm nay"
        self.assertIn("date", state["slots"])
        print(f"   Auto-extracted: date={state['slots'].get('date')}")
        
        # Step 3: Provide expected_arrival_time
        print("[E2E] Step 3: Providing arrival time...")
        interact_res = self.client.post("/action/interact", json={
            "session_id": self.session_id,
            "text": "10h30"
        })
        state = interact_res.json()
        self.assertIn("expected_arrival_time", state["slots"])
        print(f"   Collected: expected_arrival_time={state['slots'].get('expected_arrival_time')}")
        
        # Step 4: Provide reason (if not already extracted)
        if "reason" not in state["slots"]:
            print("[E2E] Step 4: Providing reason...")
            interact_res = self.client.post("/action/interact", json={
                "session_id": self.session_id,
                "text": "kẹt xe trên đường"
            })
            state = interact_res.json()
        
        # Should be in DRAFT status now
        self.assertEqual(state["status"], "DRAFT")
        print(f"   Status: {state['status']}")
        
        # Step 5: Confirm
        print("[E2E] Step 5: Confirming...")
        confirm_res = self.client.post("/action/interact", json={
            "session_id": self.session_id,
            "text": "xác nhận"
        })
        state = confirm_res.json()
        self.assertEqual(state["status"], "COMMITTED")
        print(f"   Final status: {state['status']}")
        print("✅ [E2E] Late arrival flow completed!")

class TestE2ESalaryIncrease(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.session_id = "test_salary_001"
    
    def test_full_salary_increase_flow(self):
        """Test salary increase request flow"""
        
        # Step 1: Route
        route_res = self.client.post("/route", json={
            "text": "tôi muốn xin tăng lương 15%"
        })
        data = route_res.json()
        top_action = data["top_actions"][0]["action_id"]
        self.assertEqual(top_action, "hr.salary_increase")
        
        # Step 2: Start
        start_res = self.client.post("/action/start", json={
            "session_id": self.session_id,
            "action_id": top_action,
            "initial_text": "tôi muốn xin tăng lương 15%"
        })
        state = start_res.json()
        
        # Should have extracted percentage
        # Continue collecting other required slots...
        
        print("✅ [E2E] Salary increase flow started!")

if __name__ == "__main__":
    unittest.main()
```

### 7.3 Benchmark Routing Accuracy

Tạo `scripts/benchmark_new_actions.py`:

```python
"""
Benchmark routing accuracy cho các actions mới.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.utils.config_loader import ConfigLoader
from app.router.router_final import RouterFinal
from app.router.embed_config import EmbedConfig
from app.core.models import UserRequest

# Test dataset
TEST_CASES = [
    # request.late_arrival
    {"text": "xin đi muộn hôm nay", "expected": "request.late_arrival"},
    {"text": "sáng nay tôi đến trễ 30 phút", "expected": "request.late_arrival"},
    {"text": "báo đi muộn vì kẹt xe", "expected": "request.late_arrival"},
    {"text": "10h mới đến công ty được", "expected": "request.late_arrival"},
    {"text": "xin phép đi làm trễ", "expected": "request.late_arrival"},
    
    # request.late_deadline
    {"text": "xin trễ deadline", "expected": "request.late_deadline"},
    {"text": "gia hạn nộp bài", "expected": "request.late_deadline"},
    {"text": "xin thêm thời gian hoàn thành task", "expected": "request.late_deadline"},
    
    # hr.salary_increase
    {"text": "xin tăng lương", "expected": "hr.salary_increase"},
    {"text": "đề xuất review lương", "expected": "hr.salary_increase"},
    {"text": "muốn tăng lương 10%", "expected": "hr.salary_increase"},
    
    # request.allowance
    {"text": "xin trợ cấp công tác", "expected": "request.allowance"},
    {"text": "đề xuất phụ cấp đi lại", "expected": "request.allowance"},
    
    # Existing actions (regression test)
    {"text": "xin nghỉ phép ngày mai", "expected": "leave.create"},
    {"text": "đơn nghỉ của tôi sao rồi", "expected": "leave.status"},
    {"text": "tạo khách mời", "expected": "visitor.create"},
]

def run_benchmark():
    loader = ConfigLoader(
        "config/action_catalog.yaml",
        "config/keyword_rules.yaml"
    )
    router = RouterFinal(loader, EmbedConfig(), enable_v2=True)
    
    correct = 0
    total = len(TEST_CASES)
    errors = []
    
    print("=" * 60)
    print("ROUTING ACCURACY BENCHMARK")
    print("=" * 60)
    
    for case in TEST_CASES:
        text = case["text"]
        expected = case["expected"]
        
        request = UserRequest(text=text)
        result = router.route(request)
        actual = result.top_actions[0].action_id
        score = result.top_actions[0].final_score
        
        if actual == expected:
            correct += 1
            status = "✅"
        else:
            status = "❌"
            errors.append({
                "text": text,
                "expected": expected,
                "actual": actual,
                "score": score
            })
        
        print(f"{status} '{text[:40]:<40}' -> {actual} ({score:.2f})")
    
    accuracy = correct / total * 100
    print("=" * 60)
    print(f"ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print("=" * 60)
    
    if errors:
        print("\nERRORS:")
        for err in errors:
            print(f"  - '{err['text']}'")
            print(f"    Expected: {err['expected']}")
            print(f"    Actual:   {err['actual']} ({err['score']:.2f})")
    
    return accuracy

if __name__ == "__main__":
    run_benchmark()
```

### 7.4 Chạy Tests

```bash
# Unit tests
python -m pytest tests/test_new_actions.py -v

# E2E tests
python -m pytest tests/test_e2e_new_actions.py -v

# Benchmark
python scripts/benchmark_new_actions.py

# Coverage report
python -m pytest tests/ --cov=app --cov-report=html
```

---

## 8. Ví dụ hoàn chỉnh: Tích hợp 5 API mẫu

### 8.1 Danh sách 5 API mẫu

| # | Action ID | Mô tả | External API |
|---|-----------|-------|--------------|
| 1 | request.late_arrival | Xin đi muộn | HR System |
| 2 | request.late_deadline | Xin trễ deadline | Project Management |
| 3 | hr.salary_increase | Xin tăng lương | HR System |
| 4 | request.allowance | Xin trợ cấp | Finance System |
| 5 | request.overtime | Đăng ký làm thêm giờ | HR System |

### 8.2 File config/action_catalog.yaml (Thêm 5 actions)

```yaml
# === THÊM VÀO CUỐI FILE action_catalog.yaml ===

  # ============================================
  # REQUEST DOMAIN - Các yêu cầu chung
  # ============================================
  
  - action_id: request.late_arrival
    domain: request
    business_description: >
      Đăng ký xin đi làm muộn với lý do cụ thể. Nhân viên cần cung cấp
      ngày đi muộn, thời gian dự kiến đến công ty, và lý do. Hệ thống
      sẽ gửi thông báo cho quản lý trực tiếp để phê duyệt.
    seed_phrases:
      - "xin đi muộn"
      - "xin phép đi làm muộn"
      - "đăng ký đi muộn"
      - "báo đi muộn"
      - "hôm nay tôi đến muộn"
      - "sáng nay đi muộn"
      - "tôi sẽ đến trễ"
      - "xin đến trễ"
      - "báo cáo đi trễ"
      - "kẹt xe nên đi muộn"
      - "đi muộn vì việc gia đình"
      - "xin phép trễ giờ làm"
    required_slots:
      - date
      - expected_arrival_time
      - reason
    optional_slots:
      - duration_minutes
      - notify_manager
    typical_entities:
      - date
      - time
      - duration
    examples:
      - "Sáng nay tôi đi muộn 30 phút vì kẹt xe"
      - "Xin đi muộn ngày mai, dự kiến 10h đến"
      - "Báo đi trễ hôm nay, con ốm phải đưa đi khám"

  - action_id: request.late_deadline
    domain: request
    business_description: >
      Yêu cầu gia hạn deadline cho task hoặc dự án. Nhân viên cần nêu rõ
      task/dự án cần gia hạn, deadline mới đề xuất, và lý do cần gia hạn.
    seed_phrases:
      - "xin trễ deadline"
      - "gia hạn deadline"
      - "xin thêm thời gian"
      - "kéo dài deadline"
      - "xin lùi deadline"
      - "cần thêm thời gian hoàn thành"
      - "không kịp deadline"
      - "xin gia hạn nộp bài"
      - "extend deadline"
      - "muốn dời deadline"
      - "xin dời hạn nộp"
    required_slots:
      - task_name
      - current_deadline
      - proposed_deadline
      - reason
    optional_slots:
      - project_name
      - impact_description
    typical_entities:
      - date
      - duration
    examples:
      - "Xin gia hạn deadline task ABC thêm 2 ngày"
      - "Cần thêm thời gian hoàn thành báo cáo Q4"
      - "Xin lùi deadline dự án XYZ đến cuối tuần"

  - action_id: hr.salary_increase
    domain: hr
    business_description: >
      Đề xuất xin tăng lương hoặc review lương. Nhân viên cần nêu mức
      lương đề xuất hoặc tỷ lệ tăng mong muốn, kèm theo lý do và thành
      tích để hỗ trợ đề xuất.
    seed_phrases:
      - "xin tăng lương"
      - "đề xuất tăng lương"
      - "review lương"
      - "xin xét lương"
      - "muốn tăng lương"
      - "đề nghị điều chỉnh lương"
      - "xin nâng lương"
      - "yêu cầu tăng lương"
      - "xin xem xét mức lương"
      - "đề xuất mức lương mới"
    required_slots:
      - proposed_amount
      - reason
    optional_slots:
      - effective_date
      - achievements
      - market_comparison
    typical_entities:
      - money
      - percentage
      - date
    examples:
      - "Tôi muốn xin tăng lương 15%"
      - "Đề xuất review lương sau 1 năm làm việc"
      - "Xin điều chỉnh lương lên 20 triệu"

  - action_id: request.allowance
    domain: request
    business_description: >
      Yêu cầu cấp trợ cấp hoặc phụ cấp cho các hoạt động công tác như
      đi công tác, làm việc ngoài giờ, phụ cấp đi lại, ăn trưa, v.v.
    seed_phrases:
      - "xin trợ cấp"
      - "đề xuất phụ cấp"
      - "xin trợ cấp công tác"
      - "yêu cầu phụ cấp đi lại"
      - "xin phụ cấp ăn trưa"
      - "đề nghị trợ cấp"
      - "xin cấp phụ cấp"
      - "yêu cầu trợ cấp"
      - "xin hỗ trợ chi phí"
      - "đề xuất hỗ trợ công tác"
    required_slots:
      - allowance_type
      - amount
      - reason
    optional_slots:
      - period
      - supporting_documents
    typical_entities:
      - money
      - date
    examples:
      - "Xin trợ cấp công tác 2 triệu cho chuyến đi Đà Nẵng"
      - "Đề xuất phụ cấp đi lại 500k/tháng"
      - "Xin hỗ trợ chi phí ăn trưa"

  - action_id: request.overtime
    domain: request
    business_description: >
      Đăng ký làm thêm giờ (overtime). Nhân viên cần nêu ngày làm thêm,
      số giờ dự kiến, và lý do cần làm thêm.
    seed_phrases:
      - "đăng ký làm thêm giờ"
      - "xin làm overtime"
      - "đăng ký OT"
      - "xin làm thêm"
      - "đăng ký tăng ca"
      - "xin tăng ca"
      - "làm thêm giờ"
      - "đăng ký làm ngoài giờ"
      - "xin làm cuối tuần"
      - "đăng ký làm thứ 7"
    required_slots:
      - date
      - start_time
      - end_time
      - reason
    optional_slots:
      - project_name
      - meal_allowance
    typical_entities:
      - date
      - time
      - duration
    examples:
      - "Đăng ký OT thứ 7 này từ 8h đến 17h"
      - "Xin làm thêm tối nay 3 tiếng"
      - "Đăng ký tăng ca để hoàn thành dự án"
```

### 8.3 File config/keyword_rules.yaml (Thêm rules)

```yaml
# === THÊM VÀO CUỐI FILE keyword_rules.yaml ===

  request.late_arrival:
    strong_keywords:
      - "đi muộn"
      - "đến trễ"
      - "đi làm trễ"
      - "xin phép trễ"
      - "báo trễ"
    weak_keywords:
      - "muộn"
      - "trễ"
      - "sáng nay"
      - "kẹt xe"
      - "đến công ty"
      - "giờ làm"
    negative_keywords:
      - "deadline"
      - "task"
      - "nộp bài"
      - "dự án"
      - "hoàn thành"
      - "lương"
      - "nghỉ phép"
      - "overtime"
      - "tăng ca"
    special_patterns:
      - "xin đi muộn *"
      - "* đến trễ *"
      - "đi muộn vì *"
      - "* giờ * mới đến"
      - "sáng nay * muộn"

  request.late_deadline:
    strong_keywords:
      - "trễ deadline"
      - "gia hạn deadline"
      - "lùi deadline"
      - "kéo dài deadline"
      - "extend deadline"
    weak_keywords:
      - "deadline"
      - "thêm thời gian"
      - "hoàn thành"
      - "nộp bài"
      - "task"
      - "dự án"
    negative_keywords:
      - "đi muộn"
      - "đến trễ"
      - "công ty"
      - "kẹt xe"
      - "lương"
      - "nghỉ phép"
    special_patterns:
      - "xin trễ deadline *"
      - "gia hạn * deadline"
      - "thêm * ngày *"
      - "không kịp * deadline"

  hr.salary_increase:
    strong_keywords:
      - "tăng lương"
      - "review lương"
      - "xét lương"
      - "nâng lương"
      - "điều chỉnh lương"
    weak_keywords:
      - "lương"
      - "thu nhập"
      - "đề xuất"
      - "mức lương"
    negative_keywords:
      - "nghỉ phép"
      - "deadline"
      - "đi muộn"
      - "trợ cấp"
      - "phụ cấp"
      - "overtime"
    special_patterns:
      - "xin tăng lương *"
      - "tăng lương * %"
      - "review lương *"
      - "điều chỉnh lương *"

  request.allowance:
    strong_keywords:
      - "trợ cấp"
      - "phụ cấp"
      - "hỗ trợ chi phí"
    weak_keywords:
      - "công tác"
      - "đi lại"
      - "ăn trưa"
      - "chi phí"
      - "hỗ trợ"
    negative_keywords:
      - "lương"
      - "tăng lương"
      - "nghỉ phép"
      - "deadline"
      - "overtime"
    special_patterns:
      - "xin trợ cấp *"
      - "đề xuất phụ cấp *"
      - "hỗ trợ * chi phí"

  request.overtime:
    strong_keywords:
      - "làm thêm giờ"
      - "overtime"
      - "tăng ca"
      - "làm OT"
      - "đăng ký OT"
    weak_keywords:
      - "làm thêm"
      - "ngoài giờ"
      - "cuối tuần"
      - "thứ 7"
      - "chủ nhật"
    negative_keywords:
      - "nghỉ phép"
      - "đi muộn"
      - "lương"
      - "deadline"
      - "trợ cấp"
    special_patterns:
      - "đăng ký OT *"
      - "làm thêm * giờ"
      - "tăng ca * ngày"
      - "làm * cuối tuần"
```

### 8.4 File config/pairwise_rules.yaml (Thêm disambiguation)

```yaml
# === THÊM VÀO FILE pairwise_rules.yaml ===

pairwise_disambiguation:
  # Existing rules...
  - pair: ["leave.create", "leave.status"]
    signals:
      prefer_create: ["xin nghỉ", "tạo đơn", "nghỉ phép", "đăng ký"]
      prefer_status: ["trạng thái", "đã duyệt", "kiểm tra", "xem"]
  
  - pair: ["visitor.create", "visitor.status"]
    signals:
      prefer_create: ["mời khách", "đăng ký khách", "tạo lịch"]
      prefer_status: ["khách đến chưa", "check in", "trạng thái"]

  # NEW: Late arrival vs Late deadline
  - pair: ["request.late_arrival", "request.late_deadline"]
    signals:
      prefer_late_arrival:
        - "đi muộn"
        - "đến trễ"
        - "đến công ty"
        - "kẹt xe"
        - "giờ làm"
        - "sáng nay"
      prefer_late_deadline:
        - "deadline"
        - "task"
        - "nộp bài"
        - "dự án"
        - "hoàn thành"
        - "gia hạn"

  # NEW: Salary vs Allowance
  - pair: ["hr.salary_increase", "request.allowance"]
    signals:
      prefer_salary:
        - "tăng lương"
        - "review lương"
        - "mức lương"
        - "thu nhập"
      prefer_allowance:
        - "trợ cấp"
        - "phụ cấp"
        - "công tác"
        - "đi lại"
        - "chi phí"

  # NEW: Leave vs Overtime
  - pair: ["leave.create", "request.overtime"]
    signals:
      prefer_leave:
        - "nghỉ phép"
        - "xin nghỉ"
        - "nghỉ làm"
      prefer_overtime:
        - "làm thêm"
        - "tăng ca"
        - "OT"
        - "overtime"
        - "ngoài giờ"

default_bias:
  # Existing biases...
  leave.cancel: -0.05
  visitor.cancel: -0.05
  
  # NEW: Sensitive actions need higher confidence
  hr.salary_increase: -0.03  # Nhạy cảm, cần user confirm rõ
  request.allowance: -0.02
```

### 8.5 Tổng hợp các file cần tạo/sửa

```
📁 Cấu trúc files sau khi tích hợp:

config/
├── action_catalog.yaml      # ✏️ SỬA: Thêm 5 actions mới
├── keyword_rules.yaml       # ✏️ SỬA: Thêm 5 rule sets
├── pairwise_rules.yaml      # ✏️ SỬA: Thêm 4 pairwise rules
└── learning_loop.yaml       # Không đổi

app/
├── action_flow/
│   ├── engine.py                    # ✏️ SỬA: Tích hợp connector
│   ├── entity_extractor.py          # ✏️ SỬA: Thêm entity types
│   └── entity_extractor_extended.py # 🆕 TẠO MỚI
├── connectors/                      # 🆕 THƯ MỤC MỚI
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   └── hr_connector.py
└── ...

tests/
├── test_new_actions.py      # 🆕 TẠO MỚI
└── test_e2e_new_actions.py  # 🆕 TẠO MỚI

scripts/
└── benchmark_new_actions.py # 🆕 TẠO MỚI
```

---

## 9. Checklist tích hợp

### 9.1 Checklist cho mỗi Action mới

```
□ 1. ACTION CATALOG
  □ action_id đúng format: <domain>.<intent>
  □ business_description đầy đủ, rõ ràng
  □ seed_phrases >= 10 phrases
  □ required_slots đầy đủ
  □ typical_entities chính xác
  □ examples có ít nhất 3 ví dụ

□ 2. KEYWORD RULES
  □ strong_keywords: 4-6 cụm từ đặc trưng
  □ weak_keywords: 4-8 từ liên quan
  □ negative_keywords: từ của actions dễ nhầm
  □ special_patterns: 3-5 patterns

□ 3. PAIRWISE RULES (nếu cần)
  □ Xác định actions dễ conflict
  □ Định nghĩa signals phân biệt
  □ Test với các câu ambiguous

□ 4. ENTITY EXTRACTOR (nếu cần)
  □ Implement extractor cho entity type mới
  □ Unit test cho extractor
  □ Tích hợp vào EntityExtractor chính

□ 5. API CONNECTOR
  □ Implement connector class
  □ Định nghĩa endpoint mapping
  □ Implement slot transformation
  □ Đăng ký vào registry

□ 6. TESTING
  □ Unit tests cho routing
  □ Unit tests cho entity extraction
  □ E2E test cho full flow
  □ Benchmark accuracy >= 85%

□ 7. DOCUMENTATION
  □ Cập nhật API_SPEC.md
  □ Cập nhật USER_MANUAL.md
```

### 9.2 Checklist tổng thể (50 APIs)

```
□ Phase 1: Chuẩn bị (1 ngày)
  □ Phân loại 50 APIs theo domain
  □ Xác định entity types cần thêm
  □ Thiết kế connector architecture

□ Phase 2: Config (3-4 ngày)
  □ Viết action_catalog cho 50 actions
  □ Viết keyword_rules cho 50 actions
  □ Viết pairwise_rules cho conflicts

□ Phase 3: Code (3-4 ngày)
  □ Implement entity extractors mới
  □ Implement connectors cho các systems
  □ Tích hợp vào Action Flow Engine

□ Phase 4: Testing (2-3 ngày)
  □ Unit tests
  □ E2E tests
  □ Benchmark và tuning

□ Phase 5: Deployment (1 ngày)
  □ Environment variables
  □ API keys configuration
  □ Monitoring setup
```

---

## 10. Troubleshooting

### 10.1 Routing sai action

**Triệu chứng**: User nói "xin đi muộn" nhưng route đến "request.late_deadline"

**Nguyên nhân có thể**:
1. Thiếu strong_keywords cho action đúng
2. Thiếu negative_keywords để loại trừ
3. Seed phrases overlap

**Giải pháp**:
```yaml
# Thêm strong keywords
request.late_arrival:
  strong_keywords:
    - "đi muộn"      # Thêm
    - "đến trễ"      # Thêm

# Thêm negative keywords
request.late_deadline:
  negative_keywords:
    - "đi muộn"      # Thêm
    - "đến công ty"  # Thêm
```

### 10.2 Confidence score thấp

**Triệu chứng**: Top action có score < 0.5

**Nguyên nhân có thể**:
1. Seed phrases không đủ đa dạng
2. business_description không rõ ràng
3. User input quá ngắn hoặc ambiguous

**Giải pháp**:
```yaml
# Thêm seed phrases đa dạng hơn
seed_phrases:
  - "xin đi muộn"           # Formal
  - "đi muộn nha"           # Informal
  - "hôm nay tới trễ"       # Casual
  - "báo trễ giờ làm"       # Variation
```

### 10.3 Entity extraction fail

**Triệu chứng**: Slots không được auto-fill từ initial_text

**Debug**:
```python
# Test entity extraction
from app.action_flow.entity_extractor import EntityExtractor

extractor = EntityExtractor()
result = extractor.extract(
    "xin đi muộn hôm nay 10h30 vì kẹt xe",
    ["date", "expected_arrival_time", "reason"]
)
print(result)
# Expected: {"date": "2026-01-15", "expected_arrival_time": "10:30"}
```

**Giải pháp**: Kiểm tra regex patterns trong entity_extractor.py

### 10.4 API Connector timeout

**Triệu chứng**: Commit action fail với timeout error

**Giải pháp**:
```python
# Tăng timeout
class HRConnector(BaseConnector):
    def __init__(self, base_url: str, api_key: str = None):
        super().__init__(base_url, api_key, timeout=60)  # Tăng từ 30 lên 60
```

### 10.5 Hot reload không hoạt động

**Triệu chứng**: Sửa config nhưng routing không thay đổi

**Giải pháp**:
```bash
# Force reload
curl -X POST http://localhost:8000/admin/config/actions \
  -H "Content-Type: application/json" \
  -d '{"content": "$(cat config/action_catalog.yaml)"}'
```

Hoặc restart server:
```bash
# Restart
pkill -f "uvicorn app.main"
python scripts/run_server.py
```

---

## Phụ lục: Template files

### Template: action_catalog entry

```yaml
- action_id: <domain>.<intent>
  domain: <domain>
  business_description: >
    <Mô tả chi tiết 2-3 câu về action này làm gì,
    ai sử dụng, và kết quả mong đợi>
  seed_phrases:
    - "<phrase 1>"
    - "<phrase 2>"
    # ... tối thiểu 10 phrases
  required_slots:
    - <slot_1>
    - <slot_2>
  optional_slots:
    - <slot_3>
  typical_entities:
    - <entity_type>
  examples:
    - "<Ví dụ câu user có thể nói 1>"
    - "<Ví dụ câu user có thể nói 2>"
```

### Template: keyword_rules entry

```yaml
<action_id>:
  strong_keywords:
    - "<cụm từ đặc trưng 1>"
    - "<cụm từ đặc trưng 2>"
  weak_keywords:
    - "<từ liên quan 1>"
    - "<từ liên quan 2>"
  negative_keywords:
    - "<từ của action khác 1>"
    - "<từ của action khác 2>"
  special_patterns:
    - "<pattern với * wildcard>"
```

---

*Tài liệu này được tạo cho Hybrid Intent Router v2*
*Cập nhật: Tháng 1/2026*
