# BÁO CÁO PHÂN TÍCH TOÀN DIỆN HỆ THỐNG LEARNING

**Ngày phân tích:** 16/01/2026  
**Phiên bản:** Production Final

---

## 1. TÓM TẮT ĐÁNH GIÁ

### Kết luận chính:
**Learning Loop đã được TRIỂN KHAI THỰC TẾ** nhưng **CHƯA CHẠY TỰ ĐỘNG** trong production.

| Tiêu chí | Trạng thái | Chi tiết |
|----------|------------|----------|
| Code triển khai | ✅ Hoàn thiện | Đầy đủ các module |
| Dữ liệu đầu vào | ✅ Có dữ liệu | 168 interactions, 144 selections |
| Learning đã chạy | ❌ Chưa chạy | Không có `learning_state.json` |
| Auto-tune đã chạy | ❌ Chưa chạy | Không có thư mục `backups` |
| Tích hợp vào API | ⚠️ Một phần | Cần bật `LEARNING_LOOP_AUTO=1` |

---

## 2. PHÂN TÍCH CHI TIẾT

### 2.1 Nguồn Dữ Liệu Đầu Vào

#### A. Router Interactions (`logs/router_interactions.jsonl`)
- **Số lượng:** 168 records
- **Cấu trúc dữ liệu:**
```json
{
  "timestamp": "2026-01-14T15:43:45.624710",
  "user_text": "xin nghỉ phép 2 ngày tuần sau",
  "user_id": "dev_eval",
  "top_actions": [
    {
      "action_id": "leave.create",
      "final_score": 0.97,
      "rule_score": 1.0,
      "embed_score": 0.68,
      "reasoning": ["pattern: xin nghỉ * ngày", "strong: xin nghỉ"]
    }
  ],
  "ui_strategy": "PRESELECT",
  "message": "Mình hiểu bạn đang muốn thực hiện thao tác sau:"
}
```
- **Đánh giá:** ✅ Đầy đủ thông tin cần thiết cho learning

#### B. Router Feedback (`logs/router_feedback.jsonl`)
- **Số lượng:** 281 events
  - Selection events: 144
  - Outcome events: 137
- **Cấu trúc dữ liệu:**
```json
// Selection event
{
  "event_type": "selection",
  "request_id": "uuid",
  "selected_action": "leave.create",
  "selection_index": 0,
  "selection_source": "click|preselect",
  "ui_strategy": "PRESELECT|TOP_3|CLARIFY"
}

// Outcome event
{
  "event_type": "outcome",
  "request_id": "uuid",
  "action_id": "leave.create",
  "status": "confirmed|canceled"
}
```
- **Đánh giá:** ✅ Đầy đủ feedback từ user

### 2.2 Quá Trình Xử Lý và Training

#### A. Learning Review (`scripts/learning_review.py`)
- **Chức năng:** Tạo báo cáo weekly từ logs
- **Output:** `docs/learning/weekly_report.md`
- **Metrics được tính:**
  - `top1_click_rate`: 0.766 (76.6%)
  - `clarify_rate`: 0.744 (74.4%)
  - `cancel_after_click_rate`: 0.107 (10.7%)
- **Đánh giá:** ✅ Hoạt động đúng, đã có report

#### B. Auto Tuner (`scripts/auto_tuner.py`)
- **Chức năng:** Tự động cập nhật seed phrases vào `action_catalog.yaml`
- **Logic:**
  1. Backup catalog hiện tại
  2. Tìm candidates từ feedback (score < 0.8)
  3. Thêm seed phrases mới vào catalog
- **Đánh giá:** ✅ Code hoàn thiện, ❌ Chưa được chạy

#### C. Learning Auto (`app/utils/learning_auto.py`)
- **Chức năng:** Điều phối learning loop tự động
- **Điều kiện chạy:**
  - `enabled: true` ✅
  - Trong run_window (22:00 - 06:00) ❌ Hiện tại ngoài window
  - Đủ min_selections (20) ✅ Có 144
  - Đủ min_outcomes (10) ✅ Có 137
  - Đủ min_new_selections (10) ✅
  - Cooldown 24h từ lần chạy trước ✅ (chưa chạy lần nào)
- **Đánh giá:** ⚠️ Bị block bởi run_window

### 2.3 Tích Hợp Vào Hệ Thống Chính

#### A. API Endpoints
```python
# app/main.py
POST /feedback/route   # ✅ Hoạt động
POST /feedback/outcome # ✅ Hoạt động
```

#### B. Background Thread
```python
# Cần bật bằng environment variable
set LEARNING_LOOP_AUTO=1
uvicorn app.main:app --reload
```
- **Trạng thái:** ❌ Không được bật mặc định

#### C. Config Watcher
- Router có `AtomicConfigWatcher` để reload khi config thay đổi
- Nếu AutoTuner cập nhật `action_catalog.yaml`, router sẽ tự reload
- **Đánh giá:** ✅ Cơ chế đúng

### 2.4 Kiểm Tra Log và Metrics

#### A. Learning State (`logs/learning_state.json`)
- **Trạng thái:** ❌ KHÔNG TỒN TẠI
- **Ý nghĩa:** Learning loop chưa bao giờ chạy thành công

#### B. Backup Directory (`config/backups/`)
- **Trạng thái:** ❌ KHÔNG TỒN TẠI
- **Ý nghĩa:** AutoTuner chưa bao giờ chạy

#### C. Weekly Report (`docs/learning/weekly_report.md`)
- **Trạng thái:** ✅ TỒN TẠI
- **Nội dung:** Có metrics và seed phrase candidates
- **Ý nghĩa:** `learning_review.py` đã được chạy thủ công

---

## 3. ĐÁNH GIÁ HIỆU QUẢ LEARNING

### 3.1 Metrics Hiện Tại (Từ Weekly Report)

| Metric | Giá trị | Đánh giá |
|--------|---------|----------|
| top1_click_rate | 76.6% | Tốt |
| clarify_rate | 74.4% | Cao (cần cải thiện) |
| cancel_after_click_rate | 10.7% | Chấp nhận được |

### 3.2 Phân Tích Theo Action

| Action | Selected | Top1 Correct Rate | Đánh giá |
|--------|----------|-------------------|----------|
| leave.create | 25 | 96.0% | Rất tốt |
| visitor.create | 34 | 97.1% | Rất tốt |
| visitor.cancel | 21 | 81.0% | Tốt |
| leave.status | 17 | 70.6% | Cần cải thiện |
| visitor.status | 18 | 50.0% | Cần cải thiện |
| leave.cancel | 8 | 0.0% | Cần cải thiện nhiều |
| leave.balance | 1 | 0.0% | Thiếu dữ liệu |

### 3.3 Seed Phrase Candidates (Chưa Được Áp Dụng)

Các phrases sau được đề xuất nhưng **CHƯA được thêm vào catalog**:

```yaml
leave.status:
  - "tôi muốn xin nghỉ ngày mai"  # Nhầm với leave.create
  - "tôi muốn xem bảng lương"     # Không liên quan

visitor.status:
  - "cập nhật thông tin cá nhân"  # Không liên quan
  - "tôi muốn xem bảng lương"     # Không liên quan

leave.cancel:
  - "đặt phòng họp lớn"           # Không liên quan
  - "quy trình báo cáo sự cố"     # Không liên quan
```

**Nhận xét:** Nhiều candidates không phù hợp, cần human review trước khi áp dụng.

---

## 4. CƠ CHẾ LƯU TRỮ DỮ LIỆU HỌC

### 4.1 Định Dạng và Cấu Trúc

| File | Format | Mục đích |
|------|--------|----------|
| `logs/router_interactions.jsonl` | JSONL | Log mọi routing request |
| `logs/router_feedback.jsonl` | JSONL | Log user selections & outcomes |
| `logs/learning_state.json` | JSON | Trạng thái learning loop |
| `docs/learning/weekly_report.md` | Markdown | Báo cáo metrics |
| `config/backups/*.yaml` | YAML | Backup catalog trước khi tune |

### 4.2 Vị Trí Lưu Trữ

```
project/
├── logs/
│   ├── router_interactions.jsonl  # ✅ Có dữ liệu
│   ├── router_feedback.jsonl      # ✅ Có dữ liệu
│   └── learning_state.json        # ❌ Chưa tạo
├── config/
│   ├── action_catalog.yaml        # ✅ Source of truth
│   └── backups/                   # ❌ Chưa tạo
└── docs/learning/
    └── weekly_report.md           # ✅ Có report
```

---

## 5. TÁC ĐỘNG CỦA LEARNING

### 5.1 Thành Phần Bị Thay Đổi (Khi Learning Chạy)

| Thành phần | Cách thay đổi | Trạng thái |
|------------|---------------|------------|
| `action_catalog.yaml` | Thêm seed phrases | ❌ Chưa thay đổi |
| Router embedding vectors | Rebuild khi catalog thay đổi | ❌ Chưa trigger |
| `learning_state.json` | Cập nhật timestamp & counts | ❌ Chưa tạo |

### 5.2 Thông Số/Model Được Cập Nhật

```yaml
# Khi AutoTuner chạy, sẽ cập nhật:
action_catalog:
  - action_id: leave.status
    seed_phrases:
      - "trạng thái đơn nghỉ"
      - "đơn nghỉ của tôi sao rồi"
      # + NEW phrases từ feedback
```

### 5.3 Hiệu Suất Hệ Thống Sau Khi Học

**Chưa có dữ liệu so sánh** vì learning chưa chạy.

Dự kiến cải thiện:
- `clarify_rate` giảm (ít phải hỏi lại)
- `top1_click_rate` tăng (đoán đúng hơn)
- Actions có top1_correct_rate thấp sẽ cải thiện

---

## 6. KẾT LUẬN VÀ ĐỀ XUẤT

### 6.1 Tình Trạng Hiện Tại

| Khía cạnh | Trạng thái |
|-----------|------------|
| Infrastructure | ✅ Hoàn thiện |
| Data Collection | ✅ Hoạt động |
| Learning Execution | ❌ Chưa chạy |
| Model Update | ❌ Chưa xảy ra |

### 6.2 Lý Do Learning Chưa Chạy

1. **Run Window:** Cấu hình chỉ chạy 22:00-06:00
2. **Không bật auto:** `LEARNING_LOOP_AUTO` không được set
3. **Chưa chạy thủ công:** `python scripts/auto_learning.py` chưa được gọi

### 6.3 Đề Xuất Hành Động

#### Ngay lập tức (P0):
```bash
# Chạy learning loop thủ công (bỏ qua run_window)
python scripts/learning_review.py --output docs/learning/weekly_report.md

# Hoặc force run auto_learning
python -c "
from app.utils.learning_auto import maybe_run_learning_loop, _write_state
import datetime
# Bypass run_window check
result = maybe_run_learning_loop()
print(result)
"
```

#### Cấu hình production (P1):
```yaml
# config/learning_loop.yaml
learning_loop:
  enabled: true
  run_window_start: 0   # Chạy mọi lúc
  run_window_end: 24
  auto_tune: true       # Tự động cập nhật catalog
```

#### Bật background thread (P2):
```bash
# Windows
set LEARNING_LOOP_AUTO=1
uvicorn app.main:app --reload

# Linux/Mac
LEARNING_LOOP_AUTO=1 uvicorn app.main:app --reload
```

---

## 7. APPENDIX: KIỂM TRA THỰC TẾ

### Test 1: Kiểm tra file tồn tại
```
learning_state.json exists: False
backups dir exists: False
```

### Test 2: Đếm dữ liệu
```
Feedback events: 281
Interactions: 168
Selections: 144
Outcomes: 137
```

### Test 3: Kiểm tra config
```
enabled: True
auto_tune: True
min_selections: 20
```

### Test 4: Chạy learning loop
```
Result: {'status': 'skipped', 'reason': 'outside_run_window'}
```

---

*Báo cáo được tạo bởi Kiro - 16/01/2026*
