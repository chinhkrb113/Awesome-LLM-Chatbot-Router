# Generate Feedback Script

Script tự động sinh dữ liệu feedback giả lập tích cực dựa trên interactions hiện có.

## Mục đích

- Tạo dữ liệu feedback để test Learning Loop
- Sinh dữ liệu training cho việc cải thiện routing
- Mô phỏng hành vi người dùng thực tế

## Cài đặt

Script nằm tại `scripts/generate_feedback.py`, không cần cài đặt thêm dependencies.

## Sử dụng

### Cơ bản

```bash
# Sinh feedback và ghi vào file mặc định
python scripts/generate_feedback.py

# Preview trước khi ghi (dry-run)
python scripts/generate_feedback.py --dry-run

# Append vào file hiện có
python scripts/generate_feedback.py --append
```

### Tùy chỉnh

```bash
# Chỉ định file input/output
python scripts/generate_feedback.py \
    --interactions logs/router_interactions.jsonl \
    --output logs/router_feedback.jsonl

# Điều chỉnh tỷ lệ positive
python scripts/generate_feedback.py --positive-rate 0.95

# Sử dụng seed để reproducible
python scripts/generate_feedback.py --seed 42

# Điều chỉnh batch size
python scripts/generate_feedback.py --batch-size 50
```

## Tham số

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--interactions` | `logs/router_interactions.jsonl` | File interactions đầu vào |
| `--output` | `logs/router_feedback.jsonl` | File feedback đầu ra |
| `--positive-rate` | `0.92` | Tỷ lệ feedback tích cực (>90%) |
| `--batch-size` | `100` | Số lượng xử lý mỗi batch |
| `--dry-run` | `false` | Chỉ preview, không ghi file |
| `--append` | `false` | Append thay vì overwrite |
| `--seed` | `None` | Random seed để reproducible |

## Cấu hình chi tiết

Script sử dụng các tỷ lệ sau để sinh feedback tự nhiên:

| Cấu hình | Giá trị | Mô tả |
|----------|---------|-------|
| `top1_selection_rate` | 75% | Tỷ lệ chọn action đầu tiên |
| `preselect_confirm_rate` | 98% | Tỷ lệ confirm khi PRESELECT |
| `top3_confirm_rate` | 95% | Tỷ lệ confirm khi TOP_3 |
| `clarify_confirm_rate` | 92% | Tỷ lệ confirm khi CLARIFY |

## Output

### Selection Feedback

```json
{
  "event_type": "selection",
  "request_id": "uuid-from-interaction",
  "user_id": "user_id",
  "session_id": "session_id",
  "selected_action": "leave.create",
  "selection_index": 0,
  "selection_source": "click",
  "ui_strategy": "TOP_3",
  "timestamp": "2026-01-14T10:05:03.500000",
  "_simulated": true
}
```

### Outcome Feedback

```json
{
  "event_type": "outcome",
  "request_id": "uuid-from-interaction",
  "user_id": "user_id",
  "session_id": "session_id",
  "action_id": "leave.create",
  "status": "confirmed",
  "timestamp": "2026-01-14T10:05:15.200000",
  "_simulated": true
}
```

## Thống kê

Script in ra thống kê sau khi chạy:

```
==================================================
Feedback Generation Statistics
==================================================
Total interactions processed: 168
Skipped (no request_id): 44
Skipped (no actions): 0
Generated selections: 124
Generated outcomes: 124
  - Positive (confirmed): 113
  - Negative (canceled): 11
  - Positive rate: 91.1%
==================================================
```

## Kiểm tra kết quả

Sau khi sinh feedback, chạy learning review để xác nhận:

```bash
python scripts/learning_review.py
cat docs/learning/weekly_report.md
```

## Lưu ý

1. **Không ảnh hưởng dữ liệu gốc**: Script chỉ đọc interactions, không sửa đổi
2. **Đánh dấu simulated**: Tất cả feedback có `_simulated: true` để phân biệt
3. **Timestamp tự nhiên**: Timestamp được sinh với delay ngẫu nhiên
4. **Validation**: Script tự động validate dữ liệu trước khi ghi

## Testing

```bash
# Chạy unit tests
python -m pytest tests/test_generate_feedback.py -v

# Test performance với dataset lớn
python -m pytest tests/test_generate_feedback.py::TestPerformance -v
```
