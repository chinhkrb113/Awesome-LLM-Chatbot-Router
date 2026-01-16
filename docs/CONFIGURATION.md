# Hướng dẫn Cấu hình (Configuration Guide)

Hệ thống được thiết kế theo hướng **Configuration-Driven**. Bạn có thể thêm hành động mới hoặc điều chỉnh logic nhận diện mà không cần sửa code Python.

## Cấu trúc thư mục config

```
config/
├── action_catalog.yaml   # Định nghĩa Hành động (Intents & Slots)
└── keyword_rules.yaml    # Định nghĩa Luật nhận diện (Keywords)
```

---

## 1. Action Catalog (`action_catalog.yaml`)

Tệp này định nghĩa danh sách các hành động mà chatbot có thể thực hiện.

### Cấu trúc mẫu

```yaml
actions:
  - action_id: "payment.create"
    domain: "payment"
    business_description: "Tạo lệnh chuyển tiền mới"
    seed_phrases:  # Dùng cho AI (Embedding Engine)
      - "tôi muốn chuyển tiền"
      - "bắn tiền cho bạn"
      - "thanh toán hóa đơn"
    required_slots: # Các thông tin bắt buộc phải hỏi
      - "amount"
      - "receiver"
    optional_slots: []
    typical_entities: ["money", "person"]
    examples: []
```

### Giải thích trường dữ liệu
- **`action_id`**: Định danh duy nhất (ví dụ: `domain.verb`).
- **`seed_phrases`**: Các câu mẫu dùng để huấn luyện mô hình ngữ nghĩa (Semantic Search). Cần đa dạng các cách diễn đạt.
- **`required_slots`**: Danh sách các trường thông tin mà Action Flow Engine sẽ lần lượt hỏi người dùng.

---

## 2. Keyword Rules (`keyword_rules.yaml`)

Tệp này định nghĩa các luật từ khóa cứng (Deterministic Rules) để tăng độ chính xác và khả năng kiểm soát.

### Cấu trúc mẫu

```yaml
rules:
  payment.create: # Phải khớp với action_id bên trên
    strong_keywords: # Trọng số cao (Rule Score = 1.0)
      - "chuyển tiền"
      - "bắn tiền"
      - "ck"
    weak_keywords: # Trọng số thấp (Rule Score = 0.5)
      - "gửi"
      - "trả"
    negative_keywords: # Nếu xuất hiện, loại trừ hành động này
      - "lịch sử"
      - "sao kê"
    special_patterns: [] # Regex patterns (nếu cần)
```

### Cách hoạt động
1.  **Strong Match**: Nếu tìm thấy từ khóa trong `strong_keywords`, điểm Rule = 1.0.
2.  **Weak Match**: Nếu tìm thấy từ khóa trong `weak_keywords`, điểm Rule = 0.5 (hoặc thấp hơn tùy cài đặt).
3.  **Negative Match**: Nếu tìm thấy từ khóa trong `negative_keywords`, điểm Rule = 0.0 (loại bỏ).

---

## Quy trình thêm một Action mới

1.  **Bước 1**: Mở `config/action_catalog.yaml`.
2.  **Bước 2**: Thêm block action mới (định nghĩa ID, câu mẫu, slots).
3.  **Bước 3**: Mở `config/keyword_rules.yaml`.
4.  **Bước 4**: Thêm block rule tương ứng với `action_id` vừa tạo.
5.  **Bước 5**: Khởi động lại server để nạp cấu hình mới.
