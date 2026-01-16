# API Specification

Tài liệu này mô tả chi tiết các API endpoints của hệ thống. Hệ thống sử dụng RESTful API với định dạng JSON.

## Base URL
Mặc định: `http://localhost:8000`

---

## 1. Route Intent (Định tuyến)

Xác định ý định của người dùng và trả về hành động phù hợp.

- **URL**: `/route`
- **Method**: `POST`
- **Content-Type**: `application/json`

### Request Body

```json
{
  "text": "tôi muốn kiểm tra số dư",
  "user_id": "user_123"
}
```

| Field | Type | Description | Required |
| :--- | :--- | :--- | :--- |
| `text` | string | Câu nói của người dùng | Yes |
| `user_id` | string | ID người dùng (dùng cho logging/tracking) | No |

### Response Body

```json
{
  "top_actions": [
    {
      "action_id": "account.balance",
      "rule_score": 1.0,
      "embed_score": 0.85,
      "final_score": 1.0,
      "reasoning": [
        "strong_keyword_match",
        "semantic: 0.85"
      ]
    }
  ],
  "ui_strategy": "PRESELECT",
  "message": "Tôi hiểu bạn muốn kiểm tra số dư. Hãy xác nhận bên dưới."
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `top_actions` | list | Danh sách các hành động tiềm năng, đã sắp xếp. |
| `ui_strategy` | string | `PRESELECT` (chọn sẵn), `TOP_3` (gợi ý), `CLARIFY` (hỏi lại). |
| `message` | string | Câu trả lời phản hồi cho người dùng. |

---

## 2. Start Action (Bắt đầu Hành động)

Khởi tạo một phiên làm việc (session) cho một hành động cụ thể.

- **URL**: `/action/start`
- **Method**: `POST`

### Request Body

```json
{
  "session_id": "session_abc123",
  "action_id": "account.balance"
}
```

### Response Body (`ActionState`)

```json
{
  "session_id": "session_abc123",
  "action_id": "account.balance",
  "status": "DRAFT",
  "slots": {},
  "history": []
}
```

---

## 3. Interact Action (Tương tác)

Gửi thông tin đầu vào cho một phiên đang hoạt động (ví dụ: điền thông tin, xác nhận).

- **URL**: `/action/interact`
- **Method**: `POST`

### Request Body

```json
{
  "session_id": "session_abc123",
  "text": "xác nhận"
}
```

### Response Body (`ActionState`)

```json
{
  "session_id": "session_abc123",
  "action_id": "account.balance",
  "status": "CONFIRMED",
  "slots": {},
  "history": []
}
```

### Các trạng thái (`status`) có thể có:
- `INIT`: Vừa khởi tạo.
- `COLLECTING`: Đang chờ người dùng nhập thông tin (slot).
- `DRAFT`: Đã đủ thông tin, chờ xác nhận.
- `CONFIRMED`: Người dùng đã đồng ý thực hiện.
- `COMMITTED`: Hệ thống đã thực hiện xong.
- `CANCELED`: Người dùng đã hủy bỏ.

---

## 4. Swagger UI

Hệ thống cung cấp tài liệu API tương tác tự động (OpenAPI/Swagger) tại:
- URL: `http://localhost:8000/docs`
