# Hướng dẫn Cài đặt và Sử dụng

Tài liệu này hướng dẫn cách cài đặt môi trường và vận hành hệ thống Hybrid Intent Router.

## Yêu cầu hệ thống
- Python 3.8 trở lên.
- Hệ điều hành: Windows, macOS, hoặc Linux.

---

## 1. Cài đặt

### Bước 1: Clone hoặc Download source code
Đảm bảo bạn đang ở thư mục gốc của dự án (`d:\Muoi\chatbot`).

### Bước 2: Cài đặt thư viện phụ thuộc
Sử dụng `pip` để cài đặt các thư viện từ `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Lưu ý**: Hệ thống sử dụng `sentence-transformers` cho tính năng AI (Embedding). Lần đầu chạy có thể mất vài phút để tải model về máy.

---

## 2. Vận hành

### Chế độ 1: Chạy Demo trên Console (Khuyên dùng để test nhanh)
Chạy kịch bản test mẫu được định nghĩa sẵn trong file `run_demo.py`.

```bash
python run_demo.py
```

Kết quả sẽ hiển thị log chi tiết quá trình Router chấm điểm và quyết định.

### Chế độ 2: Chạy API Server
Khởi chạy server FastAPI để phục vụ các ứng dụng khác (Frontend, Mobile App).

```bash
uvicorn app.main:app --reload
```

- Server sẽ chạy tại: `http://localhost:8000`
- API Docs (Swagger): `http://localhost:8000/docs`

---

## 3. Kiểm thử (Testing)

Để chạy kịch bản kiểm thử API tự động:

```bash
python test_api.py
```

Script này sẽ:
1.  Bật server ngầm (background).
2.  Gửi các request giả lập (`/route`, `/action/start`, `/action/interact`).
3.  Kiểm tra kết quả trả về có đúng mong đợi không.
4.  Tắt server.
