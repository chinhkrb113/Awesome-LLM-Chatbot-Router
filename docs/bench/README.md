# Benchmark artifacts

- `latest.json`: kết quả chạy `tests/router/run_eval.py` gần nhất trên devset (hiện tại mở rộng 11 mẫu).
- `baseline.json`: mốc so sánh hiện tại cho CI/regression. Cập nhật khi chốt cấu hình mới.
- Chạy lại: `python tests/router/run_eval.py --devset tests/router/fixtures/devset.yaml --output docs/bench/latest.json`
- So sánh với baseline: diff hai file JSON hoặc thêm bước CI để fail khi top1/top3/MRR giảm quá ngưỡng.
