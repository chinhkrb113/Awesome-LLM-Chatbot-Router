# Báo cáo cải tiến Hybrid Intent Router

## Mục tiêu
- Tăng độ chính xác định tuyến (precision/recall) và UX quyết định.
- Nâng khả năng mở rộng (hiệu năng, vận hành, độ tin cậy) cho môi trường sản xuất.
- Thiết kế để tự động chọn cấu hình/phiên bản tối ưu và cập nhật tài liệu dễ dàng.

## Đánh giá hiện trạng (ngắn gọn)
- Độ chính xác: tiền xử lý tối giản; match rule bằng chuỗi con/regex mở; điểm cộng dồn không kìm hãm; embedding một model, gộp trung bình; fuser/thresholds tĩnh; slot filling thô.
- Scale/vận hành: reload tải lại model và mã hóa toàn bộ; không batch/GPU/cache embedding; YAML không schema/versioning; logging JSONL local, thiếu metrics; thiếu health/circuit breaker; triển khai đơn tiến trình.

## Đề xuất nâng độ chính xác
- Tiền xử lý: normalize Unicode/dấu, map lỗi gõ, tách từ, chuẩn hóa số–thời gian; tạo biến thể không dấu để match rule.
- Rule Engine: regex biên từ, trọng số/giới hạn cộng dồn, từ đồng nghĩa/stemming, xử lý xung đột domain/intent, logging reasoning.
- Bộ luật: bổ sung pattern phủ tình huống chéo domain, từ khóa loại trừ; tune bằng bộ dev đo precision/recall.
- Embedding: thử model đa ngữ/fine-tune nhẹ, vector per-seed + pooling (max/weighted), thêm feature domain/intent.
- Fuser/UI: trọng số động theo domain/intent, bonus pattern theo độ tin cậy; hiệu chỉnh ngưỡng UI từ dữ liệu thật.
- Action Flow: NER/regex cho slot, validation giá trị, confidence nguồn, hỏi lại khi thấp.

## Đề xuất nâng khả năng scale
- Lifecycle: cache model, warmup; vector store/service; hot-reload vi sai (re-embed action thay đổi).
- Request: batch encode, bật GPU, cache embedding người dùng; song song hóa scoring.
- Config: schema validation, versioning + diff update; canary luật mới.
- Logging/metrics: structured logging, rotation/async sink; metrics latency/hit rate/confidence histogram/tỉ lệ fallback.
- Resilience: health/readiness cho rule/embedding; circuit breaker (rules-only/embed-only); retry/backoff với model service.
- Triển khai: nhiều worker, pre-fork warmup, container với deps cố định, resource limits, autoscale theo QPS/latency.

## Checklist thực thi
- [x] Thu thập và tóm tắt vấn đề chính (báo cáo này).
- [x] Thiết kế pipeline tiền xử lý mới.
- [x] Thiết kế nâng cấp Rule Engine.
- [x] Đề án embedding/fine-tune + vector per-seed & pooling.
- [x] Kế hoạch hiệu chỉnh Fuser + ngưỡng UI.
- [x] Thiết kế NER/regex + validation slot.
- [x] Kiến trúc scale: model/cache/vector store, batch/GPU, log/metrics, health/circuit breaker, deployment multi-worker.
- [x] Chuẩn bị bộ test tự động (tập câu hỏi, kỳ vọng top-k, độ chính xác) để đo tiến độ.
- [x] Lưu snapshot benchmark/baseline và cấu hình tuning mặc định cho production.

## Thiết kế pipeline tiền xử lý (đã phác thảo)
- Normalize: `unicodedata.normalize("NFKC")`, lower-case, chuẩn hóa dấu; map lỗi gõ phổ biến; chuẩn hóa khoảng trắng.
- Thời gian/số: regex đổi `10h30`→`10:30`, `10h`→`10:00`; giữ định dạng `dd/mm` hoặc `dd-mm`.
- Không dấu: tạo bản không dấu để dùng trong rule/regex (song song bản có dấu).
- Biên từ: dùng regex biên từ để tránh match chuỗi con.
- Mở rộng: hook để cắm tokenizer (pyvi/underthesea) khi cần mà không đổi interface.

## Thiết kế nâng cấp Rule Engine (đã phác thảo)
- Matching: biên từ regex cho strong/weak; so khớp cả có dấu/không dấu.
- Trọng số & trần điểm: strong ~0.30, weak ~0.12, negative -0.35, pattern ~0.45; giới hạn số hit mỗi nhóm; clamp [0,1].
- Pattern: wildcard `*` dịch sang regex, chạy trên cả hai biến thể.
- Xung đột: lưu reasoning chi tiết; sẽ thêm blacklist/priority theo domain ở bước sau.

## Đề án embedding/fine-tune + pooling (đã phác thảo)
- Model: 3 baseline (keepitreal/vietnamese-sbert, all-MiniLM-l12-v2, bge-m3) + 1 fine-tune nhẹ (siamese/contrastive) trên cặp seed/description vs câu hỏi thật.
- Vector per-seed: mã hóa từng seed + mô tả; chấm điểm bằng max/weighted max thay vì average để giữ sắc thái.
- Feature bổ sung: prefix domain + intent_type vào text encode để giảm nhầm cross-domain.
- Hiệu năng: precompute action vectors; lưu snapshot; chuẩn bị tùy chọn vector store (FAISS/Weaviate) khi > vài trăm action.
- Đánh giá: tập dev gồm câu hỏi thật + synthetic (synonym/no-accent/typo); metric top-1/top-3 recall, MRR; chọn theo chất lượng/chi phí.

## Kế hoạch hiệu chỉnh Fuser + ngưỡng UI (đã phác thảo)
- Dữ liệu: 300-500 câu gán nhãn action đúng; log điểm rule/embed thô.
- Grid search: w_rule/w_embed, bonus pattern, cap; xem AUC/top-1/top-3 và calibration (ECE/Brier).
- Dynamic weighting: theo intent_type/domain (vd: leave ưu tiên rule, visitor mở ưu tiên embed).
- UI thresholds: tìm ngưỡng PRESELECT/TOP_3/CLARIFY tối ưu chi phí lỗi (auto sai vs hỏi lại); có thể khác theo domain.
- Kết quả: bảng tham số + script tính lại khi có dữ liệu mới; xuất cấu hình runtime.

## Thiết kế NER/regex + validation slot (đã phác thảo)
- Kỹ thuật: regex cho ngày/giờ/ID; underthesea/pyvi để tách từ + pattern; cân nhắc Duckling nếu cần nhiều thực thể thời gian.
- Chiến lược: trích xuất đa ứng viên với confidence; nếu < ngưỡng → hỏi lại; ưu tiên giá trị hợp lệ.
- Validation theo slot:
  - `start_date/end_date/visit_date`: parse dd/mm, dd-mm, “hôm nay/ngày mai”; kiểm tra end_date ≥ start_date.
  - `leave_type`: tập giá trị cho phép (annual/sick/unpaid…), map synonym.
  - `visitor_id/leave_request_id`: regex định danh (alnum, dài 6-12), từ chối nếu sai.
  - `visit_time`: chuẩn hóa hh:mm; kiểm tra 00:00-23:59.
  - `location/host_name/visitor_name`: text tự do nhưng trim/clean, giới hạn độ dài.
- Nguồn & confidence: gắn nhãn nguồn (user_input/regex/ner/history) + confidence để quyết định hỏi lại.

## Kế hoạch bộ test định tuyến/top-k (đã phác thảo)
- Bộ test tĩnh (yaml/json): mỗi mẫu gồm `text`, `expected_action`, `accept_topk` (1/3), tag domain/intent.
- Phủ: có dấu/không dấu, typo nhẹ, synonym, cross-domain gây nhiễu, negation.
- Chỉ số: top-1 accuracy, top-3 recall, MRR; theo domain/intent để phát hiện lệch.
- CI: script chạy route trên tập test, ghi metric; dùng so sánh trước/sau khi chỉnh model, rule, fuser.
- Đã khởi tạo: `tests/router/fixtures/devset.yaml` (mẫu nhỏ) và runner `tests/router/run_eval.py` (CLI, xuất JSON/STDOUT).
- Cách chạy nhanh: `python tests/router/run_eval.py --devset tests/router/fixtures/devset.yaml --output docs/bench/latest.json` (stdout UTF-8 đã cấu hình trong runner).
- Snapshot benchmark: `docs/bench/latest.json` (lưu lần chạy gần nhất) và `docs/bench/baseline.json` (mốc so sánh hiện tại); README ở `docs/bench/README.md`.
- Devset đã mở rộng 11 mẫu (leave/visitor), kết quả hiện tại: top1≈0.82, top3=1.0, MRR≈0.91; leave vẫn có dư địa cải thiện.

## Kiến trúc scale (đã phác thảo chi tiết)
- Serving: nhiều worker gunicorn/uvicorn, pre-fork warmup model; tùy chọn chạy embedding as-a-service (HTTP/gRPC) khi cần tách tài nguyên GPU.
- Vector store: in-memory khi <500 action; chuyển FAISS (CPU) hoặc Weaviate/Milvus (GPU) khi >500-1k action; đồng bộ snapshot khi cập nhật action.
- Cache: model cache (lifetime toàn process), user embedding cache LRU (TTL ngắn), caching batch encode trong request pool để giảm trùng tính.
- Hot reload: watcher diff trên `action_catalog.yaml`/`keyword_rules.yaml`; khi đổi rule → reload nhẹ; khi đổi action → re-embed incremental (chỉ action thay đổi); không reload model.
- Observability: structured logging (JSON) + rotation; metrics Prometheus (latency p50/p95, QPS, hit rate top-1/top-3, ECE, tỉ lệ fallback, error rate); tracing nhẹ cho /route.
- Resilience: health/readiness per component (rule load OK, embedding OK); circuit breaker để rơi về chế độ rules-only nếu embedding down (và ngược lại); timeout + retry/backoff khi gọi model service.
- Deployment: container với deps pinned; resource limit CPU/RAM/GPU; autoscale dựa QPS + latency + GPU util; blue/green hoặc canary khi đổi model/weights.

## Kế hoạch bộ test tự động (đã phác thảo chi tiết)
- Cấu trúc: `tests/router/fixtures/devset.yaml` chứa danh sách mẫu; schema: `{text, expected_action, accept_topk, domain, intent_type, notes}`.
- Runner: script CLI `tests/router/run_eval.py` đọc devset, gọi API hoặc hàm Router trực tiếp, tính metric (top-1, top-3, MRR, ECE), xuất `docs/bench/latest.md` và `docs/bench/latest.json`.
- Phủ dữ liệu: mỗi action ≥10 mẫu; có/no accent; typo; synonym; negation; cross-domain nhiễu; boundary (ngưỡng PRESELECT/TOP_3).
- CI: job chạy runner, so sánh với baseline (từ `docs/bench/baseline.json`); fail nếu tụt > ngưỡng (vd -2% top-1 hoặc +0.02 ECE); artefact hóa kết quả để lưu vết.

## Thiết kế chọn phương án tối ưu (tự động)
- Benchmark runner: script chạy qua các tổ hợp (model × pooling × fuser weight × UI threshold) trên tập dev, tính metric (top-1/top-3/MRR, ECE).
- Scoring đa mục tiêu: hàm điểm = chất lượng (recall/top-k) – chi phí (latency, RAM, VRAM); cho phép trọng số theo mục tiêu kinh doanh.
- Xuất cấu hình: tự động sinh file `config/route_tuning.yaml` (chứa model, pooling, w_rule/w_embed, bonus, UI thresholds) từ run tốt nhất.
- Lưu vết: ghi lại metadata (commit hash, thời gian, tập dev, metric) để truy xuất và rollback nhanh.

## Thiết kế để dễ scale và cập nhật tài liệu
- Cấu hình tách lớp: model/embedding, rule, fuser/ui lưu ở YAML có schema; hỗ trợ override theo môi trường (dev/prod) và diff-based reload.
- Snapshot embedding: lưu vector action ra file/snapshot, tránh tính lại khi chỉ đổi rule; hỗ trợ tải incremental.
- Tự động sinh tài liệu: script đọc config + kết quả benchmark để render Markdown (bảng tham số, metric) → giảm công sức cập nhật thủ công.
- Thư mục tài liệu: `docs/` lưu báo cáo, `docs/bench/` lưu kết quả benchmark, `docs/tests/` lưu mô tả bộ test; tên file kèm ngày để version rõ ràng.
- Cấu hình tuning mặc định: `config/route_tuning.yaml` (model/pooling/fuser/ui ngưỡng baseline 2026-01-14); cập nhật khi có kết quả benchmark mới.

## Sẵn sàng production (hiện trạng)
- Artefact hiện có: devset + runner; benchmark snapshot + baseline; cấu hình tuning baseline; thiết kế scale/observability/resilience đã chốt.
- Cần làm tiếp (ưu tiên cao trước khi go-live): mở rộng devset thực tế theo domain, tích hợp runner vào CI với ngưỡng fail, hiện thực vector store/batch/GPU nếu QPS cao, thêm validation slot/NER vào Action Flow, triển khai health/circuit breaker và structured logging theo thiết kế.

## Bước tiếp theo đề xuất
- Hoàn thiện kiến trúc scale (checklist còn mở) và kế hoạch bộ test tự động.
- Triển khai benchmark runner và cấu hình xuất tự động `route_tuning.yaml`.
- Xây tập dev/test và bắt đầu đo trước/sau cho rule/embedding/fuser. 
