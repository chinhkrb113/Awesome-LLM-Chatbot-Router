# ĐÁNH GIÁ KIẾN TRÚC TỔNG THỂ - HYBRID INTENT ROUTER

**Ngày:** 15/01/2026  
**Phiên bản:** Production Final  
**Loại:** Architecture Review + Action Plan

---

## 1. KẾT LUẬN TỔNG THỂ

### ✅ Đánh giá chung: **PRODUCTION-READY với điều kiện**

Đây là một **Hybrid Intent Router "đúng sách"**:
- Không thần thánh hóa ML
- Kết hợp Rules + Embedding + Context + UX + Learning loop
- Tư duy **conversation-first**, không phải "classify sentence"

> **Nếu triển khai đúng, hệ thống đủ sức chạy production cho enterprise internal assistant (HR, Visitor, ERP).**

---

## 2. ĐIỂM MẠNH NỔI BẬT (GIỮ NGUYÊN)

### ✅ 2.1 Context Memory Design (P0)

**Điểm nổi bật:**
- Chuyển router từ **single-turn classifier** → **finite-state conversation router**
- Lưu trữ implicit dialog state:
  - `last_action`
  - `last_domain`
  - `last_bot_question`
  - `recent_intents`

> **Nhận xét kiến trúc:** Đang xây **lightweight dialogue manager** mà không cần full-blown Rasa/Dialogflow.

---

### ✅ 2.2 Pairwise Disambiguation (P0)

**Tại sao đúng:**
- 90% lỗi intent không nằm ở 50 intent, mà ở **5-7 cặp rất giống nhau**
- Chỉ activate khi `gap < 0.1`
- Rule chỉ áp dụng cho pair cụ thể

> **Kết quả:** Tránh overfitting, tránh rule explosion.

---

### ✅ 2.3 Smart Slot Pre-filling with Confirmation (P0)

**UX best practice của Copilot/AI Assistant hiện đại:**
- `Confidence + Source = Transparency`
- Cho user quyền confirm/edit trước khi commit

> **Impact:** Giảm "blame AI" khi sai.

---

### ✅ 2.4 Progressive Disclosure (P0)

**Clean architecture:**
- Gắn UI strategy với router output (`PRESELECT` / `TOP_3` / `CLARIFY`)
- Không hard-code UI behavior

> **Kết quả:** Clean contract giữa backend & frontend.

---

### ✅ 2.5 Realtime Learning (Rất hiếm team làm đúng)

**Tránh được 2 sai lầm lớn:**
- ❌ Không retrain model bừa bãi
- ❌ Không học từ mọi case

**Chỉ học khi:**
- User override
- Score thấp
- Có threshold

> **Đây là human-in-the-loop đúng nghĩa.**

---

## 3. RỦI RO & THIẾU SÓT CẦN BỔ SUNG

### ⚠️ 3.1 Context Memory cần TTL + Garbage Collection

**Vấn đề hiện tại:**
```python
self._contexts: Dict[str, ConversationContext]  # ❌ No TTL
```

**Rủi ro:**
- Memory leak
- Session ghost (user quay lại sau 2 ngày)

**Giải pháp bắt buộc:**

```python
@dataclass
class ConversationContext:
    session_id: str
    last_action: Optional[str] = None
    last_domain: Optional[str] = None
    last_updated_at: datetime = field(default_factory=datetime.now)  # ✅ NEW
    ttl_seconds: int = 1800  # 30 phút  # ✅ NEW
    
    def is_expired(self) -> bool:
        return (datetime.now() - self.last_updated_at).total_seconds() > self.ttl_seconds

class ContextMemoryManager:
    def get_context(self, session_id: str) -> ConversationContext:
        ctx = self._contexts.get(session_id)
        
        # Auto-expire
        if ctx and ctx.is_expired():
            del self._contexts[session_id]
            ctx = None
        
        if ctx is None:
            ctx = ConversationContext(session_id)
            self._contexts[session_id] = ctx
        
        return ctx
    
    def cleanup_expired(self):
        """Chạy periodic (mỗi 5 phút)"""
        expired = [k for k, v in self._contexts.items() if v.is_expired()]
        for k in expired:
            del self._contexts[k]
```

---

### ⚠️ 3.2 Thiếu Negative Context Reset

**Case nguy hiểm:**
```
User: xin nghỉ phép
Bot: bạn muốn nghỉ ngày nào?
User: à thôi
User: tạo khách mời  ← Router vẫn boost leave.* ❌
```

**Giải pháp:**

```python
RESET_KEYWORDS = ["thôi", "huỷ", "hủy", "cancel", "bỏ", "không", "quên đi"]

class ContextMemoryManager:
    def check_and_reset(self, session_id: str, user_text: str) -> bool:
        """Return True if context was reset"""
        text_lower = user_text.lower().strip()
        
        if any(kw in text_lower for kw in RESET_KEYWORDS):
            ctx = self._contexts.get(session_id)
            if ctx:
                ctx.last_action = None
                ctx.last_domain = None
                ctx.recent_intents.clear()
                return True
        return False
```

**Hoặc confidence-based decay:**
```python
def get_domain_boost(self, session_id: str, action_id: str, turns_since: int) -> float:
    """Decay boost theo số turn"""
    base_boost = 0.1
    decay_factor = 0.7 ** turns_since  # 0.1 → 0.07 → 0.049
    return base_boost * decay_factor
```

---

### ⚠️ 3.3 Pairwise Resolver cần Directional Rule

**Vấn đề:**
- Không phải pair nào cũng symmetric
- Có intent **default nguy hiểm hơn** (cancel)

**Giải pháp:**

```yaml
# config/pairwise_rules.yaml
pairwise_disambiguation:
  - pair: [leave.create, leave.status]
    signals:
      prefer_create: ["xin nghỉ", "tạo đơn"]
      prefer_status: ["trạng thái", "đã duyệt"]
    
  # NEW: Default bias cho dangerous intents
  default_bias:
    leave.cancel: -0.05    # Cần strong signal hơn
    visitor.cancel: -0.05  # Cần strong signal hơn
```

```python
class PairwiseResolver:
    def resolve(self, text: str, candidates: List[ActionCandidate]) -> List[ActionCandidate]:
        # Apply default bias FIRST
        for candidate in candidates:
            bias = self.default_bias.get(candidate.action_id, 0.0)
            candidate.final_score += bias
        
        # Then apply pairwise rules...
```

---

### ⚠️ 3.4 Entity Signals: Risk Overboost

**Vấn đề:**
```python
candidate.final_score += weight  # 3 entity nhỏ → override semantic score ❌
```

**Giải pháp:**

```python
class EntitySignalExtractor:
    MAX_TOTAL_BOOST = 0.15  # ✅ Clamp
    
    def apply_boosts(self, candidates: List[ActionCandidate], signals: Dict) -> None:
        for candidate in candidates:
            total_boost = 0.0
            
            for signal, present in signals.items():
                if not present:
                    continue
                config = self.entity_action_map.get(signal, {})
                if self._matches_pattern(candidate.action_id, config.get("boost", [])):
                    total_boost += config.get("weight", 0.05)
            
            # ✅ Clamp total boost
            clamped_boost = min(total_boost, self.MAX_TOTAL_BOOST)
            candidate.final_score = min(1.0, candidate.final_score + clamped_boost)
```

---

### ⚠️ 3.5 Vietnamese SBERT - Risk Vận Hành

**Rủi ro với community model (`keepitreal/vietnamese-sbert`):**
- Không được maintain
- Không có license guarantee

**Khuyến nghị thực tế:**

```yaml
# config/route_tuning.yaml
model:
  # Primary: Stable, maintained
  primary: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  
  # Secondary (A/B test): Vietnamese specific
  ab_test_variant: "keepitreal/vietnamese-sbert"
  
  # Fallback: Always available
  fallback: "BAAI/bge-small-en-v1.5"
  
  # Promotion criteria
  promotion_rules:
    min_improvement: 0.05  # 5% better accuracy
    min_samples: 1000
    stability_days: 14
```

---

## 4. ĐỀ XUẤT NÂNG CẤP KIẾN TRÚC (P1+)

### 4.1 Router Trace / Explainability (Rất nên làm)

**Lưu cho mỗi routing:**

```python
@dataclass
class RouterTrace:
    request_id: str
    user_text: str
    
    # Score breakdown
    semantic_score: float
    rule_score: float
    context_boost: float
    entity_boost: float
    pairwise_adjustment: float
    final_score: float
    
    # Decision
    selected_action: str
    ui_strategy: str
    
    # Metadata
    timestamp: datetime
    latency_ms: float

# Output example
{
    "semantic_score": 0.72,
    "rule_score": 0.81,
    "context_boost": 0.10,
    "entity_boost": 0.08,
    "pairwise_adjustment": -0.05,
    "final_score": 0.86,
    "explain": "Vì bạn vừa hỏi về nghỉ phép"
}
```

**Dùng cho:**
- Debug
- Training data
- Explain UI ("Vì bạn vừa hỏi trạng thái...")

---

### 4.2 Intent Risk Tier

**Không phải intent nào cũng như nhau:**

| Tier | Intent | Behavior |
|------|--------|----------|
| **Safe** | status, balance | Auto-execute OK |
| **Medium** | create | Single confirm |
| **Dangerous** | cancel, delete | Double confirm |

**Router output:**

```python
class RouterOutput(BaseModel):
    # ... existing fields ...
    risk_level: str = "SAFE"  # SAFE / MEDIUM / DANGEROUS
```

**Frontend handling:**

```typescript
if (result.risk_level === 'DANGEROUS') {
  // Show warning + double confirm
  showDangerConfirmDialog(result);
} else {
  // Normal flow
  proceedWithAction(result);
}
```

---

### 4.3 Domain-level Confidence Decay

**Context boost không nên binary:**

```python
def get_domain_boost(self, session_id: str, action_id: str) -> float:
    ctx = self.get_context(session_id)
    
    if not ctx.last_domain:
        return 0.0
    
    if not action_id.startswith(ctx.last_domain):
        return 0.0
    
    # Decay based on turns
    turns_since = len(ctx.recent_intents)
    
    # Turn 1: +0.10
    # Turn 2: +0.07
    # Turn 3: +0.049
    # Turn 4+: ~0
    base_boost = 0.10
    decay_rate = 0.7
    
    return base_boost * (decay_rate ** turns_since)
```

**Tránh "sticky domain" problem.**

---

## 5. ƯU TIÊN THỰC SỰ (NẾU NGUỒN LỰC HẠN CHẾ)

### Top 5 việc cần làm đầu tiên:

| # | Task | Lý do | Effort |
|---|------|-------|--------|
| 1 | **A1: Context Memory + TTL** | Bắt buộc, tránh memory leak | 2 ngày |
| 2 | **A2: Pairwise Disambiguation** | Impact cao, effort thấp | 1 ngày |
| 3 | **B1: Slot Pre-fill + Confirm** | UX critical | 2 ngày |
| 4 | **B2: Progressive Disclosure** | Frontend only | 1 ngày |
| 5 | **C1: Realtime Learning (logging only)** | Foundation cho future | 1 ngày |

### ⛔ Có thể delay:

| Task | Lý do delay |
|------|-------------|
| A4: Vietnamese Model | Cần A/B test, risk cao |
| B5: Quick Actions | Nice-to-have |
| C2: A/B Testing Framework | Cần volume trước |

---

## 6. KPI & CÁCH ĐO IMPACT THỰC TẾ

### 6.1 Core Router Metrics

| Metric | Cách đo | Target |
|--------|---------|--------|
| **Top-1 Accuracy** | `selected == top_actions[0]` | ≥ 85% |
| **Override Rate** | User chọn khác top-1 | ≤ 15% |
| **Clarify Rate** | `ui_strategy == CLARIFY` | ≤ 10% |
| **Cancel False-Positive** | Cancel được chọn nhưng user cancel flow | ≤ 2% |

```python
# Logging for metrics
def log_routing_metrics(request_id: str, result: RouterOutput, selected: str):
    metrics = {
        "request_id": request_id,
        "top1_action": result.top_actions[0].action_id if result.top_actions else None,
        "selected_action": selected,
        "is_top1_correct": selected == result.top_actions[0].action_id if result.top_actions else False,
        "ui_strategy": result.ui_strategy,
        "top1_score": result.top_actions[0].final_score if result.top_actions else 0,
        "timestamp": datetime.now().isoformat()
    }
    log_to_metrics_store(metrics)
```

---

### 6.2 UX Metrics

| Metric | Cách đo | Target |
|--------|---------|--------|
| **Avg Turns per Task** | Từ route đến commit | ≤ 3 turns |
| **Time-to-Complete** | Timestamp diff | ≤ 60 seconds |
| **Slot Correction Count** | User sửa slot sau auto-fill | ≤ 0.5/task |
| **Abandon Rate** | Start nhưng không commit | ≤ 10% |

```python
# Task completion tracking
def log_task_metrics(session_id: str, action_id: str, outcome: str):
    task = get_task_by_session(session_id)
    metrics = {
        "session_id": session_id,
        "action_id": action_id,
        "outcome": outcome,  # COMMITTED / CANCELED / ABANDONED
        "total_turns": task.turn_count,
        "duration_seconds": (datetime.now() - task.started_at).total_seconds(),
        "slot_corrections": task.correction_count,
        "auto_filled_slots": task.auto_fill_count
    }
    log_to_metrics_store(metrics)
```

---

### 6.3 Learning Quality Metrics

| Metric | Cách đo | Target |
|--------|---------|--------|
| **Phrase Reuse Rate** | New phrases được dùng lại | ≥ 30% |
| **Regression Rate** | Accuracy drop sau update | ≤ 2% |
| **Catalog Growth** | Seed phrases added/week | 5-15/action |

```python
# Learning quality tracking
def log_learning_metrics(update_batch: dict):
    metrics = {
        "batch_id": update_batch["id"],
        "phrases_added": len(update_batch["new_phrases"]),
        "actions_updated": len(update_batch["affected_actions"]),
        "accuracy_before": get_current_accuracy(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Schedule accuracy check after 24h
    schedule_accuracy_check(metrics["batch_id"], delay_hours=24)
```

---

### 6.4 Dashboard Queries (SQL/Analytics)

```sql
-- Top-1 Accuracy (Daily)
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_routes,
    SUM(CASE WHEN is_top1_correct THEN 1 ELSE 0 END) as correct,
    ROUND(SUM(CASE WHEN is_top1_correct THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as accuracy_pct
FROM router_metrics
WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(timestamp)
ORDER BY date;

-- Confusing Pairs Analysis
SELECT 
    top1_action,
    selected_action,
    COUNT(*) as override_count
FROM router_metrics
WHERE is_top1_correct = FALSE
GROUP BY top1_action, selected_action
ORDER BY override_count DESC
LIMIT 10;

-- Task Completion Funnel
SELECT 
    action_id,
    COUNT(*) as started,
    SUM(CASE WHEN outcome = 'COMMITTED' THEN 1 ELSE 0 END) as completed,
    ROUND(SUM(CASE WHEN outcome = 'COMMITTED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as completion_rate
FROM task_metrics
WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY action_id;
```

---

## 7. BỔ SUNG: LATENCY BUDGET & KILL-SWITCH

### 7.1 Router Latency Budget (Quan trọng khi scale)

**Nguyên tắc:** Router không được là bottleneck UX.

```python
# app/router/router_final.py - thêm latency guardrail

class RouterFinal:
    LATENCY_BUDGET_MS = 300  # Hard limit
    
    def route(self, request: UserRequest) -> RouterOutput:
        start_time = time.perf_counter()
        
        # ... preprocessing, rule scoring ...
        
        # Check budget before expensive operations
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        remaining_budget = self.LATENCY_BUDGET_MS - elapsed_ms
        
        # Skip optional enrichments if budget tight
        skip_pairwise = remaining_budget < 50
        skip_entity = remaining_budget < 100
        
        if not skip_entity:
            entity_boosts = self.entity_extractor.get_boosts(clean_text)
            # apply boosts...
        
        if not skip_pairwise:
            candidates = self.pairwise_resolver.resolve(clean_text, candidates)
        
        # Log degradation
        if skip_pairwise or skip_entity:
            logger.warning(f"Latency budget exceeded: skipped pairwise={skip_pairwise}, entity={skip_entity}")
            self.metrics.record_degraded_route()
        
        return result
```

```yaml
# config/route_tuning.yaml
latency:
  budget_ms: 300
  skip_pairwise_threshold_ms: 250
  skip_entity_threshold_ms: 200
  log_slow_routes_above_ms: 200
```

---

### 7.2 Kill-switch cho Learning (Ops Hygiene)

**Nguyên tắc:** Một batch phrase xấu có thể làm hỏng router trong 1 đêm.

```yaml
# config/learning_loop.yaml
learning_loop:
  # MASTER KILL-SWITCH
  enabled: false  # ⚠️ Default OFF in production
  
  # Granular controls
  auto_tune_enabled: false      # Auto-update seed phrases
  realtime_learning: false      # Learn from clicks immediately
  weekly_report_only: true      # Safe: chỉ generate report
  
  # Safety limits
  max_phrases_per_action: 5     # Không add quá nhiều 1 lần
  max_actions_per_batch: 3      # Không update quá nhiều action
  require_manual_approval: true # Human review trước khi apply
  
  # Rollback
  backup_before_update: true
  max_backups: 10
  auto_rollback_on_regression: true
  regression_threshold: 0.05   # Rollback nếu accuracy drop > 5%
```

```python
# app/utils/learning_auto.py - thêm safety checks

class SafeLearningLoop:
    def maybe_run(self) -> Dict[str, Any]:
        cfg = self.config
        
        # Kill-switch check
        if not cfg.get("enabled"):
            return {"status": "disabled", "reason": "kill_switch_off"}
        
        # Auto-tune specific check
        if not cfg.get("auto_tune_enabled"):
            # Only generate report, no updates
            return self._generate_report_only()
        
        # Manual approval check
        if cfg.get("require_manual_approval"):
            candidates = self._get_candidates()
            self._save_for_review(candidates)
            return {"status": "pending_approval", "candidates": len(candidates)}
        
        # Proceed with caution
        return self._run_with_rollback_protection()
    
    def _run_with_rollback_protection(self):
        # Backup
        backup_path = self._create_backup()
        
        # Apply updates
        result = self._apply_updates()
        
        # Check regression (after 1 hour)
        schedule_regression_check(
            backup_path=backup_path,
            threshold=self.config["regression_threshold"],
            delay_minutes=60
        )
        
        return result
```

**Ops Runbook:**
```bash
# Emergency: Disable learning immediately
sed -i 's/enabled: true/enabled: false/' config/learning_loop.yaml

# Rollback to last backup
cp config/backups/action_catalog_YYYYMMDD.yaml config/action_catalog.yaml

# Restart router to pick up changes
systemctl restart hybrid-router
```

---

## 8. SƠ ĐỒ END-TO-END (REFERENCE ARCHITECTURE)

### 8.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID INTENT ROUTER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌─────────────────────────────────────────────────────┐    │
│  │   User   │───▶│                    ROUTER PIPELINE                   │    │
│  │  Input   │    │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐│    │
│  └──────────┘    │  │Preprocess │─▶│  Context  │─▶│   Score Fusion    ││    │
│                  │  │           │  │  Memory   │  │                   ││    │
│                  │  │• lowercase│  │           │  │ ┌───────────────┐ ││    │
│                  │  │• normalize│  │• TTL check│  │ │  Rule Engine  │ ││    │
│                  │  │• time fmt │  │• domain   │  │ │  (0.6 weight) │ ││    │
│                  │  └───────────┘  │  boost    │  │ └───────────────┘ ││    │
│                  │                 │• reset    │  │         +         ││    │
│                  │                 │  check    │  │ ┌───────────────┐ ││    │
│                  │                 └───────────┘  │ │Embed Engine   │ ││    │
│                  │                                │ │  (0.4 weight) │ ││    │
│                  │                                │ └───────────────┘ ││    │
│                  │                                │         +         ││    │
│                  │                                │ ┌───────────────┐ ││    │
│                  │                                │ │Entity Signals │ ││    │
│                  │                                │ │ (≤0.15 boost) │ ││    │
│                  │                                │ └───────────────┘ ││    │
│                  │                                └───────────────────┘│    │
│                  └──────────────────────────┬──────────────────────────┘    │
│                                             │                                │
│                                             ▼                                │
│                  ┌──────────────────────────────────────────────────────┐   │
│                  │                 POST-PROCESSING                       │   │
│                  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │   │
│                  │  │  Pairwise   │─▶│ Risk Tier   │─▶│ UI Strategy  │  │   │
│                  │  │  Resolver   │  │  Assignment │  │  Decision    │  │   │
│                  │  │             │  │             │  │              │  │   │
│                  │  │• gap < 0.1  │  │• SAFE       │  │• PRESELECT   │  │   │
│                  │  │• pair rules │  │• MEDIUM     │  │• TOP_3       │  │   │
│                  │  │• bias adj   │  │• DANGEROUS  │  │• CLARIFY     │  │   │
│                  │  └─────────────┘  └─────────────┘  └──────────────┘  │   │
│                  └──────────────────────────┬───────────────────────────┘   │
│                                             │                                │
└─────────────────────────────────────────────┼────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ROUTER OUTPUT                                   │
│  {                                                                           │
│    "top_actions": [...],                                                     │
│    "ui_strategy": "TOP_3",                                                   │
│    "risk_level": "SAFE",                                                     │
│    "trace": { "rule": 0.6, "embed": 0.72, "context": 0.1, ... }             │
│  }                                                                           │
└──────────────────────────────────────────────┬──────────────────────────────┘
                                               │
                      ┌────────────────────────┼────────────────────────┐
                      │                        │                        │
                      ▼                        ▼                        ▼
              ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
              │   FRONTEND   │        │ACTION FLOW   │        │   METRICS    │
              │              │        │   ENGINE     │        │   & LOGS     │
              │• Show buttons│        │              │        │              │
              │• Progressive │        │• INIT        │        │• RouterTrace │
              │  disclosure  │        │• COLLECTING  │        │• Latency     │
              │• Risk confirm│        │• DRAFT       │        │• Accuracy    │
              └──────┬───────┘        │• CONFIRMED   │        │• Override    │
                     │                │• COMMITTED   │        └──────────────┘
                     │                └──────┬───────┘                │
                     │                       │                        │
                     ▼                       ▼                        ▼
              ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
              │ User Select  │        │  Backend     │        │  Learning    │
              │   Action     │───────▶│    API       │        │    Loop      │
              │              │        │   Commit     │        │              │
              │• Click button│        │              │        │• Weekly      │
              │• Feedback log│        │• Validate    │        │• Auto-tune   │
              └──────────────┘        │• Execute     │        │• Kill-switch │
                                      └──────────────┘        └──────────────┘
```

---

### 8.2 Sequence Diagram (Happy Path)

```
User          Frontend       Router         Context        ActionFlow      Backend
 │               │              │              │               │              │
 │ "Mai cho A    │              │              │               │              │
 │  vào cổng"    │              │              │               │              │
 │──────────────▶│              │              │               │              │
 │               │ POST /route  │              │               │              │
 │               │─────────────▶│              │               │              │
 │               │              │ get_context()│               │              │
 │               │              │─────────────▶│               │              │
 │               │              │◀─────────────│               │              │
 │               │              │ domain_boost │               │              │
 │               │              │              │               │              │
 │               │              │──┐ Rule+Embed+Entity         │              │
 │               │              │  │ Scoring                   │              │
 │               │              │◀─┘                           │              │
 │               │              │                              │              │
 │               │              │──┐ Pairwise                  │              │
 │               │              │  │ Resolve                   │              │
 │               │              │◀─┘                           │              │
 │               │              │                              │              │
 │               │ RouterOutput │                              │              │
 │               │ (TOP_3,SAFE) │                              │              │
 │               │◀─────────────│                              │              │
 │               │              │                              │              │
 │  Show 3       │              │                              │              │
 │  buttons      │              │                              │              │
 │◀──────────────│              │                              │              │
 │               │              │                              │              │
 │ Click         │              │                              │              │
 │ "Tạo khách"   │              │                              │              │
 │──────────────▶│              │                              │              │
 │               │ POST /action/start                          │              │
 │               │─────────────────────────────────────────────▶              │
 │               │                              │               │              │
 │               │                              │ Extract slots │              │
 │               │                              │ from text     │              │
 │               │                              │               │              │
 │               │ ActionState (COLLECTING)    │               │              │
 │               │◀─────────────────────────────────────────────│              │
 │               │              │              │               │              │
 │  "Đã nhận:    │              │              │               │              │
 │   visitor=A   │              │              │               │              │
 │   date=mai"   │              │              │               │              │
 │◀──────────────│              │              │               │              │
 │               │              │              │               │              │
 │ "Đúng rồi"    │              │              │               │              │
 │──────────────▶│              │              │               │              │
 │               │ POST /action/interact       │               │              │
 │               │─────────────────────────────────────────────▶│              │
 │               │              │              │               │──┐           │
 │               │              │              │               │  │ Validate  │
 │               │              │              │               │◀─┘           │
 │               │              │              │               │              │
 │               │ ActionState (DRAFT)         │               │              │
 │               │◀─────────────────────────────────────────────│              │
 │               │              │              │               │              │
 │  Show Draft   │              │              │               │              │
 │  [Confirm]    │              │              │               │              │
 │  [Edit][Huỷ]  │              │              │               │              │
 │◀──────────────│              │              │               │              │
 │               │              │              │               │              │
 │ "Xác nhận"    │              │              │               │              │
 │──────────────▶│              │              │               │              │
 │               │ POST /action/interact       │               │              │
 │               │─────────────────────────────────────────────▶│              │
 │               │              │              │               │ Commit       │
 │               │              │              │               │─────────────▶│
 │               │              │              │               │◀─────────────│
 │               │              │              │               │              │
 │               │ ActionState (COMMITTED)     │               │              │
 │               │◀─────────────────────────────────────────────│              │
 │               │              │              │               │              │
 │  "Thành công" │              │              │               │              │
 │◀──────────────│              │              │               │              │
 │               │              │              │               │              │
```

---

### 8.3 Component Dependency Map

```
                    ┌─────────────────────────────────────┐
                    │           CONFIG LAYER              │
                    │  ┌─────────────┐ ┌───────────────┐  │
                    │  │action_catalog│ │keyword_rules │  │
                    │  │    .yaml    │ │    .yaml     │  │
                    │  └──────┬──────┘ └───────┬───────┘  │
                    │         │                │          │
                    │  ┌──────┴────────────────┴───────┐  │
                    │  │       ConfigLoader            │  │
                    │  └──────────────┬────────────────┘  │
                    └─────────────────┼───────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌─────────────────┐           ┌─────────────────┐
│  RuleEngine   │           │  EmbedEngine    │           │ ActionFlowEngine│
│               │           │                 │           │                 │
│• strong_kw    │           │• embed_anything │           │• state machine  │
│• weak_kw      │           │• vector_store   │           │• slot collection│
│• negative_kw  │           │• query_cache    │           │• validation     │
│• patterns     │           │                 │           │                 │
└───────┬───────┘           └────────┬────────┘           └────────┬────────┘
        │                            │                             │
        └────────────┬───────────────┘                             │
                     │                                             │
                     ▼                                             │
           ┌─────────────────┐                                     │
           │   RouterFinal   │◀────────────────────────────────────┘
           │                 │
           │• Preprocessor   │
           │• ContextMemory  │
           │• Fuser          │
           │• PairwiseResolver
           │• UIDecision     │
           │• Metrics        │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   FastAPI App   │
           │                 │
           │• /route         │
           │• /action/*      │
           │• /feedback/*    │
           │• /admin/*       │
           └─────────────────┘
```

---

## 9. IMPLEMENTATION CHECKLIST

### Phase 1: Foundation (Week 1-2)

- [ ] **Context Memory với TTL**
  - [ ] Add `last_updated_at`, `ttl_seconds` to ConversationContext
  - [ ] Implement `is_expired()` check
  - [ ] Add periodic cleanup job
  - [ ] Add negative context reset keywords

- [ ] **Pairwise Disambiguation**
  - [ ] Create `config/pairwise_rules.yaml`
  - [ ] Implement `PairwiseResolver` class
  - [ ] Add default bias for dangerous intents
  - [ ] Integrate into router pipeline

- [ ] **Metrics Logging**
  - [ ] Add RouterTrace dataclass
  - [ ] Log all routing decisions
  - [ ] Setup basic dashboard queries

### Phase 2: UX (Week 3-4)

- [ ] **Slot Pre-fill + Confirmation**
  - [ ] Update `start_action()` to show extracted info
  - [ ] Add confirmation buttons
  - [ ] Track slot corrections

- [ ] **Progressive Disclosure**
  - [ ] Update frontend ActionButtons component
  - [ ] Add "show more" functionality
  - [ ] Add reasoning display on hover

### Phase 3: Optimization (Week 5-6)

- [ ] **Entity Signals**
  - [ ] Implement EntitySignalExtractor
  - [ ] Add boost clamping
  - [ ] Integrate into fuser

- [ ] **Router Trace / Explainability**
  - [ ] Add score breakdown to output
  - [ ] Add explain text generation
  - [ ] Update frontend to show explanations

### Phase 4: Advanced (Week 7-8)

- [ ] **Intent Risk Tier**
  - [ ] Add risk_level to RouterOutput
  - [ ] Implement double-confirm for dangerous
  - [ ] Update frontend handling

- [ ] **Domain Confidence Decay**
  - [ ] Implement decay formula
  - [ ] Add turn tracking
  - [ ] Test sticky domain scenarios

---

## 8. TÓM TẮT

### Điểm mạnh cần giữ:
1. ✅ Hybrid approach (Rules + Embedding + Context)
2. ✅ Human-in-the-loop design
3. ✅ Clean UI strategy contract
4. ✅ Selective learning (không học bừa)

### Điểm cần bổ sung ngay:
1. ⚠️ Context TTL + Garbage Collection
2. ⚠️ Negative Context Reset
3. ⚠️ Pairwise Default Bias
4. ⚠️ Entity Boost Clamping

### KPI chính cần track:
1. 📊 Top-1 Accuracy ≥ 85%
2. 📊 Override Rate ≤ 15%
3. 📊 Avg Turns ≤ 3
4. 📊 Cancel False-Positive ≤ 2%

---

---

## 10. FINAL VERDICT

### Sign-off Status: ✅ **APPROVED WITH MANDATORY FIXES**

| Category | Status | Notes |
|----------|--------|-------|
| Architecture Design | ✅ Approved | Hybrid approach đúng hướng |
| Production Readiness | ⚠️ Conditional | Cần hoàn thành M1-M4 |
| Scalability | ✅ Approved | Latency budget đã có |
| Operability | ✅ Approved | Kill-switch + rollback |
| Observability | ✅ Approved | Metrics + Trace đầy đủ |

### Mandatory Fixes Before Go-Live:

| # | Fix | Owner | Deadline |
|---|-----|-------|----------|
| M1 | Context TTL + Reset | Backend | Week 1 |
| M2 | Dangerous Intent Bias | Backend | Week 1 |
| M3 | Entity Boost Clamping | Backend | Week 1 |
| M4 | Logging Infrastructure | DevOps | Week 1 |

### Post Go-Live (P1):

| # | Enhancement | Target |
|---|-------------|--------|
| P1 | Vietnamese Model A/B | Week 3 |
| P2 | Full Learning Loop | Week 4 |
| P3 | A/B Testing Framework | Week 6 |

---

> **Architecture Review Board Decision:**  
> GO LIVE được với điều kiện bắt buộc hoàn thành M1-M4 trước khi mở rộng traffic.

---

*Tài liệu được tổng hợp từ Architecture Review - 15/01/2026*  
*Approved by: Principal Architect Review*
