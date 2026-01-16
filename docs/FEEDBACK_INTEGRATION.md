# Feedback API Integration Guide

## Overview

Hệ thống Learning Loop cần thu thập feedback từ người dùng để cải thiện routing. Document này mô tả cách frontend tích hợp với Feedback API.

## API Endpoints

### 1. POST /feedback/route - Selection Feedback

Gọi khi user chọn một action từ danh sách gợi ý.

```typescript
interface RouteFeedbackRequest {
  request_id: string;        // Required: từ response /route
  user_id?: string;          // Optional: ID người dùng
  session_id?: string;       // Optional: Session ID
  selected_action?: string;  // Action ID được chọn
  selection_index?: number;  // Vị trí trong danh sách (0-based)
  selection_source?: string; // 'click' | 'preselect'
  ui_strategy?: string;      // 'PRESELECT' | 'TOP_3' | 'CLARIFY'
}
```

### 2. POST /feedback/outcome - Outcome Feedback

Gọi khi action hoàn thành hoặc bị hủy.

```typescript
interface OutcomeFeedbackRequest {
  request_id: string;        // Required: từ response /route
  user_id?: string;          // Optional: ID người dùng
  session_id?: string;       // Optional: Session ID
  action_id?: string;        // Action ID đã thực hiện
  status: 'confirmed' | 'canceled';  // Required: kết quả
}
```

## Frontend Integration Flow

### Step 1: Route Request

```typescript
const response = await api.route(userText);
const { request_id, top_actions, ui_strategy } = response.data;

// Lưu request_id để dùng cho feedback
feedbackContext.requestId = request_id;
feedbackContext.uiStrategy = ui_strategy;
feedbackContext.topActions = top_actions;
```

### Step 2: User Selects Action

```typescript
// Khi user click chọn action
const handleActionSelect = async (actionId: string) => {
  const index = feedbackContext.topActions.findIndex(a => a.action_id === actionId);
  
  // Gửi selection feedback
  await api.feedbackRoute({
    request_id: feedbackContext.requestId,
    user_id: currentUser.id,
    selected_action: actionId,
    selection_index: index,
    selection_source: 'click',
    ui_strategy: feedbackContext.uiStrategy,
  });
  
  // Tiếp tục start action...
};
```

### Step 3: Action Completes

```typescript
// Khi action hoàn thành (COMMITTED/CONFIRMED)
const handleActionComplete = async (status: 'confirmed' | 'canceled') => {
  await api.feedbackOutcome({
    request_id: feedbackContext.requestId,
    user_id: currentUser.id,
    action_id: currentAction.actionId,
    status: status,
  });
};
```

## Selection Source Values

| Value | Mô tả |
|-------|-------|
| `preselect` | Hệ thống tự động chọn (PRESELECT strategy) |
| `click` | User click chọn từ danh sách |

## UI Strategy Values

| Value | Mô tả |
|-------|-------|
| `PRESELECT` | Confidence cao, tự động chọn action đầu tiên |
| `TOP_3` | Hiển thị 3 options cho user chọn |
| `CLARIFY` | Confidence thấp, cần user xác nhận |

## Data Flow

```
User Input
    ↓
POST /route → response.request_id
    ↓
User selects action
    ↓
POST /feedback/route (selection)
    ↓
Action flow (collecting slots)
    ↓
Action completes/cancels
    ↓
POST /feedback/outcome (outcome)
```

## Files Changed

### Frontend

1. `frontend/src/services/api.ts`
   - Added `RouteFeedbackRequest` interface
   - Added `OutcomeFeedbackRequest` interface
   - Added `feedbackRoute()` method
   - Added `feedbackOutcome()` method
   - Updated `RouterOutput` to include `request_id`

2. `frontend/src/hooks/useChatSession.ts`
   - Added `FeedbackContext` to track request_id
   - Added `sendSelectionFeedback()` function
   - Added `sendOutcomeFeedback()` function
   - Updated `startNewAction()` to send selection feedback
   - Updated `handleActionState()` to send outcome feedback

### Backend (Already existed)

1. `app/main.py`
   - `POST /feedback/route` endpoint
   - `POST /feedback/outcome` endpoint

2. `app/utils/feedback_logger.py`
   - `log_feedback_event()` function

## Testing

### Unit Tests
```bash
python -m pytest tests/test_feedback_integration.py -v
```

### E2E Tests
```bash
python scripts/test_feedback_e2e.py --base-url http://localhost:8000
```

### Manual Testing
1. Start backend: `uvicorn app.main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Interact with chatbot
4. Check `logs/router_feedback.jsonl` for feedback entries

## Verification

Sau khi tích hợp, kiểm tra:

1. **Selection feedback được log:**
```bash
grep "selection" logs/router_feedback.jsonl | tail -5
```

2. **Outcome feedback được log:**
```bash
grep "outcome" logs/router_feedback.jsonl | tail -5
```

3. **Learning report có dữ liệu:**
```bash
python scripts/learning_review.py
cat docs/learning/weekly_report.md
```

## Troubleshooting

### Feedback không được log
- Kiểm tra `request_id` có được truyền đúng không
- Kiểm tra network tab trong browser DevTools
- Kiểm tra CORS settings

### Learning report không có selections
- Đảm bảo `request_id` trong feedback match với interactions
- Kiểm tra file `logs/router_interactions.jsonl` có dữ liệu
