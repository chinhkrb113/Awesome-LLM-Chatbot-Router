# Learning Loop (Operational)

This is the minimal, production-friendly loop for improving routing without heavy training.

## 1) Capture feedback
After `/route`, the client posts what the user selected:

```
POST /feedback/route
{
  "request_id": "<from /route response>",
  "user_id": "user_123",
  "session_id": "session_abc",
  "selected_action": "visitor.create",
  "selection_index": 0,
  "selection_source": "preselect",
  "ui_strategy": "PRESELECT"
}
```

After the action finishes, post the outcome:

```
POST /feedback/outcome
{
  "request_id": "<same as above>",
  "user_id": "user_123",
  "session_id": "session_abc",
  "action_id": "visitor.create",
  "status": "confirmed"
}
```

Events are stored in `logs/router_feedback.jsonl`.

## 2) Weekly review
Run:

```
python scripts/learning_review.py --output docs/learning/weekly_report.md
```

The report includes:
- top1_click_rate
- clarify_rate
- cancel_after_click_rate
- seed phrase candidates (based on low rule/embed scores or wrong top1)

## 3) Apply updates
Use the report to update:
- `config/action_catalog.yaml` (seed phrases)
- `config/keyword_rules.yaml` (strong/weak/negative keywords)

Keep updates small and reversible.

## 4) Auto mode (smart run)
Enable smart auto-run by editing `config/learning_loop.yaml`:
- `enabled: true`
- set thresholds and run window

Run once:
```
python scripts/auto_learning.py --config config/learning_loop.yaml
```

Run continuously:
```
python scripts/auto_learning.py --config config/learning_loop.yaml --loop
```

Or start with the API server:
```
set LEARNING_LOOP_AUTO=1
uvicorn app.main:app --reload
```
