import datetime
import json
import os
from typing import Dict, Any


def log_feedback_event(event: Dict[str, Any], log_path: str = "logs/router_feedback.jsonl") -> None:
    event = dict(event)
    event.setdefault("timestamp", datetime.datetime.now().isoformat())
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
