import datetime
import json
import os
import threading
import time
from typing import Dict, Any, Tuple

import yaml

from scripts.learning_review import generate_report, read_jsonl
from scripts.auto_tuner import AutoTuner


DEFAULT_CONFIG = {
    "enabled": False,
    "interactions_path": "logs/router_interactions.jsonl",
    "feedback_path": "logs/router_feedback.jsonl",
    "report_path": "docs/learning/weekly_report.md",
    "state_path": "logs/learning_state.json",
    "action_catalog_path": "config/action_catalog.yaml",
    "min_selections": 20,
    "min_outcomes": 10,
    "min_new_selections": 10,
    "min_hours_between_runs": 24,
    "run_window_start": 22,
    "run_window_end": 6,
    "check_interval_minutes": 60,
    "rule_threshold": 0.4,
    "embed_threshold": 0.6,
    "auto_tune": False,
}


def _merge_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    merged.update(raw or {})
    return merged


def load_learning_config(path: str = "config/learning_loop.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        return dict(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _merge_config(data.get("learning_loop", {}))


def _parse_iso(ts: str) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(ts)
    except Exception:
        return datetime.datetime.min


def _is_in_window(now: datetime.datetime, start_hour: int, end_hour: int) -> bool:
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= now.hour < end_hour
    return now.hour >= start_hour or now.hour < end_hour


def _read_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _count_feedback(feedback_path: str) -> Tuple[int, int]:
    feedback = read_jsonl(feedback_path)
    selections = sum(1 for e in feedback if e.get("event_type") == "selection")
    outcomes = sum(1 for e in feedback if e.get("event_type") == "outcome")
    return selections, outcomes


def maybe_run_learning_loop(config_path: str = "config/learning_loop.yaml") -> Dict[str, Any]:
    cfg = load_learning_config(config_path)
    if not cfg.get("enabled"):
        return {"status": "skipped", "reason": "disabled"}

    now = datetime.datetime.now()
    if not _is_in_window(now, int(cfg["run_window_start"]), int(cfg["run_window_end"])):
        return {"status": "skipped", "reason": "outside_run_window"}

    state = _read_state(cfg["state_path"])
    last_run_at = _parse_iso(state.get("last_run_at", ""))
    hours_since_run = (now - last_run_at).total_seconds() / 3600 if last_run_at else 1e9
    if hours_since_run < float(cfg["min_hours_between_runs"]):
        return {"status": "skipped", "reason": "cooldown"}

    selections, outcomes = _count_feedback(cfg["feedback_path"])
    if selections < int(cfg["min_selections"]) or outcomes < int(cfg["min_outcomes"]):
        return {"status": "skipped", "reason": "not_enough_feedback"}

    last_selections = int(state.get("selection_count", 0))
    new_selections = selections - last_selections
    if new_selections < int(cfg["min_new_selections"]):
        return {"status": "skipped", "reason": "not_enough_new_feedback"}

    # 1. Generate Report (Keep existing logic)
    generate_report(
        interactions_path=cfg["interactions_path"],
        feedback_path=cfg["feedback_path"],
        output_path=cfg["report_path"],
        rule_threshold=float(cfg["rule_threshold"]),
        embed_threshold=float(cfg["embed_threshold"]),
    )

    # 2. Auto Tune (New Feature)
    tune_result = None
    if cfg.get("auto_tune"):
        from scripts.learning_review import generate_report as get_report_data
        # Hacky way to get data, ideally refactor generate_report to return data struct
        # For now, let's re-implement candidate extraction quickly here or rely on report
        # Better: Refactor generate_report in scripts/learning_review.py to be reusable.
        # But for minimal change, let's extract candidates again.
        
        # NOTE: This is a simplified extraction for AutoTuner
        # Reuse logic from generate_report
        interactions = read_jsonl(cfg["interactions_path"])
        feedback = read_jsonl(cfg["feedback_path"])
        
        # Build map
        by_id = {item["request_id"]: item for item in interactions if "request_id" in item}
        
        candidates_map = {} # {action_id: [text1, text2]}
        
        for event in feedback:
            if event.get("event_type") == "selection":
                req_id = event.get("request_id")
                sel_action = event.get("selected_action")
                if req_id in by_id and sel_action:
                    interaction = by_id[req_id]
                    # Check scores
                    # Logic: If confidence low, add to candidates
                    # Simplified for now: just trust user selection if score low
                    top_actions = interaction.get("top_actions", [])
                    
                    # Find score of selected action
                    sel_score = 0.0
                    for a in top_actions:
                        if a["action_id"] == sel_action:
                            sel_score = a.get("final_score", 0.0)
                            break
                    
                    if sel_score < 0.8: # Threshold for auto-tuning
                         if sel_action not in candidates_map:
                             candidates_map[sel_action] = []
                         candidates_map[sel_action].append(interaction["user_text"])

        tuner = AutoTuner(cfg["action_catalog_path"])
        tune_result = tuner.tune(candidates_map)

    _write_state(
        cfg["state_path"],
        {
            "last_run_at": now.isoformat(),
            "selection_count": selections,
            "outcome_count": outcomes,
            "tune_result": tune_result
        },
    )

    return {"status": "ran", "report_path": cfg["report_path"], "tune_result": tune_result}


def start_learning_loop_thread(config_path: str = "config/learning_loop.yaml") -> None:
    def _worker():
        while True:
            try:
                maybe_run_learning_loop(config_path)
            except Exception:
                pass
            cfg = load_learning_config(config_path)
            sleep_seconds = int(cfg.get("check_interval_minutes", 60)) * 60
            time.sleep(max(60, sleep_seconds))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
