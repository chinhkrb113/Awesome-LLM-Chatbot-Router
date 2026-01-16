import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Any, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    except FileNotFoundError:
        return []
    return items


def build_interaction_map(interactions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id = {}
    for item in interactions:
        request_id = item.get("request_id")
        if request_id:
            by_id[request_id] = item
    return by_id


def find_action_scores(top_actions: List[Dict[str, Any]], action_id: str) -> Dict[str, Any]:
    for action in top_actions or []:
        if action.get("action_id") == action_id:
            return action
    return {}


def generate_report(
    interactions_path: str,
    feedback_path: str,
    output_path: str,
    rule_threshold: float = 0.4,
    embed_threshold: float = 0.6,
) -> None:
    interactions = read_jsonl(interactions_path)
    feedback = read_jsonl(feedback_path)
    interaction_map = build_interaction_map(interactions)

    selection_events = [e for e in feedback if e.get("event_type") == "selection"]
    outcome_events = [e for e in feedback if e.get("event_type") == "outcome"]

    top1_correct = 0
    total_selections = 0
    clarify_count = 0
    cancel_count = 0
    outcome_count = 0

    action_stats = defaultdict(lambda: {"selected": 0, "top1_correct": 0, "candidates": []})

    for interaction in interactions:
        if interaction.get("ui_strategy") == "CLARIFY":
            clarify_count += 1

    for event in selection_events:
        request_id = event.get("request_id")
        selected_action = event.get("selected_action")
        if not request_id or not selected_action:
            continue
        interaction = interaction_map.get(request_id)
        if not interaction:
            continue

        total_selections += 1
        action_stats[selected_action]["selected"] += 1

        top_actions = interaction.get("top_actions", [])
        top1_action = top_actions[0]["action_id"] if top_actions else None
        is_top1 = selected_action == top1_action
        if is_top1:
            top1_correct += 1
            action_stats[selected_action]["top1_correct"] += 1

        user_text = interaction.get("user_text")
        selected_scores = find_action_scores(top_actions, selected_action)
        rule_score = selected_scores.get("rule_score")
        embed_score = selected_scores.get("embed_score")

        if user_text and (not is_top1 or (
            rule_score is not None and embed_score is not None
            and rule_score < rule_threshold
            and embed_score < embed_threshold
        )):
            action_stats[selected_action]["candidates"].append(user_text)

    for event in outcome_events:
        status = event.get("status")
        if not status:
            continue
        outcome_count += 1
        if status == "canceled":
            cancel_count += 1

    total_interactions = len(interactions)
    top1_click_rate = (top1_correct / total_selections) if total_selections else 0.0
    clarify_rate = (clarify_count / total_interactions) if total_interactions else 0.0
    cancel_rate = (cancel_count / outcome_count) if outcome_count else 0.0

    lines = []
    lines.append("# Learning Loop Weekly Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- total_interactions: {total_interactions}")
    lines.append(f"- total_selections: {total_selections}")
    lines.append(f"- top1_click_rate: {top1_click_rate:.3f}")
    lines.append(f"- clarify_rate: {clarify_rate:.3f}")
    lines.append(f"- cancel_after_click_rate: {cancel_rate:.3f}")
    lines.append("")
    lines.append("## Action Breakdown")
    for action_id, stats in sorted(action_stats.items()):
        selected = stats["selected"]
        top1_rate = (stats["top1_correct"] / selected) if selected else 0.0
        lines.append(f"### {action_id}")
        lines.append(f"- selected: {selected}")
        lines.append(f"- top1_correct_rate: {top1_rate:.3f}")
        candidates = stats["candidates"][:10]
        if candidates:
            lines.append("- seed_phrase_candidates:")
            for text in candidates:
                lines.append(f"  - {text}")
        lines.append("")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote report to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate learning loop review report from logs.")
    parser.add_argument("--interactions", default="logs/router_interactions.jsonl")
    parser.add_argument("--feedback", default="logs/router_feedback.jsonl")
    parser.add_argument("--output", default="docs/learning/weekly_report.md")
    parser.add_argument("--rule-threshold", type=float, default=0.4)
    parser.add_argument("--embed-threshold", type=float, default=0.6)
    args = parser.parse_args()

    generate_report(
        interactions_path=args.interactions,
        feedback_path=args.feedback,
        output_path=args.output,
        rule_threshold=args.rule_threshold,
        embed_threshold=args.embed_threshold,
    )


if __name__ == "__main__":
    main()
