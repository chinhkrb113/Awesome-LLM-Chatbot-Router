"""
Script tự động sinh dữ liệu feedback giả lập tích cực dựa trên interactions hiện có.

Usage:
    python scripts/generate_feedback.py [options]

Options:
    --interactions PATH    Path to interactions file (default: logs/router_interactions.jsonl)
    --output PATH          Path to output feedback file (default: logs/router_feedback.jsonl)
    --positive-rate FLOAT  Tỷ lệ feedback tích cực (default: 0.92)
    --batch-size INT       Số lượng xử lý mỗi batch (default: 100)
    --dry-run              Chỉ preview, không ghi file
    --append               Append vào file hiện có thay vì overwrite
    --seed INT             Random seed để reproducible (default: None)
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class FeedbackConfig:
    """Configuration for feedback generation."""
    positive_rate: float = 0.92  # Tỷ lệ feedback tích cực (>90%)
    top1_selection_rate: float = 0.75  # Tỷ lệ chọn action đầu tiên
    preselect_confirm_rate: float = 0.98  # Tỷ lệ confirm khi PRESELECT
    top3_confirm_rate: float = 0.95  # Tỷ lệ confirm khi TOP_3
    clarify_confirm_rate: float = 0.92  # Tỷ lệ confirm khi CLARIFY
    min_response_delay_ms: int = 500  # Delay tối thiểu (ms)
    max_response_delay_ms: int = 5000  # Delay tối đa (ms)
    cancel_reasons: List[str] = None
    
    def __post_init__(self):
        if self.cancel_reasons is None:
            self.cancel_reasons = [
                "user_changed_mind",
                "wrong_action",
                "timeout",
                "error",
            ]


class FeedbackGenerator:
    """Generator for realistic feedback data based on interactions."""
    
    def __init__(self, config: FeedbackConfig = None, seed: int = None):
        self.config = config or FeedbackConfig()
        self.stats = {
            "total_interactions": 0,
            "generated_selections": 0,
            "generated_outcomes": 0,
            "positive_outcomes": 0,
            "negative_outcomes": 0,
            "skipped_no_actions": 0,
            "skipped_no_request_id": 0,
        }
        
        if seed is not None:
            random.seed(seed)
    
    def load_interactions(self, path: str) -> List[Dict[str, Any]]:
        """Load interactions from JSONL file."""
        interactions = []
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Interactions file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    interactions.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
        
        self.stats["total_interactions"] = len(interactions)
        return interactions
    
    def _generate_response_delay(self) -> int:
        """Generate realistic response delay in milliseconds."""
        # Use log-normal distribution for more realistic delays
        # Most responses are quick, some take longer
        mean_delay = (self.config.min_response_delay_ms + self.config.max_response_delay_ms) / 2
        delay = random.gauss(mean_delay, mean_delay / 3)
        return max(
            self.config.min_response_delay_ms,
            min(int(delay), self.config.max_response_delay_ms)
        )
    
    def _select_action(self, interaction: Dict[str, Any]) -> Tuple[str, int, str]:
        """
        Select an action from top_actions based on realistic behavior.
        Returns: (action_id, selection_index, selection_source)
        """
        top_actions = interaction.get("top_actions", [])
        ui_strategy = interaction.get("ui_strategy", "CLARIFY")
        
        if not top_actions:
            return None, -1, None
        
        # PRESELECT: almost always select first action
        if ui_strategy == "PRESELECT":
            return top_actions[0]["action_id"], 0, "preselect"
        
        # TOP_3 or CLARIFY: user clicks to select
        # Higher probability for top-ranked actions
        if random.random() < self.config.top1_selection_rate:
            # Select first action
            return top_actions[0]["action_id"], 0, "click"
        elif len(top_actions) > 1 and random.random() < 0.7:
            # Select second action
            return top_actions[1]["action_id"], 1, "click"
        elif len(top_actions) > 2:
            # Select third or later action
            idx = random.randint(2, min(len(top_actions) - 1, 4))
            return top_actions[idx]["action_id"], idx, "click"
        else:
            # Fallback to first
            return top_actions[0]["action_id"], 0, "click"
    
    def _determine_outcome(self, interaction: Dict[str, Any], selected_action: str) -> str:
        """Determine outcome (confirmed/canceled) based on context."""
        ui_strategy = interaction.get("ui_strategy", "CLARIFY")
        top_actions = interaction.get("top_actions", [])
        
        # Base positive rate based on UI strategy
        if ui_strategy == "PRESELECT":
            positive_rate = self.config.preselect_confirm_rate
        elif ui_strategy == "TOP_3":
            positive_rate = self.config.top3_confirm_rate
        else:  # CLARIFY
            positive_rate = self.config.clarify_confirm_rate
        
        # Small adjustment based on score (không giảm quá nhiều)
        for action in top_actions:
            if action.get("action_id") == selected_action:
                score = action.get("final_score", 0.5)
                if score > 0.8:
                    positive_rate = min(positive_rate + 0.02, 0.99)
                elif score < 0.2:
                    positive_rate = max(positive_rate - 0.02, 0.88)
                break
        
        return "confirmed" if random.random() < positive_rate else "canceled"
    
    def _generate_timestamp(self, base_timestamp: str, delay_ms: int) -> str:
        """Generate timestamp with delay from base timestamp."""
        try:
            base_dt = datetime.fromisoformat(base_timestamp)
        except (ValueError, TypeError):
            base_dt = datetime.now()
        
        new_dt = base_dt + timedelta(milliseconds=delay_ms)
        return new_dt.isoformat()
    
    def generate_feedback_for_interaction(
        self, 
        interaction: Dict[str, Any]
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Generate selection and outcome feedback for a single interaction.
        Returns: (selection_feedback, outcome_feedback) or (None, None) if skipped
        """
        # Validate interaction
        request_id = interaction.get("request_id")
        if not request_id:
            self.stats["skipped_no_request_id"] += 1
            return None, None
        
        top_actions = interaction.get("top_actions", [])
        if not top_actions:
            self.stats["skipped_no_actions"] += 1
            return None, None
        
        # Select action
        selected_action, selection_index, selection_source = self._select_action(interaction)
        if not selected_action:
            return None, None
        
        # Generate timestamps
        base_timestamp = interaction.get("timestamp", datetime.now().isoformat())
        selection_delay = self._generate_response_delay()
        outcome_delay = selection_delay + self._generate_response_delay() * 3  # Outcome takes longer
        
        # Create selection feedback
        selection_feedback = {
            "event_type": "selection",
            "request_id": request_id,
            "user_id": interaction.get("user_id", "simulated_user"),
            "session_id": interaction.get("session_id"),
            "selected_action": selected_action,
            "selection_index": selection_index,
            "selection_source": selection_source,
            "ui_strategy": interaction.get("ui_strategy"),
            "timestamp": self._generate_timestamp(base_timestamp, selection_delay),
            "_simulated": True,  # Mark as simulated data
        }
        self.stats["generated_selections"] += 1
        
        # Determine and create outcome feedback
        outcome_status = self._determine_outcome(interaction, selected_action)
        outcome_feedback = {
            "event_type": "outcome",
            "request_id": request_id,
            "user_id": interaction.get("user_id", "simulated_user"),
            "session_id": interaction.get("session_id"),
            "action_id": selected_action,
            "status": outcome_status,
            "timestamp": self._generate_timestamp(base_timestamp, outcome_delay),
            "_simulated": True,
        }
        self.stats["generated_outcomes"] += 1
        
        if outcome_status == "confirmed":
            self.stats["positive_outcomes"] += 1
        else:
            self.stats["negative_outcomes"] += 1
        
        return selection_feedback, outcome_feedback
    
    def generate_batch(
        self, 
        interactions: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate feedback for a batch of interactions."""
        all_feedback = []
        
        for i in range(0, len(interactions), batch_size):
            batch = interactions[i:i + batch_size]
            
            for interaction in batch:
                selection, outcome = self.generate_feedback_for_interaction(interaction)
                
                if selection:
                    all_feedback.append(selection)
                if outcome:
                    all_feedback.append(outcome)
            
            # Progress indicator
            processed = min(i + batch_size, len(interactions))
            print(f"Processed {processed}/{len(interactions)} interactions...")
        
        return all_feedback
    
    def save_feedback(
        self, 
        feedback: List[Dict[str, Any]], 
        output_path: str,
        append: bool = False
    ) -> None:
        """Save feedback to JSONL file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        mode = "a" if append else "w"
        with open(output_path, mode, encoding="utf-8") as f:
            for item in feedback:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"Saved {len(feedback)} feedback entries to {output_path}")
    
    def print_stats(self) -> None:
        """Print generation statistics."""
        print("\n" + "=" * 50)
        print("Feedback Generation Statistics")
        print("=" * 50)
        print(f"Total interactions processed: {self.stats['total_interactions']}")
        print(f"Skipped (no request_id): {self.stats['skipped_no_request_id']}")
        print(f"Skipped (no actions): {self.stats['skipped_no_actions']}")
        print(f"Generated selections: {self.stats['generated_selections']}")
        print(f"Generated outcomes: {self.stats['generated_outcomes']}")
        print(f"  - Positive (confirmed): {self.stats['positive_outcomes']}")
        print(f"  - Negative (canceled): {self.stats['negative_outcomes']}")
        
        if self.stats['generated_outcomes'] > 0:
            positive_rate = self.stats['positive_outcomes'] / self.stats['generated_outcomes'] * 100
            print(f"  - Positive rate: {positive_rate:.1f}%")
        print("=" * 50)
    
    def validate_feedback(self, feedback: List[Dict[str, Any]]) -> bool:
        """Validate generated feedback data."""
        errors = []
        
        for i, item in enumerate(feedback):
            # Check required fields
            if "event_type" not in item:
                errors.append(f"Item {i}: missing event_type")
            if "request_id" not in item:
                errors.append(f"Item {i}: missing request_id")
            if "timestamp" not in item:
                errors.append(f"Item {i}: missing timestamp")
            
            # Check event-specific fields
            if item.get("event_type") == "selection":
                if "selected_action" not in item:
                    errors.append(f"Item {i}: selection missing selected_action")
            elif item.get("event_type") == "outcome":
                if "status" not in item:
                    errors.append(f"Item {i}: outcome missing status")
                elif item["status"] not in ["confirmed", "canceled"]:
                    errors.append(f"Item {i}: invalid status '{item['status']}'")
        
        if errors:
            print("Validation errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
            return False
        
        print("Validation passed: all feedback entries are valid")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated feedback data from interactions"
    )
    parser.add_argument(
        "--interactions", 
        default="logs/router_interactions.jsonl",
        help="Path to interactions file"
    )
    parser.add_argument(
        "--output", 
        default="logs/router_feedback.jsonl",
        help="Path to output feedback file"
    )
    parser.add_argument(
        "--positive-rate", 
        type=float, 
        default=0.92,
        help="Positive feedback rate (default: 0.92)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=100,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Preview only, don't write to file"
    )
    parser.add_argument(
        "--append", 
        action="store_true",
        help="Append to existing file instead of overwrite"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = FeedbackConfig(positive_rate=args.positive_rate)
    
    # Create generator
    generator = FeedbackGenerator(config=config, seed=args.seed)
    
    print(f"Loading interactions from: {args.interactions}")
    try:
        interactions = generator.load_interactions(args.interactions)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(interactions)} interactions")
    print(f"Generating feedback with {args.positive_rate*100:.0f}% positive rate...")
    
    # Generate feedback
    feedback = generator.generate_batch(interactions, batch_size=args.batch_size)
    
    # Validate
    if not generator.validate_feedback(feedback):
        print("Warning: Some feedback entries failed validation")
    
    # Print stats
    generator.print_stats()
    
    # Save or preview
    if args.dry_run:
        print("\n[DRY RUN] Preview of first 5 feedback entries:")
        for item in feedback[:5]:
            print(json.dumps(item, indent=2, ensure_ascii=False))
    else:
        generator.save_feedback(feedback, args.output, append=args.append)
        print(f"\nFeedback saved to: {args.output}")


if __name__ == "__main__":
    main()
