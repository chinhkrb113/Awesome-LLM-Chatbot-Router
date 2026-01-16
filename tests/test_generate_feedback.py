"""
Unit tests for feedback generation script.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_feedback import FeedbackConfig, FeedbackGenerator


@pytest.fixture
def sample_interactions():
    """Sample interactions for testing."""
    return [
        {
            "request_id": "test-001",
            "timestamp": "2026-01-14T10:00:00.000000",
            "user_text": "xin nghỉ phép",
            "user_id": "user_1",
            "top_actions": [
                {"action_id": "leave.create", "final_score": 0.95, "rule_score": 0.8, "embed_score": 0.9},
                {"action_id": "leave.status", "final_score": 0.6, "rule_score": 0.3, "embed_score": 0.7},
            ],
            "ui_strategy": "PRESELECT",
        },
        {
            "request_id": "test-002",
            "timestamp": "2026-01-14T10:05:00.000000",
            "user_text": "tạo visitor",
            "user_id": "user_2",
            "top_actions": [
                {"action_id": "visitor.create", "final_score": 0.7, "rule_score": 0.5, "embed_score": 0.6},
                {"action_id": "visitor.status", "final_score": 0.5, "rule_score": 0.2, "embed_score": 0.5},
                {"action_id": "leave.create", "final_score": 0.3, "rule_score": 0.1, "embed_score": 0.3},
            ],
            "ui_strategy": "TOP_3",
        },
        {
            "request_id": "test-003",
            "timestamp": "2026-01-14T10:10:00.000000",
            "user_text": "nghỉ",
            "user_id": "user_1",
            "top_actions": [
                {"action_id": "leave.create", "final_score": 0.4, "rule_score": 0.2, "embed_score": 0.4},
                {"action_id": "leave.status", "final_score": 0.35, "rule_score": 0.1, "embed_score": 0.35},
            ],
            "ui_strategy": "CLARIFY",
        },
    ]


@pytest.fixture
def temp_interactions_file(sample_interactions, tmp_path):
    """Create temporary interactions file."""
    file_path = tmp_path / "test_interactions.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in sample_interactions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(file_path)


@pytest.fixture
def generator():
    """Create generator with fixed seed for reproducibility."""
    config = FeedbackConfig()
    return FeedbackGenerator(config=config, seed=42)


class TestFeedbackConfig:
    """Tests for FeedbackConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FeedbackConfig()
        assert config.positive_rate == 0.92
        assert config.top1_selection_rate == 0.75
        assert config.preselect_confirm_rate == 0.98
        assert config.clarify_confirm_rate == 0.92

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FeedbackConfig(positive_rate=0.95, top1_selection_rate=0.8)
        assert config.positive_rate == 0.95
        assert config.top1_selection_rate == 0.8


class TestFeedbackGenerator:
    """Tests for FeedbackGenerator."""

    def test_load_interactions(self, generator, temp_interactions_file):
        """Test loading interactions from file."""
        interactions = generator.load_interactions(temp_interactions_file)
        assert len(interactions) == 3
        assert interactions[0]["request_id"] == "test-001"

    def test_load_interactions_file_not_found(self, generator):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            generator.load_interactions("nonexistent_file.jsonl")

    def test_generate_feedback_for_interaction(self, generator, sample_interactions):
        """Test generating feedback for a single interaction."""
        selection, outcome = generator.generate_feedback_for_interaction(sample_interactions[0])
        
        assert selection is not None
        assert outcome is not None
        
        # Check selection feedback
        assert selection["event_type"] == "selection"
        assert selection["request_id"] == "test-001"
        assert selection["selected_action"] in ["leave.create", "leave.status"]
        assert selection["selection_index"] >= 0
        assert selection["_simulated"] is True
        
        # Check outcome feedback
        assert outcome["event_type"] == "outcome"
        assert outcome["request_id"] == "test-001"
        assert outcome["status"] in ["confirmed", "canceled"]

    def test_skip_interaction_without_request_id(self, generator):
        """Test skipping interactions without request_id."""
        interaction = {
            "user_text": "test",
            "top_actions": [{"action_id": "test.action", "final_score": 0.5}],
        }
        selection, outcome = generator.generate_feedback_for_interaction(interaction)
        
        assert selection is None
        assert outcome is None
        assert generator.stats["skipped_no_request_id"] == 1

    def test_skip_interaction_without_actions(self, generator):
        """Test skipping interactions without top_actions."""
        interaction = {
            "request_id": "test-skip",
            "user_text": "test",
            "top_actions": [],
        }
        selection, outcome = generator.generate_feedback_for_interaction(interaction)
        
        assert selection is None
        assert outcome is None
        assert generator.stats["skipped_no_actions"] == 1

    def test_preselect_strategy(self, generator, sample_interactions):
        """Test PRESELECT strategy always selects first action."""
        preselect_interaction = sample_interactions[0]  # PRESELECT
        
        # Run multiple times to verify consistency
        for _ in range(10):
            selection, _ = generator.generate_feedback_for_interaction(preselect_interaction)
            assert selection["selection_source"] == "preselect"
            assert selection["selection_index"] == 0

    def test_generate_batch(self, generator, sample_interactions):
        """Test batch generation."""
        feedback = generator.generate_batch(sample_interactions, batch_size=2)
        
        # Should have 2 feedback items per interaction (selection + outcome)
        assert len(feedback) == 6
        
        # Check alternating selection/outcome
        assert feedback[0]["event_type"] == "selection"
        assert feedback[1]["event_type"] == "outcome"

    def test_validate_feedback(self, generator, sample_interactions):
        """Test feedback validation."""
        feedback = generator.generate_batch(sample_interactions)
        assert generator.validate_feedback(feedback) is True

    def test_validate_feedback_invalid(self, generator):
        """Test validation catches invalid feedback."""
        invalid_feedback = [
            {"event_type": "selection"},  # Missing request_id
            {"request_id": "test", "event_type": "outcome"},  # Missing status
        ]
        assert generator.validate_feedback(invalid_feedback) is False

    def test_save_feedback(self, generator, sample_interactions, tmp_path):
        """Test saving feedback to file."""
        feedback = generator.generate_batch(sample_interactions)
        output_path = str(tmp_path / "test_output.jsonl")
        
        generator.save_feedback(feedback, output_path)
        
        # Verify file was created and has correct content
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 6

    def test_save_feedback_append(self, generator, sample_interactions, tmp_path):
        """Test appending feedback to existing file."""
        output_path = str(tmp_path / "test_append.jsonl")
        
        # First save
        feedback1 = generator.generate_batch(sample_interactions[:1])
        generator.save_feedback(feedback1, output_path)
        
        # Append
        feedback2 = generator.generate_batch(sample_interactions[1:2])
        generator.save_feedback(feedback2, output_path, append=True)
        
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 4  # 2 + 2

    def test_timestamp_generation(self, generator, sample_interactions):
        """Test timestamp is properly generated with delay."""
        selection, outcome = generator.generate_feedback_for_interaction(sample_interactions[0])
        
        base_ts = sample_interactions[0]["timestamp"]
        selection_ts = selection["timestamp"]
        outcome_ts = outcome["timestamp"]
        
        # Selection should be after base
        assert selection_ts > base_ts
        # Outcome should be after selection
        assert outcome_ts > selection_ts

    def test_positive_rate_above_90(self, sample_interactions):
        """Test that positive rate is above 90% with enough samples."""
        config = FeedbackConfig()
        generator = FeedbackGenerator(config=config, seed=123)
        
        # Generate many samples
        large_interactions = sample_interactions * 100  # 300 interactions
        feedback = generator.generate_batch(large_interactions, batch_size=50)
        
        # Calculate actual positive rate
        outcomes = [f for f in feedback if f["event_type"] == "outcome"]
        confirmed = sum(1 for o in outcomes if o["status"] == "confirmed")
        positive_rate = confirmed / len(outcomes)
        
        assert positive_rate >= 0.90, f"Positive rate {positive_rate:.2%} is below 90%"


class TestPerformance:
    """Performance tests for large datasets."""

    def test_large_batch_performance(self, tmp_path):
        """Test performance with large number of interactions."""
        import time
        
        # Create large dataset
        large_interactions = []
        for i in range(1000):
            large_interactions.append({
                "request_id": f"perf-test-{i}",
                "timestamp": "2026-01-14T10:00:00.000000",
                "user_text": f"test message {i}",
                "user_id": f"user_{i % 10}",
                "top_actions": [
                    {"action_id": "action.one", "final_score": 0.8},
                    {"action_id": "action.two", "final_score": 0.5},
                ],
                "ui_strategy": ["PRESELECT", "TOP_3", "CLARIFY"][i % 3],
            })
        
        generator = FeedbackGenerator(seed=42)
        
        start_time = time.time()
        feedback = generator.generate_batch(large_interactions, batch_size=100)
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 5 seconds for 1000 items)
        assert elapsed < 5.0, f"Generation took too long: {elapsed:.2f}s"
        assert len(feedback) == 2000  # 2 feedback per interaction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
