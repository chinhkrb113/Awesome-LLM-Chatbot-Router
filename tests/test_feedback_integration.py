"""
Integration tests for Feedback API endpoints and Learning Loop.
Tests the complete flow from routing to feedback collection.
"""

import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def temp_feedback_file(tmp_path):
    """Create temporary feedback file for testing."""
    feedback_file = tmp_path / "router_feedback.jsonl"
    feedback_file.touch()
    return str(feedback_file)


class TestFeedbackRouteEndpoint:
    """Tests for POST /feedback/route endpoint."""

    def test_selection_feedback_success(self, client):
        """Test successful selection feedback submission."""
        feedback_data = {
            "request_id": "test-request-123",
            "user_id": "test_user",
            "session_id": "test_session",
            "selected_action": "leave.create",
            "selection_index": 0,
            "selection_source": "click",
            "ui_strategy": "TOP_3"
        }
        
        response = client.post("/feedback/route", json=feedback_data)
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_selection_feedback_minimal(self, client):
        """Test selection feedback with minimal required fields."""
        feedback_data = {
            "request_id": "test-request-456"
        }
        
        response = client.post("/feedback/route", json=feedback_data)
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_selection_feedback_preselect(self, client):
        """Test selection feedback for PRESELECT strategy."""
        feedback_data = {
            "request_id": "test-request-789",
            "selected_action": "visitor.create",
            "selection_index": 0,
            "selection_source": "preselect",
            "ui_strategy": "PRESELECT"
        }
        
        response = client.post("/feedback/route", json=feedback_data)
        
        assert response.status_code == 200


class TestFeedbackOutcomeEndpoint:
    """Tests for POST /feedback/outcome endpoint."""

    def test_confirmed_outcome_success(self, client):
        """Test successful confirmed outcome submission."""
        feedback_data = {
            "request_id": "test-request-123",
            "user_id": "test_user",
            "session_id": "test_session",
            "action_id": "leave.create",
            "status": "confirmed"
        }
        
        response = client.post("/feedback/outcome", json=feedback_data)
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_canceled_outcome_success(self, client):
        """Test successful canceled outcome submission."""
        feedback_data = {
            "request_id": "test-request-456",
            "action_id": "visitor.create",
            "status": "canceled"
        }
        
        response = client.post("/feedback/outcome", json=feedback_data)
        
        assert response.status_code == 200

    def test_outcome_missing_status(self, client):
        """Test outcome feedback requires status field."""
        feedback_data = {
            "request_id": "test-request-789",
            "action_id": "leave.create"
        }
        
        response = client.post("/feedback/outcome", json=feedback_data)
        
        # Should fail validation - status is required
        assert response.status_code == 422


class TestRouteToFeedbackFlow:
    """Integration tests for complete route -> feedback flow."""

    @pytest.mark.skipif(True, reason="Requires embedding engine initialization")
    def test_complete_flow_preselect(self, client):
        """Test complete flow: route -> selection -> outcome (PRESELECT)."""
        # Step 1: Route request
        route_response = client.post("/route", json={
            "text": "xin nghỉ phép 2 ngày",
            "user_id": "integration_test"
        })
        
        assert route_response.status_code == 200
        route_data = route_response.json()
        request_id = route_data.get("request_id")
        
        assert request_id is not None
        assert len(route_data["top_actions"]) > 0
        
        # Step 2: Selection feedback
        selected_action = route_data["top_actions"][0]["action_id"]
        selection_response = client.post("/feedback/route", json={
            "request_id": request_id,
            "user_id": "integration_test",
            "selected_action": selected_action,
            "selection_index": 0,
            "selection_source": "preselect",
            "ui_strategy": route_data["ui_strategy"]
        })
        
        assert selection_response.status_code == 200
        
        # Step 3: Outcome feedback
        outcome_response = client.post("/feedback/outcome", json={
            "request_id": request_id,
            "user_id": "integration_test",
            "action_id": selected_action,
            "status": "confirmed"
        })
        
        assert outcome_response.status_code == 200

    @pytest.mark.skipif(True, reason="Requires embedding engine initialization")
    def test_complete_flow_clarify_then_cancel(self, client):
        """Test flow: route (CLARIFY) -> selection -> cancel."""
        # Step 1: Route with ambiguous text
        route_response = client.post("/route", json={
            "text": "nghỉ",
            "user_id": "integration_test"
        })
        
        assert route_response.status_code == 200
        route_data = route_response.json()
        request_id = route_data.get("request_id")
        
        # Step 2: User selects second option
        if len(route_data["top_actions"]) > 1:
            selected_action = route_data["top_actions"][1]["action_id"]
            selection_index = 1
        else:
            selected_action = route_data["top_actions"][0]["action_id"]
            selection_index = 0
            
        selection_response = client.post("/feedback/route", json={
            "request_id": request_id,
            "selected_action": selected_action,
            "selection_index": selection_index,
            "selection_source": "click",
            "ui_strategy": route_data["ui_strategy"]
        })
        
        assert selection_response.status_code == 200
        
        # Step 3: User cancels
        outcome_response = client.post("/feedback/outcome", json={
            "request_id": request_id,
            "action_id": selected_action,
            "status": "canceled"
        })
        
        assert outcome_response.status_code == 200


class TestFeedbackLogging:
    """Tests for feedback logging functionality."""

    def test_feedback_logged_to_file(self, client, temp_feedback_file):
        """Test that feedback is properly logged to file."""
        with patch('app.main.log_feedback_event') as mock_log:
            feedback_data = {
                "request_id": "log-test-123",
                "selected_action": "leave.create",
                "selection_source": "click"
            }
            
            client.post("/feedback/route", json=feedback_data)
            
            # Verify log function was called
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert call_args["event_type"] == "selection"
            assert call_args["request_id"] == "log-test-123"

    def test_outcome_logged_to_file(self, client):
        """Test that outcome is properly logged."""
        with patch('app.main.log_feedback_event') as mock_log:
            feedback_data = {
                "request_id": "outcome-test-456",
                "action_id": "visitor.create",
                "status": "confirmed"
            }
            
            client.post("/feedback/outcome", json=feedback_data)
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert call_args["event_type"] == "outcome"
            assert call_args["status"] == "confirmed"


class TestRequestIdGeneration:
    """Tests for request_id generation in route endpoint."""

    @pytest.mark.skipif(True, reason="Requires embedding engine initialization")
    def test_route_returns_request_id(self, client):
        """Test that /route endpoint returns request_id."""
        response = client.post("/route", json={
            "text": "test message",
            "user_id": "test_user"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert data["request_id"] is not None
        assert len(data["request_id"]) > 0

    @pytest.mark.skipif(True, reason="Requires embedding engine initialization")
    def test_route_uses_provided_request_id(self, client):
        """Test that /route uses provided request_id if given."""
        custom_id = "custom-request-id-12345"
        response = client.post("/route", json={
            "text": "test message",
            "user_id": "test_user",
            "request_id": custom_id
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == custom_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
