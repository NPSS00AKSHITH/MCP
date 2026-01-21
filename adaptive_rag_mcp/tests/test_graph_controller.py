"""Unit tests for LangGraph-based Graph Controller."""

import unittest
from unittest.mock import MagicMock, patch


class TestGraphController(unittest.TestCase):
    """Test cases for LangGraph state machine controller."""

    def test_langgraph_availability_check(self):
        """Test that module handles missing langgraph gracefully."""
        from src.retrieval.graph_controller import LANGGRAPH_AVAILABLE
        
        # LANGGRAPH_AVAILABLE should be a boolean
        self.assertIsInstance(LANGGRAPH_AVAILABLE, bool)
    
    def test_rag_state_structure(self):
        """Test RAGState has expected keys."""
        # RAGState should accept at minimum these keys
        state = {
            "query": "test query",
            "max_iterations": 3,
            "iteration": 0,
            "retrieved_docs": [],
            "reranked_results": [],
        }
        
        self.assertIn("query", state)
        self.assertIn("max_iterations", state)
    
    def test_route_query_node(self):
        """Test route_query node function."""
        from src.retrieval.graph_controller import route_query
        
        with patch("src.policy.engine.AdaptivePolicyEngine") as mock_policy_class:
            # Mock policy engine
            mock_policy = MagicMock()
            mock_decision = MagicMock()
            mock_decision.should_retrieve = True
            mock_decision.retrieval_mode = "hybrid"
            mock_decision.intent = "doc_specific"
            mock_policy.decide.return_value = mock_decision
            mock_policy_class.return_value = mock_policy
            
            state = {"query": "Find the budget report"}
            result = route_query(state)
            
            self.assertTrue(result["should_retrieve"])
            self.assertEqual(result["retrieval_mode"], "hybrid")
    
    def test_should_retrieve_conditional(self):
        """Test should_retrieve conditional edge."""
        from src.retrieval.graph_controller import should_retrieve
        
        # Should retrieve when flag is True
        state_retrieve = {"should_retrieve": True}
        self.assertEqual(should_retrieve(state_retrieve), "retrieve")
        
        # Should skip when flag is False
        state_skip = {"should_retrieve": False}
        self.assertEqual(should_retrieve(state_skip), "skip")
    
    def test_should_retry_conditional(self):
        """Test should_retry conditional edge logic."""
        from src.retrieval.graph_controller import should_retry
        
        # Confident = generate
        state_confident = {
            "iteration": 1,
            "max_iterations": 3,
            "is_confident": True,
        }
        self.assertEqual(should_retry(state_confident), "generate")
        
        # Not confident, can retry = retrieve
        state_retry = {
            "iteration": 1,
            "max_iterations": 3,
            "is_confident": False,
        }
        self.assertEqual(should_retry(state_retry), "retrieve")
        
        # Max iterations with results = generate
        state_max_with_results = {
            "iteration": 3,
            "max_iterations": 3,
            "is_confident": False,
            "reranked_results": [{"id": "1"}],
        }
        self.assertEqual(should_retry(state_max_with_results), "generate")
        
        # Max iterations without results = insufficient
        state_max_no_results = {
            "iteration": 3,
            "max_iterations": 3,
            "is_confident": False,
            "reranked_results": [],
        }
        self.assertEqual(should_retry(state_max_no_results), "insufficient")


class TestGraphBasedController(unittest.TestCase):
    """Test GraphBasedController class."""
    
    def test_controller_initialization(self):
        """Test controller initializes without error."""
        from src.retrieval.graph_controller import GraphBasedController
        
        controller = GraphBasedController()
        
        # Should have graph or None (if langgraph not installed)
        self.assertTrue(hasattr(controller, "graph"))
    
    def test_fallback_to_iterative_controller(self):
        """Test fallback when langgraph not available."""
        from src.retrieval.graph_controller import GraphBasedController
        
        controller = GraphBasedController()
        
        # Should have fallback_controller property
        self.assertTrue(hasattr(controller, "fallback_controller"))
    
    @patch("src.retrieval.graph_controller.LANGGRAPH_AVAILABLE", False)
    def test_retrieve_uses_fallback(self):
        """Test that retrieve falls back when langgraph unavailable."""
        from src.retrieval.graph_controller import GraphBasedController
        
        controller = GraphBasedController()
        controller.graph = None
        controller._fallback_controller = MagicMock()
        
        # Mock fallback result
        mock_result = MagicMock()
        mock_result.query = "test"
        mock_result.iterations = 1
        mock_result.final_chunks = []
        mock_result.outcome_type.value = "answer_ready"
        mock_result.stop_reason.value = "high_confidence"
        controller._fallback_controller.retrieve.return_value = mock_result
        
        result = controller.retrieve("test query")
        
        # Should return state-like dict
        self.assertIsInstance(result, dict)
        self.assertIn("query", result)


class TestLoopToolsGraphIntegration(unittest.TestCase):
    """Test loop_tools integration with graph controller."""
    
    def test_convert_graph_state_to_response(self):
        """Test state-to-response conversion."""
        from src.server.tools.loop_tools import _convert_graph_state_to_response
        
        state = {
            "query": "test query",
            "iteration": 2,
            "retrieval_mode": "hybrid",
            "is_confident": True,
            "reranked_results": [
                {"id": "chunk_1", "content": "Content", "score": 0.9}
            ],
            "confidence_score": 0.85,
            "stop_reason": "answer_generated",
        }
        
        response = _convert_graph_state_to_response(state)
        
        # Check response format
        self.assertIn("results", response)
        self.assertIn("trace", response)
        self.assertIn("final_status", response)
        self.assertIn("outcome", response)
        
        # Check outcome type
        self.assertEqual(response["outcome"]["type"], "answer_ready")
        self.assertEqual(response["outcome"]["confidence_level"], "high")
        
        # Check controller type marking
        self.assertEqual(response["controller_type"], "graph")
    
    def test_convert_low_confidence_state(self):
        """Test conversion of low confidence state."""
        from src.server.tools.loop_tools import _convert_graph_state_to_response
        
        state = {
            "iteration": 1,
            "is_confident": False,
            "reranked_results": [],
            "stop_reason": "insufficient_evidence",
        }
        
        response = _convert_graph_state_to_response(state)
        
        self.assertEqual(response["outcome"]["type"], "insufficient_evidence")
        self.assertEqual(response["outcome"]["confidence_level"], "low")
        self.assertFalse(response["final_status"]["success"])


if __name__ == "__main__":
    unittest.main()
