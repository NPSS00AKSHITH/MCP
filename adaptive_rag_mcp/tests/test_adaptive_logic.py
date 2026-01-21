
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.server.policy import PolicyEngine, QueryType, RetrievalDecision
from src.retrieval.reranker import CrossEncoderReranker, QualitySignals
from src.server.loop import RetrievalLoop, LoopStep

class TestAdaptiveLogic(unittest.TestCase):

    def setUp(self):
        self.policy = PolicyEngine()
        self.reranker = CrossEncoderReranker()

    def test_reranker_entropy(self):
        """Test entropy calculation."""
        # Case 1: All same scores -> Max entropy (normalized = 1.0)
        # Note: If scores are identical, probability distribution is uniform.
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        entropy = self.reranker._compute_entropy(scores)
        self.assertAlmostEqual(entropy, 1.0)

        # Case 2: One dominant score -> Low entropy
        scores_peaked = np.array([0.9, 0.1, 0.0, 0.0])
        entropy_peaked = self.reranker._compute_entropy(scores_peaked)
        self.assertTrue(entropy_peaked < 0.5)

    def test_reranker_gap(self):
        """Test score gap and signals."""
        scores = [0.9, 0.8, 0.2]
        quality = self.reranker._compute_quality_signals(scores)
        self.assertAlmostEqual(quality.score_gap, 0.1)
        
        scores_gap = [0.95, 0.4, 0.3]
        quality_gap = self.reranker._compute_quality_signals(scores_gap)
        self.assertAlmostEqual(quality_gap.score_gap, 0.55)
        # Score drop is not a negative flag, so we don't expect it in confidence_flags
        # self.assertIn("score_drop", quality_gap.confidence_flags)

    @patch("src.server.policy.get_llm_client")
    def test_policy_classification_llm(self, mock_get_llm):
        """Test LLM-based classification."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Manually set llm on policy (since it was initialized in setUp)
        self.policy.llm = mock_llm
        
        # Case: General Knowledge
        mock_llm.generate_text.return_value = "general_knowledge"
        qt = self.policy._classify_query_llm("Capital of France?")
        self.assertEqual(qt, QueryType.GENERAL_KNOWLEDGE)
        
        # Case: Ambiguous
        mock_llm.generate_text.return_value = "ambiguous"
        qt2 = self.policy._classify_query_llm("The bank")
        self.assertEqual(qt2, QueryType.AMBIGUOUS)

    def test_evaluate_evidence(self):
        """Test evidence sufficiency logic."""
        # Good case
        good_signals = QualitySignals(
            top_score=0.9, score_spread=0.2, mean_score=0.5,
            score_gap=0.2, evidence_entropy=0.5,
            relevant_count=2, total_count=5, confidence_flags=[]
        )
        self.assertTrue(self.policy.evaluate_evidence(good_signals))

        # Bad case: Low score (even if gap is ok)
        bad_signals = QualitySignals(
            top_score=0.5, score_spread=0.1, mean_score=0.3,
            score_gap=0.2, evidence_entropy=0.8,
            relevant_count=1, total_count=5, confidence_flags=["low_top_score"]
        )
        self.assertFalse(self.policy.evaluate_evidence(bad_signals))

    def test_retry_strategy(self):
        """Test retry parameters."""
        history = [MagicMock()] # 1 step in history
        params = self.policy.determine_retry_strategy(history, "query")
        self.assertEqual(params["search_mode"], "dense")
        self.assertEqual(params["max_k"], 10)

        history.append(MagicMock()) # 2 steps
        params2 = self.policy.determine_retry_strategy(history, "query")
        self.assertEqual(params2["search_mode"], "hybrid")
        self.assertEqual(params2["max_k"], 15)

if __name__ == "__main__":
    unittest.main()
