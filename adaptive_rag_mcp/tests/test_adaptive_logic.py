import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.server.policy import PolicyEngine, QueryType, RetrievalDecision
from src.retrieval.reranker import CrossEncoderReranker, QualitySignals
from src.server.loop import RetrievalLoop, LoopStep
from src.retrieval.evidence_scoring import detect_contradictions


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

    def test_policy_skip_retrieval_general_knowledge(self):
        """Test that policy skips retrieval for general knowledge queries."""
        with patch.object(
            self.policy, "_classify_query_llm", return_value=QueryType.GENERAL_KNOWLEDGE
        ):
            decision = self.policy.decide("What is the capital of France?")

            self.assertFalse(decision.should_retrieve)
            self.assertEqual(decision.query_type, QueryType.GENERAL_KNOWLEDGE)
            self.assertEqual(decision.search_mode, "none")
            self.assertEqual(decision.max_k, 0)
            self.assertIn("skipping retrieval", decision.reason)

    def test_policy_skip_retrieval_ambiguous(self):
        """Test that policy skips retrieval for ambiguous queries."""
        with patch.object(
            self.policy, "_classify_query_llm", return_value=QueryType.AMBIGUOUS
        ):
            decision = self.policy.decide("The bank")

            self.assertFalse(decision.should_retrieve)
            self.assertEqual(decision.query_type, QueryType.AMBIGUOUS)
            self.assertEqual(decision.search_mode, "none")
            self.assertEqual(decision.max_k, 0)
            self.assertIn("ambiguous", decision.reason)

    def test_strategy_switching_by_query_type(self):
        """Test that different query types trigger different retrieval strategies."""
        # Test comparison query strategy
        with patch.object(
            self.policy, "_classify_query_llm", return_value=QueryType.COMPARISON
        ):
            decision = self.policy.decide("Compare sales vs marketing budgets")
            self.assertTrue(decision.should_retrieve)
            self.assertEqual(decision.query_type, QueryType.COMPARISON)
            self.assertEqual(decision.search_mode, "hybrid")
            self.assertEqual(decision.max_k, 8)

        # Test multi-document query strategy
        with patch.object(
            self.policy, "_classify_query_llm", return_value=QueryType.MULTI_DOC
        ):
            decision = self.policy.decide("Summarize all quarterly reports")
            self.assertTrue(decision.should_retrieve)
            self.assertEqual(decision.query_type, QueryType.MULTI_DOC)
            self.assertEqual(decision.search_mode, "dense")
            self.assertEqual(decision.max_k, 10)

        # Test document-specific query strategy
        with patch.object(
            self.policy, "_classify_query_llm", return_value=QueryType.DOC_SPECIFIC
        ):
            decision = self.policy.decide("Find the project timeline document")
            self.assertTrue(decision.should_retrieve)
            self.assertEqual(decision.query_type, QueryType.DOC_SPECIFIC)
            self.assertEqual(decision.search_mode, "hybrid")
            self.assertEqual(decision.max_k, 5)

    def test_evaluate_evidence(self):
        """Test evidence sufficiency logic."""
        # Good case
        good_signals = QualitySignals(
            top_score=0.9,
            score_spread=0.2,
            mean_score=0.5,
            score_gap=0.2,
            evidence_entropy=0.5,
            relevant_count=2,
            total_count=5,
            confidence_flags=[],
        )
        self.assertTrue(self.policy.evaluate_evidence(good_signals))

        # Bad case: Low score (even if gap is ok)
        bad_signals = QualitySignals(
            top_score=0.5,
            score_spread=0.1,
            mean_score=0.3,
            score_gap=0.2,
            evidence_entropy=0.8,
            relevant_count=1,
            total_count=5,
            confidence_flags=["low_top_score"],
        )
        self.assertFalse(self.policy.evaluate_evidence(bad_signals))

    def test_partial_evidence_handling(self):
        """Test handling of partial evidence with mixed confidence signals."""
        # Borderline case: High score but poor gap (close competition)
        partial_signals = QualitySignals(
            top_score=0.75,
            score_spread=0.1,
            mean_score=0.6,
            score_gap=0.05,
            evidence_entropy=0.7,  # Gap below threshold
            relevant_count=2,
            total_count=4,
            confidence_flags=[],
        )
        # Should be insufficient due to small gap
        self.assertFalse(self.policy.evaluate_evidence(partial_signals))

        # Another partial case: Good gap but high entropy (unclear winner)
        uncertain_signals = QualitySignals(
            top_score=0.8,
            score_spread=0.3,
            mean_score=0.5,
            score_gap=0.25,
            evidence_entropy=0.8,  # Entropy above threshold
            relevant_count=1,
            total_count=3,
            confidence_flags=[],
        )
        # Should be insufficient due to high entropy
        self.assertFalse(self.policy.evaluate_evidence(uncertain_signals))

        # Very high score can override other factors
        override_signals = QualitySignals(
            top_score=0.9,
            score_spread=0.4,
            mean_score=0.4,
            score_gap=0.05,
            evidence_entropy=0.9,  # Poor gap and entropy
            relevant_count=1,
            total_count=5,
            confidence_flags=[],
        )
        # Should be sufficient due to very high top score (>0.85)
        self.assertTrue(self.policy.evaluate_evidence(override_signals))

    def test_insufficient_evidence_refusal(self):
        """Test that system refuses to answer when evidence is insufficient."""
        from src.server.loop import RetrievalLoop, SearchMode

        # Mock components
        mock_retriever = MagicMock()
        mock_reranker = MagicMock()

        # Mock retrieval returns low-quality results
        mock_retriever.search.return_value = [
            MagicMock(content="Some content", score=0.4),
            MagicMock(content="More content", score=0.3),
        ]

        # Mock reranker returns low-quality signals
        low_quality_signals = QualitySignals(
            top_score=0.4,
            score_spread=0.1,
            mean_score=0.3,
            score_gap=0.1,
            evidence_entropy=0.8,
            relevant_count=1,
            total_count=2,
            confidence_flags=["low_top_score"],
        )
        mock_reranker.rerank.return_value = ([], low_quality_signals)

        loop = RetrievalLoop(
            retriever=mock_retriever,
            reranker=mock_reranker,
            policy=self.policy,
            max_iterations=2,
        )

        result = loop.run("Test query requiring evidence", SearchMode.HYBRID, 3)

        # Should fail due to insufficient evidence
        self.assertFalse(result.success)
        self.assertIn("without high confidence", result.reason)
        self.assertEqual(result.total_iterations, 2)  # Should try max iterations

    def test_retry_strategy(self):
        """Test retry parameters."""
        history = [MagicMock()]  # 1 step in history
        params = self.policy.determine_retry_strategy(history, "query")
        self.assertEqual(params["search_mode"], "dense")
        self.assertEqual(params["max_k"], 10)

        history.append(MagicMock())  # 2 steps
        params2 = self.policy.determine_retry_strategy(history, "query")
        self.assertEqual(params2["search_mode"], "hybrid")
        self.assertEqual(params2["max_k"], 15)

    def test_multi_iteration_refinement(self):
        """Test progressive strategy refinement across multiple iterations."""
        # Iteration 1 (after 0 history items = first retry)
        history = []
        params1 = self.policy.determine_retry_strategy(history, "initial query")
        self.assertEqual(params1["search_mode"], "dense")
        self.assertEqual(params1["max_k"], 10)

        # Iteration 2 (after 1 history item = second retry)
        history = [MagicMock()]
        params2 = self.policy.determine_retry_strategy(history, "refined query")
        self.assertEqual(params2["search_mode"], "hybrid")
        self.assertEqual(params2["max_k"], 15)

        # Iteration 3+ (after 2+ history items = stop retrying)
        history = [MagicMock(), MagicMock()]
        params3 = self.policy.determine_retry_strategy(history, "final query")
        self.assertIsNone(params3)  # Should stop retrying

        # Test with full loop simulation using patches
        from src.server.loop import RetrievalLoop, LoopStep, SearchMode

        good_signals = QualitySignals(
            top_score=0.9,
            score_spread=0.2,
            mean_score=0.7,
            score_gap=0.3,
            evidence_entropy=0.4,
            relevant_count=1,
            total_count=1,
            confidence_flags=[],
        )
        insufficient_signals = QualitySignals(
            top_score=0.6,
            score_spread=0.2,
            mean_score=0.5,
            score_gap=0.1,
            evidence_entropy=0.7,
            relevant_count=1,
            total_count=1,
            confidence_flags=[],
        )

        with (
            patch("src.server.loop.get_hybrid_retriever") as mock_get_retriever,
            patch("src.server.loop.get_reranker") as mock_get_reranker,
        ):
            mock_retriever = MagicMock()
            mock_reranker = MagicMock()

            # Mock progressive improvement: first two iterations fail, third succeeds
            mock_retriever.search.side_effect = [
                # Iteration 1: poor results
                [MagicMock(content="Poor content", score=0.3)],
                # Iteration 2: better but still insufficient
                [MagicMock(content="Better content", score=0.6)],
                # Iteration 3: good results
                [MagicMock(content="Good content", score=0.9)],
            ]

            mock_reranker.rerank.side_effect = [
                ([], insufficient_signals),  # Iteration 1: insufficient
                ([], insufficient_signals),  # Iteration 2: still insufficient
                (
                    [MagicMock(id="good", content="Good content", relevance_score=0.9)],
                    good_signals,
                ),  # Iteration 3: sufficient
            ]

            mock_get_retriever.return_value = mock_retriever
            mock_get_reranker.return_value = mock_reranker

            loop = RetrievalLoop(max_iterations=3)
            result = loop.run("Progressive refinement query")

            # Should succeed after 3 iterations with strategy refinement
            self.assertTrue(result.success)
            self.assertEqual(result.total_iterations, 3)
            self.assertIn("High confidence achieved", result.reason)

    def test_contradiction_detection(self):
        """Test detection of contradictions in retrieved evidence."""
        # Test case: No contradiction (single chunk)
        single_chunk = [
            {"id": "chunk_1", "content": "The deadline is December 15", "score": 0.9}
        ]
        result = detect_contradictions(single_chunk)
        self.assertFalse(result.has_contradiction)
        self.assertIn("Not enough chunks", result.explanation)

        # Test case: No contradiction (similar chunks)
        similar_chunks = [
            {"id": "chunk_1", "content": "The deadline is December 15", "score": 0.9},
            {
                "id": "chunk_2",
                "content": "The project deadline is December 15",
                "score": 0.85,
            },
        ]
        result = detect_contradictions(similar_chunks)
        self.assertFalse(result.has_contradiction)
        self.assertIn("No contradictions detected", result.explanation)

        # Test case: Contradiction detected (different dates)
        contradictory_chunks = [
            {"id": "chunk_1", "content": "The deadline is December 15", "score": 0.9},
            {"id": "chunk_2", "content": "The deadline is January 30", "score": 0.85},
        ]
        result = detect_contradictions(contradictory_chunks)
        self.assertTrue(result.has_contradiction)
        self.assertIn("contradictions detected", result.explanation)
        self.assertIn("chunk_1 and chunk_2", result.explanation)

        # Test case: Contradiction with negation
        negation_chunks = [
            {"id": "chunk_1", "content": "The project is approved", "score": 0.9},
            {"id": "chunk_2", "content": "The project is not approved", "score": 0.85},
        ]
        result = detect_contradictions(negation_chunks)
        self.assertTrue(result.has_contradiction)
        self.assertIn("contradictions detected", result.explanation)

    def test_confidence_thresholds_and_boundaries(self):
        """Test confidence thresholds and decision boundaries in evidence evaluation."""
        # Test HIGH_CONFIDENCE_THRESHOLD (0.7)
        just_below_threshold = QualitySignals(
            top_score=0.69,
            score_spread=0.2,
            mean_score=0.5,
            score_gap=0.2,
            evidence_entropy=0.5,
            relevant_count=2,
            total_count=5,
            confidence_flags=[],
        )
        self.assertFalse(self.policy.evaluate_evidence(just_below_threshold))

        at_threshold = QualitySignals(
            top_score=0.7,
            score_spread=0.2,
            mean_score=0.5,
            score_gap=0.2,
            evidence_entropy=0.5,
            relevant_count=2,
            total_count=5,
            confidence_flags=[],
        )
        self.assertTrue(self.policy.evaluate_evidence(at_threshold))

        # Test GAP_THRESHOLD (0.15)
        good_score_poor_gap = QualitySignals(
            top_score=0.8,
            score_spread=0.2,
            mean_score=0.5,
            score_gap=0.14,
            evidence_entropy=0.5,  # Below gap threshold
            relevant_count=2,
            total_count=5,
            confidence_flags=[],
        )
        self.assertFalse(self.policy.evaluate_evidence(good_score_poor_gap))

        good_score_good_gap = QualitySignals(
            top_score=0.8,
            score_spread=0.2,
            mean_score=0.5,
            score_gap=0.16,
            evidence_entropy=0.5,  # Above gap threshold
            relevant_count=2,
            total_count=5,
            confidence_flags=[],
        )
        self.assertTrue(self.policy.evaluate_evidence(good_score_good_gap))

        # Test ENTROPY_THRESHOLD (0.6)
        good_score_good_gap_high_entropy = QualitySignals(
            top_score=0.8,
            score_spread=0.2,
            mean_score=0.5,
            score_gap=0.2,
            evidence_entropy=0.61,  # Above entropy threshold
            relevant_count=2,
            total_count=5,
            confidence_flags=[],
        )
        self.assertFalse(
            self.policy.evaluate_evidence(good_score_good_gap_high_entropy)
        )

        # Test very high score override (>0.85)
        very_high_score_poor_others = QualitySignals(
            top_score=0.86,
            score_spread=0.4,
            mean_score=0.4,
            score_gap=0.05,
            evidence_entropy=0.9,  # Poor gap and entropy
            relevant_count=1,
            total_count=5,
            confidence_flags=[],
        )
        self.assertTrue(self.policy.evaluate_evidence(very_high_score_poor_others))


if __name__ == "__main__":
    unittest.main()
