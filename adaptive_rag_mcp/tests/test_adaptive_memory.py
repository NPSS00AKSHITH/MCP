"""Unit tests for Adaptive Memory Ranker."""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestAdaptiveMemory(unittest.TestCase):
    """Test cases for adaptive memory ranker."""

    def setUp(self):
        """Set up test fixtures with temp directory."""
        from src.retrieval.adaptive_memory import AdaptiveMemoryRanker
        from src.retrieval.reranker import RerankResult, QualitySignals
        
        # Use temp file for memory
        self.temp_dir = tempfile.mkdtemp()
        self.memory_path = Path(self.temp_dir) / "test_memory.json"
        
        # Create mock reranker
        self.mock_reranker = MagicMock()
        self.mock_results = [
            RerankResult(id="chunk_1", content="Content 1", relevance_score=0.9, original_rank=0),
            RerankResult(id="chunk_2", content="Content 2", relevance_score=0.7, original_rank=1),
            RerankResult(id="chunk_3", content="Content 3", relevance_score=0.5, original_rank=2),
        ]
        self.mock_quality = QualitySignals(
            top_score=0.9, score_spread=0.4, mean_score=0.7,
            score_gap=0.2, evidence_entropy=0.5,
            relevant_count=2, total_count=3, confidence_flags=[]
        )
        self.mock_reranker.rerank.return_value = (self.mock_results, self.mock_quality)
        
        self.memory = AdaptiveMemoryRanker(
            base_reranker=self.mock_reranker,
            memory_path=self.memory_path,
        )
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rerank_with_memory_returns_results(self):
        """Test basic reranking returns results."""
        documents = [
            {"id": "chunk_1", "content": "Content 1"},
            {"id": "chunk_2", "content": "Content 2"},
        ]
        
        results, quality = self.memory.rerank_with_memory(
            query="test query",
            documents=documents,
        )
        
        self.assertEqual(len(results), 3)
        self.assertIsNotNone(quality)
    
    def test_record_feedback_updates_success_rate(self):
        """Test that recording feedback updates success rates."""
        self.memory.record_feedback(
            chunk_id="chunk_1",
            query="test query",
            query_type="doc_specific",
            accepted=True,
            original_score=0.9,
        )
        
        # Check success rate was updated
        self.assertIn("chunk_1", self.memory.chunk_success_rates)
        # After one positive feedback, rate should be above 0.5
        self.assertGreater(self.memory.chunk_success_rates["chunk_1"], 0.5)
    
    def test_record_feedback_persists_to_disk(self):
        """Test that feedback is persisted to disk."""
        self.memory.record_feedback(
            chunk_id="chunk_1",
            query="test query",
            query_type="doc_specific",
            accepted=True,
        )
        
        # Check file was created
        self.assertTrue(self.memory_path.exists())
        
        # Verify contents
        with open(self.memory_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn("chunk_success_rates", data)
        self.assertIn("chunk_1", data["chunk_success_rates"])
    
    def test_memory_boost_applied_to_successful_chunks(self):
        """Test that successful chunks get boosted scores."""
        # Record positive feedback for chunk_1
        self.memory.chunk_success_rates["chunk_1"] = 0.9  # High success rate
        
        documents = [
            {"id": "chunk_1", "content": "Content 1"},
            {"id": "chunk_2", "content": "Content 2"},
        ]
        
        results, _ = self.memory.rerank_with_memory(
            query="test query",
            documents=documents,
        )
        
        # Find chunk_1 in results
        chunk_1_result = next((r for r in results if r.id == "chunk_1"), None)
        
        # Should have memory_boost in metadata
        self.assertIsNotNone(chunk_1_result)
        self.assertIn("memory_boost", chunk_1_result.metadata)
        self.assertGreater(chunk_1_result.metadata["memory_boost"], 0)
    
    def test_query_pattern_learning(self):
        """Test that query patterns are learned."""
        # Record positive feedback with query type
        self.memory.record_feedback(
            chunk_id="chunk_1",
            query="find budget report",
            query_type="doc_specific",
            accepted=True,
        )
        
        # Check pattern was recorded
        self.assertIn("doc_specific", self.memory.query_patterns)
        self.assertIn("chunk_1", self.memory.query_patterns["doc_specific"])
    
    def test_negative_feedback_removes_from_patterns(self):
        """Test that negative feedback removes from patterns."""
        # First add to patterns
        self.memory.query_patterns["doc_specific"].append("chunk_1")
        
        # Then record negative feedback
        self.memory.record_feedback(
            chunk_id="chunk_1",
            query="find budget report",
            query_type="doc_specific",
            accepted=False,
        )
        
        # Should be removed from pattern
        self.assertNotIn("chunk_1", self.memory.query_patterns["doc_specific"])
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        self.memory.record_feedback("chunk_1", "q1", "type1", True)
        self.memory.record_feedback("chunk_2", "q2", "type2", True)
        
        stats = self.memory.get_stats()
        
        self.assertEqual(stats["chunks_tracked"], 2)
        self.assertEqual(stats["total_feedback"], 2)
        self.assertIn("query_patterns", stats)
    
    def test_clear_memory(self):
        """Test memory clearing."""
        self.memory.record_feedback("chunk_1", "q1", "type1", True)
        self.memory.clear_memory()
        
        self.assertEqual(len(self.memory.chunk_success_rates), 0)
        self.assertEqual(len(self.memory.feedback_history), 0)
    
    def test_memory_load_on_init(self):
        """Test that memory is loaded from disk on init."""
        # Save some data
        data = {
            "chunk_success_rates": {"chunk_1": 0.8},
            "query_patterns": {"doc_specific": ["chunk_1"]},
            "feedback_history": [],
            "last_updated": 0,
        }
        with open(self.memory_path, 'w') as f:
            json.dump(data, f)
        
        # Create new memory instance
        from src.retrieval.adaptive_memory import AdaptiveMemoryRanker
        new_memory = AdaptiveMemoryRanker(
            base_reranker=self.mock_reranker,
            memory_path=self.memory_path,
        )
        
        # Should have loaded data
        self.assertIn("chunk_1", new_memory.chunk_success_rates)
        self.assertEqual(new_memory.chunk_success_rates["chunk_1"], 0.8)
    
    def test_empty_documents_returns_empty_results(self):
        """Test handling of empty document list."""
        results, quality = self.memory.rerank_with_memory(
            query="test query",
            documents=[],
        )
        
        self.assertEqual(len(results), 0)
        self.assertIn("no_documents", quality.confidence_flags)


if __name__ == "__main__":
    unittest.main()
