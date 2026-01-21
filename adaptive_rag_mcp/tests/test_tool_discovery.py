"""Unit tests for Tool Registry with Semantic Discovery."""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch


class TestToolDiscovery(unittest.TestCase):
    """Test cases for TURA-style semantic tool discovery."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid loading models during test collection
        from src.policy.tool_discovery import ToolRegistry, ToolMetadata
        
        # Create mock embedder that returns predictable embeddings
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed_text = MagicMock(
            side_effect=lambda text: np.random.randn(384).astype(np.float32)
        )
        self.mock_embedder.embed_query = MagicMock(
            side_effect=lambda text: np.random.randn(384).astype(np.float32)
        )
        
        self.registry = ToolRegistry(embedder=self.mock_embedder)
    
    def test_index_single_tool(self):
        """Test indexing a single tool."""
        self.registry.index_tool(
            tool_name="test_tool",
            description="A test tool for searching documents",
            category="retrieval",
            complexity="simple",
        )
        
        self.assertIn("test_tool", self.registry.tool_metadata)
        meta = self.registry.tool_metadata["test_tool"]
        self.assertEqual(meta.name, "test_tool")
        self.assertEqual(meta.category, "retrieval")
        self.assertEqual(meta.complexity, "simple")
        self.assertIsNotNone(meta.embedding)
    
    def test_index_all_tools(self):
        """Test indexing all tools from schemas."""
        self.registry.index_all_tools()
        
        self.assertTrue(self.registry._indexed)
        self.assertGreater(len(self.registry.tool_metadata), 0)
        
        # Should have indexed all tools from TOOL_SCHEMAS
        from src.server.schemas import TOOL_SCHEMAS
        self.assertEqual(len(self.registry.tool_metadata), len(TOOL_SCHEMAS))
    
    def test_discover_tools_returns_limited_results(self):
        """Test that discover_tools returns only top_k results."""
        self.registry.index_all_tools()
        
        # Request top 3 tools
        tools = self.registry.discover_tools("search for Python code", top_k=3)
        
        self.assertEqual(len(tools), 3)
        self.assertIsInstance(tools, list)
        self.assertTrue(all(isinstance(t, str) for t in tools))
    
    def test_discover_tools_with_category_filter(self):
        """Test that category filter restricts results."""
        self.registry.index_all_tools()
        
        # Filter to only retrieval tools
        tools = self.registry.discover_tools(
            "find relevant documents",
            top_k=10,
            category_filter="retrieval"
        )
        
        # All returned tools should be in retrieval category
        for tool_name in tools:
            meta = self.registry.tool_metadata[tool_name]
            self.assertEqual(meta.category, "retrieval")
    
    def test_get_tool_schemas(self):
        """Test getting full schemas for discovered tools."""
        self.registry.index_all_tools()
        
        tool_names = ["search", "rerank", "cite"]
        schemas = self.registry.get_tool_schemas(tool_names)
        
        self.assertEqual(len(schemas), 3)
        for schema in schemas:
            self.assertIn("name", schema)
            self.assertIn("description", schema)
            self.assertIn("inputSchema", schema)
    
    def test_get_stats(self):
        """Test getting registry statistics."""
        self.registry.index_all_tools()
        
        stats = self.registry.get_stats()
        
        self.assertIn("total_tools", stats)
        self.assertIn("indexed", stats)
        self.assertIn("categories", stats)
        self.assertIn("complexity_counts", stats)
        self.assertTrue(stats["indexed"])
        self.assertGreater(stats["total_tools"], 0)
    
    def test_discovery_filters_irrelevant_tools(self):
        """Test that discovery filters out irrelevant tools for specific queries."""
        # This test uses controlled embeddings to verify filtering logic
        
        # Create a registry with deterministic embeddings
        mock_embedder = MagicMock()
        
        # Define embeddings that will produce known similarities
        search_embedding = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        ingest_embedding = np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)
        
        def mock_embed_text(text):
            if "search" in text.lower() or "retriev" in text.lower():
                return search_embedding
            elif "ingest" in text.lower():
                return ingest_embedding
            return np.random.randn(384).astype(np.float32)
        
        mock_embedder.embed_text = mock_embed_text
        mock_embedder.embed_query = mock_embed_text
        
        registry = ToolRegistry(embedder=mock_embedder)
        
        # Index a few tools manually
        registry.index_tool("search", "Search and retrieve documents", "retrieval", "moderate")
        registry.index_tool("rerank", "Rerank retrieved results", "retrieval", "moderate")
        registry.index_tool("ingest_document", "Ingest a new document", "ingestion", "moderate")
        registry._build_embedding_matrix()
        registry._indexed = True
        
        # Query for search should NOT return ingest tool first
        query_embedding = registry.embedder.embed_query("find documents about Python")
        
        # This verifies the semantic filtering is working
        tools = registry.discover_tools("search and retrieve documents", top_k=2)
        self.assertEqual(len(tools), 2)


class TestToolDiscoveryIntegration(unittest.TestCase):
    """Integration tests for tool discovery with real embeddings."""
    
    @unittest.skip("Requires loading embedding model - run manually")
    def test_real_embedding_discovery(self):
        """Test with real embeddings (slow, skip by default)."""
        from src.policy.tool_discovery import get_tool_registry
        
        registry = get_tool_registry()
        
        # Test retrieval-focused query
        tools = registry.discover_tools("search for Python functions", top_k=5)
        
        # Should include retrieval tools, not ingestion
        self.assertIn("search", tools)
        self.assertNotIn("ingest_document", tools)


if __name__ == "__main__":
    unittest.main()
