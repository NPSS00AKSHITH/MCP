"""TURA-style semantic tool discovery to prevent prompt bloat.

When agent has 50+ MCP servers available, don't load all tool schemas.
Instead: semantic search against tool descriptions → load only top-k relevant.

Design Philosophy:
- On init: Index all tool descriptions as vectors
- On query: Semantic search → return top-k relevant tool names
- Agent only sees descriptions of relevant tools

Example:
    >>> registry = ToolRegistry()
    >>> registry.index_all_tools()
    >>> relevant = registry.discover_tools("Search for code examples", top_k=3)
    >>> # Returns: ["search", "rerank", "cite"] - NOT all 17 tools
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from src.retrieval.embedder import get_embedder
from src.server.logging import get_logger
from src.server.schemas import TOOL_SCHEMAS

logger = get_logger(__name__)


# =============================================================================
# TOOL METADATA
# =============================================================================

@dataclass
class ToolMetadata:
    """Rich metadata for a tool, used for discovery.
    
    Attributes:
        name: Tool identifier (e.g., "adaptive_retrieve")
        description: Rich description for semantic matching
        category: Tool category for filtering ("ingestion", "retrieval", etc.)
        complexity: Complexity level ("simple", "moderate", "complex")
        embedding: Cached embedding vector (computed on indexing)
    """
    name: str
    description: str
    category: str
    complexity: str
    embedding: Optional[np.ndarray] = None


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """Semantic index of available MCP tools.
    
    Prevents prompt bloat by only loading relevant tools for each query.
    Uses sentence embeddings to find semantically similar tools.
    
    Design:
    - On init: Index all tool descriptions as vectors
    - On query: Semantic search → return top-k relevant tool names
    - Agent only sees descriptions of relevant tools
    
    Example:
        >>> registry = ToolRegistry()
        >>> registry.index_all_tools()
        >>> relevant = registry.discover_tools("Search for code examples", top_k=3)
        >>> # Returns: ["search", "rerank", "cite"] - NOT all 17 tools
    """
    
    # Default category mappings based on tool purpose
    DEFAULT_CATEGORIES = {
        "adaptive_retrieve": ("retrieval", "complex"),
        "decide_retrieval": ("retrieval", "simple"),
        "embed_query": ("retrieval", "simple"),
        "search": ("retrieval", "moderate"),
        "rerank": ("retrieval", "moderate"),
        "summarize": ("generation", "moderate"),
        "cite": ("generation", "moderate"),
        "compare_documents": ("analysis", "complex"),
        "generate_response": ("generation", "moderate"),
        "ingest_document": ("ingestion", "moderate"),
        "list_documents": ("ingestion", "simple"),
        "get_document_chunks": ("ingestion", "simple"),
        "delete_document": ("ingestion", "simple"),
        "get_ingestion_stats": ("ingestion", "simple"),
        "index_document": ("retrieval", "moderate"),
        "get_retrieval_stats": ("retrieval", "simple"),
    }
    
    def __init__(self, embedder=None):
        """Initialize the tool registry.
        
        Args:
            embedder: Optional embedder instance. Uses global embedder if not provided.
        """
        self.embedder = embedder or get_embedder()
        self.tool_metadata: Dict[str, ToolMetadata] = {}
        self._indexed = False
        self._embedding_matrix: Optional[np.ndarray] = None
        self._tool_names: List[str] = []
    
    def index_tool(
        self,
        tool_name: str,
        description: str,
        category: str,
        complexity: str
    ) -> None:
        """Index a single tool for semantic discovery.
        
        Args:
            tool_name: Tool identifier (e.g., "adaptive_retrieve")
            description: Rich description of what tool does
            category: Tool category for filtering
            complexity: Complexity level for routing decisions
        """
        # Embed the description
        embedding = self.embedder.embed_text(description)
        
        # Store metadata with embedding
        metadata = ToolMetadata(
            name=tool_name,
            description=description,
            category=category,
            complexity=complexity,
            embedding=embedding,
        )
        self.tool_metadata[tool_name] = metadata
        
        logger.debug(
            "tool_indexed",
            tool_name=tool_name,
            category=category,
            complexity=complexity,
        )
    
    def index_all_tools(self) -> None:
        """Index all tools from TOOL_SCHEMAS registry.
        
        Uses default category/complexity mappings if not specified in schema.
        """
        if self._indexed:
            logger.debug("tools_already_indexed", count=len(self.tool_metadata))
            return
        
        logger.info("indexing_tools", total=len(TOOL_SCHEMAS))
        
        for tool_name, schema in TOOL_SCHEMAS.items():
            description = schema.get("description", tool_name)
            
            # Get category and complexity from schema or defaults
            category = schema.get(
                "category",
                self.DEFAULT_CATEGORIES.get(tool_name, ("analysis", "moderate"))[0]
            )
            complexity = schema.get(
                "complexity",
                self.DEFAULT_CATEGORIES.get(tool_name, ("analysis", "moderate"))[1]
            )
            
            self.index_tool(tool_name, description, category, complexity)
        
        # Build embedding matrix for efficient batch similarity
        self._build_embedding_matrix()
        self._indexed = True
        
        logger.info(
            "tools_indexed",
            count=len(self.tool_metadata),
            categories=list(set(t.category for t in self.tool_metadata.values())),
        )
    
    def _build_embedding_matrix(self) -> None:
        """Build embedding matrix for efficient cosine similarity search."""
        self._tool_names = list(self.tool_metadata.keys())
        embeddings = [
            self.tool_metadata[name].embedding
            for name in self._tool_names
        ]
        self._embedding_matrix = np.vstack(embeddings)
        # Normalize for cosine similarity
        norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
        self._embedding_matrix = self._embedding_matrix / (norms + 1e-10)
    
    def discover_tools(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[str]:
        """Discover relevant tools for a query via semantic search.
        
        This is the CORE TURA innovation: instead of loading all tools,
        semantically match the query against tool descriptions.
        
        Args:
            query: User query or intent description
            top_k: Number of tools to return
            category_filter: Optional category to restrict search
            
        Returns:
            List of tool names sorted by relevance
            
        Example:
            >>> discover_tools("Find documents about Python", top_k=3)
            ['search', 'rerank', 'cite']  # NOT all 17 tools
        """
        if not self._indexed:
            logger.warning("tools_not_indexed", msg="Call index_all_tools() first")
            return list(TOOL_SCHEMAS.keys())[:top_k]
        
        # Embed the query
        query_embedding = self.embedder.embed_query(query)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Compute cosine similarity
        similarities = np.dot(self._embedding_matrix, query_embedding)
        
        # Apply category filter if specified
        if category_filter:
            mask = np.array([
                self.tool_metadata[name].category == category_filter
                for name in self._tool_names
            ])
            similarities = np.where(mask, similarities, -np.inf)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return tool names
        discovered = [self._tool_names[i] for i in top_indices]
        
        logger.info(
            "tools_discovered",
            query_length=len(query),
            top_k=top_k,
            category_filter=category_filter,
            discovered=discovered,
            top_scores=[round(float(similarities[i]), 4) for i in top_indices],
        )
        
        return discovered
    
    def get_tool_schemas(self, tool_names: List[str]) -> List[dict]:
        """Get full schemas for discovered tools.
        
        After discovery, load only the relevant tool schemas.
        
        Args:
            tool_names: List of tool names to get schemas for.
            
        Returns:
            List of tool schema dictionaries.
        """
        return [TOOL_SCHEMAS[name] for name in tool_names if name in TOOL_SCHEMAS]
    
    def get_stats(self) -> dict:
        """Return registry statistics.
        
        Returns:
            Dictionary with total_tools, indexed status, and categories.
        """
        return {
            "total_tools": len(self.tool_metadata),
            "indexed": self._indexed,
            "categories": list(set(t.category for t in self.tool_metadata.values())),
            "complexity_counts": {
                "simple": sum(1 for t in self.tool_metadata.values() if t.complexity == "simple"),
                "moderate": sum(1 for t in self.tool_metadata.values() if t.complexity == "moderate"),
                "complex": sum(1 for t in self.tool_metadata.values() if t.complexity == "complex"),
            },
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create global tool registry.
    
    Lazily initializes and indexes the registry on first call.
    
    Returns:
        The global ToolRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _registry.index_all_tools()
    return _registry
