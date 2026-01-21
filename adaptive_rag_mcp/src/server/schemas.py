"""JSON Schemas for MCP tools - derived from Phase 0 contract."""

from typing import Any

# Tool schemas registry
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "adaptive_retrieve": {
        "name": "adaptive_retrieve",
        "description": "Iteratively retrieves and refines results to find high-quality documents.",
        "category": "retrieval",
        "complexity": "complex",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "User query to answer"
                },
                "max_iterations": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum number of retry attempts"
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.6,
                    "description": "Score threshold to stop early"
                }
            },
            "required": ["query"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "score": {"type": "number"}
                        }
                    }
                },
                "trace": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "integer"},
                            "strategy": {"type": "string"},
                            "retrieved": {"type": "integer"},
                            "top_score": {"type": "number"},
                            "confident": {"type": "boolean"}
                        }
                    }
                },
                "final_status": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "iterations": {"type": "integer"}
                    }
                },
                "_meta": {
                    "type": "object",
                    "properties": {
                        "version": {"type": "string", "default": "1.0.0"}
                    }
                }
            },
            "required": ["results", "final_status"]
        }
    },
    "decide_retrieval": {
        "name": "decide_retrieval",
        "description": "Analyzes a query to determine optimal retrieval strategy. Returns recommendation only - does not execute retrieval.",
        "category": "retrieval",
        "complexity": "simple",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's natural language query"
                },
                "collection_id": {
                    "type": "string",
                    "description": "Target collection to consider for retrieval",
                    "default": "default"
                },
                "context_hint": {
                    "type": ["string", "null"],
                    "description": "Optional hint about conversation context or domain",
                    "default": None
                }
            },
            "required": ["query"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["none", "single", "multi_step", "hybrid"]
                },
                "complexity": {
                    "type": "string",
                    "enum": ["simple", "moderate", "complex"]
                },
                "reasoning": {"type": "string"},
                "suggested_k": {"type": "integer"},
                "query_rewrites": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "metadata": {"type": "object"}
            },
            "required": ["strategy", "complexity", "reasoning", "suggested_k"]
        }
    },
    
    "embed_query": {
        "name": "embed_query",
        "description": "Generates vector embedding for a query or text passage. Use before calling search.",
        "category": "retrieval",
        "complexity": "simple",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to embed"
                },
                "model": {
                    "type": "string",
                    "description": "Embedding model identifier",
                    "default": "all-MiniLM-L6-v2"
                }
            },
            "required": ["text"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "embedding": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "dimensions": {"type": "integer"},
                "model_used": {"type": "string"},
                "metadata": {"type": "object"}
            },
            "required": ["embedding", "dimensions", "model_used"]
        }
    },
    
    "search": {
        "name": "search",
        "description": "Performs similarity search in a collection. Supports dense (vector), sparse (BM25), or hybrid modes.",
        "category": "retrieval",
        "complexity": "moderate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "mode": {
                    "type": "string",
                    "enum": ["dense", "sparse", "hybrid"],
                    "description": "Search mode: dense (vector), sparse (BM25), or hybrid (RRF fusion)",
                    "default": "hybrid"
                },
                "embedding": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "description": "Pre-computed query embedding (optional, for dense mode)",
                    "default": None
                },
                "collection_id": {
                    "type": "string",
                    "default": "default"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 100
                },
                "filter": {
                    "type": ["object", "null"],
                    "description": "Metadata filter",
                    "default": None
                },
                "include_embeddings": {
                    "type": "boolean",
                    "default": False
                }
            },
            "required": ["query"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string"},
                            "chunk_id": {"type": "string"},
                            "content": {"type": "string"},
                            "score": {"type": "number"},
                            "metadata": {"type": "object"},
                            "dense_score": {"type": ["number", "null"]},
                            "sparse_score": {"type": ["number", "null"]},
                            "dense_rank": {"type": ["integer", "null"]},
                            "sparse_rank": {"type": ["integer", "null"]}
                        },
                        "required": ["document_id", "chunk_id", "content", "score"]
                    }
                },
                "total_searched": {"type": "integer"},
                "mode": {"type": "string"},
                "mode": {"type": "string"},
                "metadata": {"type": "object"},
                "_meta": {
                    "type": "object",
                    "properties": {
                        "version": {"type": "string", "default": "1.0.0"}
                    }
                }
            },
            "required": ["results", "total_searched", "mode"]
        }
    },
    
    "rerank": {
        "name": "rerank",
        "description": "Re-ranks documents by relevance to query using cross-encoder. Returns quality signals for evidence assessment.",
        "category": "retrieval",
        "complexity": "moderate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query for relevance scoring"
                },
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "metadata": {"type": "object"}
                        },
                        "required": ["id", "content"]
                    },
                    "description": "Documents to rerank"
                },
                "top_k": {
                    "type": ["integer", "null"],
                    "description": "Return only top K after reranking",
                    "default": None
                },
                "model": {
                    "type": "string",
                    "enum": ["cross-encoder", "simple"],
                    "description": "Reranker type",
                    "default": "cross-encoder"
                },
                "threshold": {
                    "type": "number",
                    "description": "Relevance threshold for quality signals",
                    "default": 0.3,
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["query", "documents"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "relevance_score": {"type": "number"},
                            "original_rank": {"type": "integer"},
                            "metadata": {"type": "object"}
                        },
                        "required": ["id", "content", "relevance_score"]
                    }
                },
                "quality_signals": {
                    "type": "object",
                    "properties": {
                        "top_score": {"type": "number"},
                        "score_spread": {"type": "number"},
                        "mean_score": {"type": "number"},
                        "relevant_count": {"type": "integer"},
                        "total_count": {"type": "integer"},
                        "confidence_flags": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "is_high_confidence": {"type": "boolean"}
                    }
                },
                "model_used": {"type": "string"},
                "metadata": {"type": "object"}
            },
            "required": ["results", "quality_signals", "model_used"]
        }
    },
    
    "summarize": {
        "name": "summarize",
        "description": "Generates a summary from provided context. Does NOT retrieve - context must be supplied.",
        "category": "generation",
        "complexity": "moderate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["content"]
                    },
                    "description": "Context passages to summarize"
                },
                "query": {
                    "type": ["string", "null"],
                    "description": "Optional query to focus the summary",
                    "default": None
                },
                "style": {
                    "type": "string",
                    "enum": ["concise", "detailed", "bullet_points", "executive"],
                    "default": "concise"
                },
                "max_length": {
                    "type": "integer",
                    "default": 200
                }
            },
            "required": ["context"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "metadata": {"type": "object"}
            },
            "required": ["summary"]
        }
    },
    
    "cite": {
        "name": "cite",
        "description": "Generates a response with inline citations to source documents.",
        "category": "generation",
        "complexity": "moderate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "User question to answer"
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "title": {"type": ["string", "null"]},
                            "url": {"type": ["string", "null"]}
                        },
                        "required": ["id", "content"]
                    },
                    "description": "Source documents for citation"
                },
                "citation_style": {
                    "type": "string",
                    "enum": ["inline_number", "inline_name", "footnote"],
                    "default": "inline_number"
                },
                "require_citation": {
                    "type": "boolean",
                    "default": True
                }
            },
            "required": ["query", "sources"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "marker": {"type": "string"},
                            "source_id": {"type": "string"},
                            "excerpt": {"type": "string"},
                            "title": {"type": ["string", "null"]},
                            "url": {"type": ["string", "null"]}
                        },
                        "required": ["marker", "source_id", "excerpt"]
                    }
                },
                "uncited_claims": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "metadata": {"type": "object"}
            },
            "required": ["response", "citations"]
        }
    },
    
    "compare_documents": {
        "name": "compare_documents",
        "description": "Compares multiple documents, identifying similarities, differences, and conflicts.",
        "category": "analysis",
        "complexity": "complex",
        "inputSchema": {
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "label": {"type": "string"}
                        },
                        "required": ["id", "content"]
                    },
                    "minItems": 2,
                    "maxItems": 5
                },
                "focus": {
                    "type": ["string", "null"],
                    "description": "Optional aspect to focus comparison on",
                    "default": None
                },
                "comparison_type": {
                    "type": "string",
                    "enum": ["full", "factual_claims", "sentiment", "key_points"],
                    "default": "full"
                }
            },
            "required": ["documents"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "similarities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "document_ids": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                },
                "differences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "aspect": {"type": "string"},
                            "positions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "document_id": {"type": "string"},
                                        "stance": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                "conflicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "conflicting_sources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "document_id": {"type": "string"},
                                        "excerpt": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                "summary": {"type": "string"},
                "metadata": {"type": "object"}
            },
            "required": ["similarities", "differences", "summary"]
        }
    },
    
    "generate_response": {
        "name": "generate_response",
        "description": "Generates a natural language response to a query using provided context.",
        "category": "generation",
        "complexity": "moderate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "User query to respond to"
                },
                "context": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "source": {"type": ["string", "null"]}
                        },
                        "required": ["content"]
                    },
                    "description": "Retrieved context passages"
                },
                "system_prompt": {
                    "type": ["string", "null"],
                    "default": None
                },
                "temperature": {
                    "type": "number",
                    "default": 0.7,
                    "minimum": 0,
                    "maximum": 2
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 500
                }
            },
            "required": ["query", "context"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "context_used": {"type": "boolean"},
                "confidence": {"type": ["number", "null"]},
                "metadata": {"type": "object"}
            },
            "required": ["response", "context_used"]
        }
    },
    
    # Phase 2: Ingestion tools
    "ingest_document": {
        "name": "ingest_document",
        "description": "Ingest a document from file path or raw text. Chunks and stores for retrieval.",
        "category": "ingestion",
        "complexity": "moderate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": ["string", "null"],
                    "description": "Path to file to ingest (PDF, Markdown, or Text)"
                },
                "text": {
                    "type": ["string", "null"],
                    "description": "Raw text to ingest (alternative to file_path)"
                },
                "doc_id": {
                    "type": "string",
                    "description": "Document ID (required for raw text, auto-generated for files)"
                },
                "file_name": {
                    "type": "string",
                    "description": "Optional name for the document"
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata"
                }
            },
            "required": []
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "file_name": {"type": "string"},
                "file_type": {"type": "string"},
                "total_chunks": {"type": "integer"},
                "total_characters": {"type": "integer"},
                "success": {"type": "boolean"},
                "error": {"type": ["string", "null"]}
            },
            "required": ["success"]
        }
    },
    
    "list_documents": {
        "name": "list_documents",
        "description": "List all ingested documents.",
        "category": "ingestion",
        "complexity": "simple",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "documents": {"type": "array"},
                "total": {"type": "integer"}
            },
            "required": ["documents", "total"]
        }
    },
    
    "get_document_chunks": {
        "name": "get_document_chunks",
        "description": "Get all chunks for a specific document.",
        "category": "ingestion",
        "complexity": "simple",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID"
                }
            },
            "required": ["doc_id"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "document": {"type": "object"},
                "chunks": {"type": "array"},
                "total": {"type": "integer"}
            },
            "required": ["chunks", "total"]
        }
    },
    
    "delete_document": {
        "name": "delete_document",
        "description": "Delete a document and all its chunks.",
        "category": "ingestion",
        "complexity": "simple",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID to delete"
                }
            },
            "required": ["doc_id"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "doc_id": {"type": "string"},
                "error": {"type": ["string", "null"]}
            },
            "required": ["success"]
        }
    },
    
    "get_ingestion_stats": {
        "name": "get_ingestion_stats",
        "description": "Get statistics about ingested documents and chunks.",
        "category": "ingestion",
        "complexity": "simple",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "document_count": {"type": "integer"},
                "chunk_count": {"type": "integer"},
                "total_characters": {"type": "integer"},
                "database_path": {"type": "string"}
            },
            "required": ["document_count", "chunk_count"]
        }
    },
    
    # Phase 3: Retrieval tools
    "index_document": {
        "name": "index_document",
        "description": "Index an ingested document for vector search. Run after ingest_document.",
        "category": "retrieval",
        "complexity": "moderate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID to index"
                }
            },
            "required": ["doc_id"]
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "doc_id": {"type": "string"},
                "chunks_indexed": {"type": "integer"},
                "error": {"type": ["string", "null"]}
            },
            "required": ["success"]
        }
    },
    
    "get_retrieval_stats": {
        "name": "get_retrieval_stats",
        "description": "Get statistics about the vector index.",
        "category": "retrieval",
        "complexity": "simple",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "embedder_model": {"type": "string"},
                "embedder_dimensions": {"type": "integer"},
                "total_vectors": {"type": "integer"},
                "index_path": {"type": "string"}
            },
            "required": ["total_vectors"]
        }
    }
}


def get_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """Get the schema for a specific tool."""
    return TOOL_SCHEMAS.get(tool_name)


def get_all_tool_names() -> list[str]:
    """Get list of all registered tool names."""
    return list(TOOL_SCHEMAS.keys())


def get_tool_input_schema(tool_name: str) -> dict[str, Any] | None:
    """Get just the input schema for a tool."""
    schema = TOOL_SCHEMAS.get(tool_name)
    return schema["inputSchema"] if schema else None
