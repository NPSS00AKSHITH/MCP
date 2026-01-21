# Sample Document for Testing

This is a sample Markdown document for testing the ingestion pipeline.

## Introduction

The Adaptive RAG (Retrieval-Augmented Generation) MCP Server is designed to provide intelligent document retrieval capabilities. It uses multiple strategies to optimize the retrieval process based on query complexity.

## Key Features

### Document Ingestion
The system supports multiple document formats including PDF, Markdown, and plain text files. Documents are automatically chunked with overlap to maintain context continuity.

### Chunking Strategy
Text is split using a recursive approach:
1. First, try to split by paragraph breaks
2. Then by line breaks if paragraphs are too long
3. Finally by sentences or words as needed

Each chunk maintains overlap with adjacent chunks to preserve semantic context at boundaries.

### Query Routing
Queries are analyzed for complexity and routed to appropriate retrieval strategies:
- **Simple queries**: Direct LLM response without retrieval
- **Moderate queries**: Single-step vector search
- **Complex queries**: Multi-step iterative retrieval

## Architecture

The system follows a modular architecture with clear separation of concerns:

1. **Ingestion Layer**: Handles document loading and chunking
2. **Storage Layer**: Manages chunk persistence with SQLite
3. **Retrieval Layer**: Implements vector search (future phase)
4. **Generation Layer**: Produces final responses (future phase)

## Conclusion

This modular design allows for extensibility and maintainability as the system evolves.
