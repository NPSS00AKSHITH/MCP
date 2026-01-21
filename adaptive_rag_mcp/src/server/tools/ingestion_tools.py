"""Real tool implementations for ingestion.

These replace the mock implementations for ingestion-related tools.
"""

from typing import Any
from pathlib import Path

from src.ingestion.pipeline import get_pipeline


def ingest_document(input_data: dict[str, Any]) -> dict[str, Any]:
    """Ingest a document from file path or raw text.
    
    Input:
        file_path: Path to file to ingest (optional)
        text: Raw text to ingest (optional, alternative to file_path)
        doc_id: Document ID for raw text (required if using text)
        file_name: Optional name for the document
        metadata: Optional metadata dict
        
    Output:
        doc_id, file_name, file_type, total_chunks, total_characters, success, error
    """
    pipeline = get_pipeline()
    
    file_path = input_data.get("file_path")
    text = input_data.get("text")
    
    if file_path:
        result = pipeline.ingest_file(file_path)
    elif text:
        doc_id = input_data.get("doc_id", "inline_doc")
        file_name = input_data.get("file_name", "inline_text")
        metadata = input_data.get("metadata", {})
        result = pipeline.ingest_text(text, doc_id, file_name, metadata)
    else:
        return {
            "success": False,
            "error": "Either file_path or text must be provided",
            "doc_id": "",
            "file_name": "",
            "file_type": "",
            "total_chunks": 0,
            "total_characters": 0,
        }
    
    return {
        "doc_id": result.doc_id,
        "file_name": result.file_name,
        "file_type": result.file_type,
        "total_chunks": result.total_chunks,
        "total_characters": result.total_characters,
        "success": result.success,
        "error": result.error,
    }


def list_documents(input_data: dict[str, Any]) -> dict[str, Any]:
    """List all ingested documents.
    
    Output:
        documents: List of document metadata
        total: Total document count
    """
    pipeline = get_pipeline()
    docs = pipeline.list_documents()
    
    return {
        "documents": docs,
        "total": len(docs),
    }


def get_document_chunks(input_data: dict[str, Any]) -> dict[str, Any]:
    """Get all chunks for a document.
    
    Input:
        doc_id: Document ID
        
    Output:
        chunks: List of chunk data
        total: Total chunk count
    """
    pipeline = get_pipeline()
    doc_id = input_data.get("doc_id", "")
    
    doc = pipeline.get_document(doc_id)
    if not doc:
        return {
            "chunks": [],
            "total": 0,
            "error": f"Document not found: {doc_id}",
        }
    
    chunks = pipeline.get_chunks(doc_id)
    
    return {
        "document": doc,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
        "total": len(chunks),
    }


def delete_document(input_data: dict[str, Any]) -> dict[str, Any]:
    """Delete a document and its chunks.
    
    Input:
        doc_id: Document ID to delete
        
    Output:
        success: Whether deletion succeeded
        doc_id: The deleted document ID
    """
    pipeline = get_pipeline()
    doc_id = input_data.get("doc_id", "")
    
    deleted = pipeline.delete_document(doc_id)
    
    return {
        "success": deleted,
        "doc_id": doc_id,
        "error": None if deleted else f"Document not found: {doc_id}",
    }


def get_ingestion_stats(input_data: dict[str, Any]) -> dict[str, Any]:
    """Get ingestion statistics.
    
    Output:
        document_count, chunk_count, total_characters, database_path
    """
    pipeline = get_pipeline()
    return pipeline.get_stats()


# Registry for ingestion tools
INGESTION_TOOLS: dict[str, callable] = {
    "ingest_document": ingest_document,
    "list_documents": list_documents,
    "get_document_chunks": get_document_chunks,
    "delete_document": delete_document,
    "get_ingestion_stats": get_ingestion_stats,
}
