"""SQLite storage for document chunks.

Stores chunks with metadata for later retrieval.
No embeddings stored here - that's Phase 3.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

from src.ingestion.chunker import Chunk


class ChunkStore:
    """SQLite-based storage for document chunks.
    
    Schema:
    - documents: doc_id, source_path, file_name, file_type, metadata, created_at
    - chunks: chunk_id, doc_id, content, chunk_index, start_char, end_char, metadata
    """
    
    def __init__(self, db_path: str | Path = "./data/chunks.db"):
        """Initialize chunk store.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(doc_id, chunk_index);
            """)
    
    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection as context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def store_document(
        self,
        doc_id: str,
        source_path: str,
        file_name: str,
        file_type: str,
        chunks: list[Chunk],
        metadata: dict | None = None,
    ) -> int:
        """Store a document and its chunks.
        
        Args:
            doc_id: Unique document identifier.
            source_path: Original file path.
            file_name: File name.
            file_type: File type (pdf, markdown, text).
            chunks: List of chunks to store.
            metadata: Optional document metadata.
            
        Returns:
            Number of chunks stored.
        """
        now = datetime.utcnow().isoformat()
        metadata = metadata or {}
        
        with self._get_connection() as conn:
            # Delete existing document and chunks (cascade)
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            
            # Insert document
            conn.execute("""
                INSERT INTO documents (doc_id, source_path, file_name, file_type, 
                                       total_chunks, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id, source_path, file_name, file_type,
                len(chunks), json.dumps(metadata), now, now
            ))
            
            # Insert chunks
            for chunk in chunks:
                conn.execute("""
                    INSERT INTO chunks (chunk_id, doc_id, content, chunk_index,
                                        start_char, end_char, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id, chunk.doc_id, chunk.content, chunk.chunk_index,
                    chunk.start_char, chunk.end_char, json.dumps(chunk.metadata), now
                ))
        
        return len(chunks)
    
    def get_document(self, doc_id: str) -> dict | None:
        """Get document metadata by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            return {
                "doc_id": row["doc_id"],
                "source_path": row["source_path"],
                "file_name": row["file_name"],
                "file_type": row["file_type"],
                "total_chunks": row["total_chunks"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
    
    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
                (doc_id,)
            ).fetchall()
            
            return [
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    content=row["content"],
                    chunk_index=row["chunk_index"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    metadata=json.loads(row["metadata"]),
                )
                for row in rows
            ]
    
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a single chunk by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            return Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                content=row["content"],
                chunk_index=row["chunk_index"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                metadata=json.loads(row["metadata"]),
            )
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks.
        
        Returns:
            True if document was deleted, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
            )
            return cursor.rowcount > 0
    
    def list_documents(self) -> list[dict]:
        """List all documents."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
            
            return [
                {
                    "doc_id": row["doc_id"],
                    "source_path": row["source_path"],
                    "file_name": row["file_name"],
                    "file_type": row["file_type"],
                    "total_chunks": row["total_chunks"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
    
    def get_stats(self) -> dict:
        """Get storage statistics."""
        with self._get_connection() as conn:
            doc_count = conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            
            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]
            
            total_chars = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM chunks"
            ).fetchone()[0]
            
            return {
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "total_characters": total_chars,
                "database_path": str(self.db_path),
            }
    
    def search_chunks(self, query: str, limit: int = 10) -> list[Chunk]:
        """Simple text search in chunks (for testing, not production use)."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM chunks 
                WHERE content LIKE ? 
                ORDER BY chunk_index 
                LIMIT ?
            """, (f"%{query}%", limit)).fetchall()
            
            return [
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    content=row["content"],
                    chunk_index=row["chunk_index"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    metadata=json.loads(row["metadata"]),
                )
                for row in rows
            ]
