"""Document ingestion pipeline.

Orchestrates: load → chunk → store
"""

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.loaders import load_document, LoadedDocument
from src.ingestion.chunker import RecursiveChunker, PageAwareChunker, Chunk
from src.ingestion.storage import ChunkStore
from src.server.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    
    doc_id: str
    file_name: str
    file_type: str
    total_chunks: int
    total_characters: int
    success: bool
    error: str | None = None


class IngestionPipeline:
    """Document ingestion pipeline: load → chunk → store."""
    
    def __init__(
        self,
        chunk_store: ChunkStore | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize pipeline.
        
        Args:
            chunk_store: Storage for chunks. Uses default if None.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks.
        """
        self.store = chunk_store or ChunkStore()
        self.chunker = PageAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def ingest_file(self, file_path: str | Path) -> IngestionResult:
        """Ingest a single file.
        
        Args:
            file_path: Path to the file to ingest.
            
        Returns:
            IngestionResult with details about the ingestion.
        """
        path = Path(file_path)
        
        try:
            # Step 1: Load document
            logger.info("loading_document", file_path=str(path))
            doc = load_document(path)
            
            # Step 2: Chunk document
            logger.info("chunking_document", doc_id=doc.doc_id, pages=len(doc.pages))
            pages = [(page.content, page.page_number) for page in doc.pages]
            chunks = self.chunker.chunk_document(pages, doc.doc_id)
            
            # Step 3: Store chunks
            logger.info("storing_chunks", doc_id=doc.doc_id, chunk_count=len(chunks))
            stored_count = self.store.store_document(
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                file_name=doc.file_name,
                file_type=doc.file_type,
                chunks=chunks,
                metadata=doc.metadata,
            )
            
            total_chars = sum(len(chunk.content) for chunk in chunks)
            
            logger.info(
                "ingestion_complete",
                doc_id=doc.doc_id,
                chunks=stored_count,
                characters=total_chars,
            )
            
            return IngestionResult(
                doc_id=doc.doc_id,
                file_name=doc.file_name,
                file_type=doc.file_type,
                total_chunks=stored_count,
                total_characters=total_chars,
                success=True,
            )
            
        except Exception as e:
            logger.error("ingestion_failed", error=str(e), file_path=str(path))
            return IngestionResult(
                doc_id="",
                file_name=path.name if path.exists() else str(path),
                file_type="unknown",
                total_chunks=0,
                total_characters=0,
                success=False,
                error=str(e),
            )
    
    def ingest_text(
        self,
        text: str,
        doc_id: str,
        file_name: str = "inline_text",
        metadata: dict | None = None,
    ) -> IngestionResult:
        """Ingest raw text directly.
        
        Args:
            text: Text content to ingest.
            doc_id: ID to assign to this document.
            file_name: Name for the document.
            metadata: Optional metadata.
            
        Returns:
            IngestionResult with details about the ingestion.
        """
        try:
            # Chunk text
            chunks = self.chunker.chunk_text(text, doc_id)
            
            # Store
            stored_count = self.store.store_document(
                doc_id=doc_id,
                source_path="inline",
                file_name=file_name,
                file_type="text",
                chunks=chunks,
                metadata=metadata or {},
            )
            
            total_chars = sum(len(chunk.content) for chunk in chunks)
            
            return IngestionResult(
                doc_id=doc_id,
                file_name=file_name,
                file_type="text",
                total_chunks=stored_count,
                total_characters=total_chars,
                success=True,
            )
            
        except Exception as e:
            return IngestionResult(
                doc_id=doc_id,
                file_name=file_name,
                file_type="text",
                total_chunks=0,
                total_characters=0,
                success=False,
                error=str(e),
            )
    
    def get_document(self, doc_id: str) -> dict | None:
        """Get document metadata."""
        return self.store.get_document(doc_id)
    
    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        return self.store.get_chunks(doc_id)
    
    def list_documents(self) -> list[dict]:
        """List all ingested documents."""
        return self.store.list_documents()
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        return self.store.delete_document(doc_id)
    
    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        return self.store.get_stats()


# Default pipeline instance
_default_pipeline: IngestionPipeline | None = None


def get_pipeline() -> IngestionPipeline:
    """Get the default pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = IngestionPipeline()
    return _default_pipeline
