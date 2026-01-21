"""Text chunking strategies with overlap.

Chunking Strategy Explanation:
=============================

This module implements a recursive character text splitter with overlap:

1. CHUNK SIZE: Target number of characters per chunk (default 1000)
   - Large enough to capture meaningful context
   - Small enough for efficient embedding and retrieval

2. CHUNK OVERLAP: Characters shared between adjacent chunks (default 200)
   - Prevents context loss at chunk boundaries
   - Important for semantic continuity

3. RECURSIVE SPLITTING: Tries multiple separators in order of preference:
   - Paragraph breaks (\\n\\n) - preserves natural document structure
   - Line breaks (\\n) - falls back to line-level splits
   - Sentences (. ) - maintains sentence integrity
   - Spaces - last resort for very long words/phrases

4. DETERMINISTIC CHUNK IDs: Generated from:
   - Document ID (stable across runs)
   - Chunk index (position in document)
   - Content hash (changes if content changes)
   
   This ensures:
   - Same document → same chunk IDs (for incremental updates)
   - Content changes → new chunk IDs (triggers re-embedding)
"""

import hashlib
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.content)


class RecursiveChunker:
    """Recursive text chunker with overlap.
    
    Splits text by trying larger separators first, falling back to smaller ones.
    Maintains overlap between chunks for context continuity.
    """
    
    # Separators in order of preference (try largest units first)
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentences
        ", ",    # Clauses
        " ",     # Words
        "",      # Characters (last resort)
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        """Initialize chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
            separators: List of separators to try, in order of preference.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
    
    def chunk_text(self, text: str, doc_id: str) -> list[Chunk]:
        """Split text into chunks with overlap.
        
        Args:
            text: The text to chunk.
            doc_id: Document ID for generating chunk IDs.
            
        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []
        
        # Get raw splits
        splits = self._recursive_split(text, self.separators)
        
        # Merge splits into chunks of appropriate size
        chunks = self._merge_splits(splits, doc_id)
        
        return chunks
    
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators in order."""
        if not separators:
            # Base case: return text as characters
            return list(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Empty separator means split into characters
            return list(text)
        
        # Split by current separator
        parts = text.split(separator)
        
        result = []
        for i, part in enumerate(parts):
            # If part is too large, split it further
            if len(part) > self.chunk_size and remaining_separators:
                result.extend(self._recursive_split(part, remaining_separators))
            else:
                result.append(part)
            
            # Re-add separator (except for last part)
            if i < len(parts) - 1 and separator != "":
                # Append separator to previous part instead of as separate item
                if result:
                    result[-1] += separator
        
        return [r for r in result if r]  # Filter empty strings
    
    def _merge_splits(self, splits: list[str], doc_id: str) -> list[Chunk]:
        """Merge small splits into chunks, maintaining overlap."""
        chunks = []
        current_chunk = ""
        current_start = 0
        char_position = 0
        
        for split in splits:
            # If adding this split would exceed chunk_size
            if len(current_chunk) + len(split) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk = self._create_chunk(
                        content=current_chunk.strip(),
                        doc_id=doc_id,
                        chunk_index=len(chunks),
                        start_char=current_start,
                        end_char=char_position,
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + split
                    current_start = char_position - (len(current_chunk) - len(split))
                else:
                    # Split is larger than chunk_size, take it as-is
                    current_chunk = split
            else:
                current_chunk += split
            
            char_position += len(split)
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                content=current_chunk.strip(),
                doc_id=doc_id,
                chunk_index=len(chunks),
                start_char=current_start,
                end_char=char_position,
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        doc_id: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
    ) -> Chunk:
        """Create a chunk with deterministic ID."""
        # Deterministic chunk ID based on:
        # - doc_id (stable for same file)
        # - chunk_index (position in document)
        # - content hash (changes if content changes)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        chunk_id = f"{doc_id}_{chunk_index:04d}_{content_hash}"
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=content,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata={
                "char_count": len(content),
                "word_count": len(content.split()),
            },
        )


class PageAwareChunker(RecursiveChunker):
    """Chunker that respects page boundaries.
    
    Adds page metadata to chunks based on their position.
    """
    
    def chunk_document(
        self,
        pages: list[tuple[str, int]],  # (content, page_number)
        doc_id: str,
    ) -> list[Chunk]:
        """Chunk a multi-page document, tracking page info.
        
        Args:
            pages: List of (page_content, page_number) tuples.
            doc_id: Document ID.
            
        Returns:
            List of chunks with page metadata.
        """
        all_chunks = []
        global_index = 0
        
        for page_content, page_number in pages:
            if not page_content.strip():
                continue
            
            # Chunk this page
            page_chunks = self.chunk_text(page_content, doc_id)
            
            # Update chunk indices and add page metadata
            for chunk in page_chunks:
                chunk.chunk_index = global_index
                chunk.metadata["page"] = page_number
                # Regenerate ID with global index
                content_hash = hashlib.md5(chunk.content.encode()).hexdigest()[:8]
                chunk.chunk_id = f"{doc_id}_{global_index:04d}_{content_hash}"
                all_chunks.append(chunk)
                global_index += 1
        
        return all_chunks


# Default chunker instance
default_chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)


def chunk_text(text: str, doc_id: str) -> list[Chunk]:
    """Convenience function to chunk text with default settings."""
    return default_chunker.chunk_text(text, doc_id)
