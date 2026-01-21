"""Document loaders for PDF and Markdown files.

Each loader extracts text and metadata from documents.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader


@dataclass
class DocumentPage:
    """Represents a single page/section from a document."""
    
    content: str
    page_number: int
    metadata: dict = field(default_factory=dict)


@dataclass
class LoadedDocument:
    """Represents a fully loaded document with all pages."""
    
    doc_id: str
    source_path: str
    file_name: str
    file_type: str
    pages: list[DocumentPage]
    metadata: dict = field(default_factory=dict)
    
    @property
    def total_pages(self) -> int:
        return len(self.pages)
    
    @property
    def full_text(self) -> str:
        """Concatenate all pages into single text."""
        return "\n\n".join(page.content for page in self.pages)


class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: Path) -> LoadedDocument:
        """Load a document from the given path."""
        pass
    
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        pass
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate a deterministic document ID from file path and content hash."""
        # Use file path + modification time for stability
        stat = file_path.stat()
        id_string = f"{file_path.absolute()}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]


class PDFLoader(BaseLoader):
    """Loader for PDF documents using pypdf."""
    
    def supported_extensions(self) -> list[str]:
        return [".pdf"]
    
    def load(self, file_path: Path) -> LoadedDocument:
        """Load a PDF document, extracting text from each page."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        reader = PdfReader(file_path)
        pages = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(DocumentPage(
                content=text.strip(),
                page_number=page_num,
                metadata={"page": page_num}
            ))
        
        # Extract PDF metadata
        pdf_metadata = {}
        if reader.metadata:
            pdf_metadata = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
            }
            # Filter out empty values
            pdf_metadata = {k: v for k, v in pdf_metadata.items() if v}
        
        return LoadedDocument(
            doc_id=self._generate_doc_id(file_path),
            source_path=str(file_path.absolute()),
            file_name=file_path.name,
            file_type="pdf",
            pages=pages,
            metadata=pdf_metadata,
        )


class MarkdownLoader(BaseLoader):
    """Loader for Markdown documents.
    
    Splits by headings to create logical sections.
    """
    
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]
    
    def load(self, file_path: Path) -> LoadedDocument:
        """Load a Markdown document, splitting by headings."""
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        
        content = file_path.read_text(encoding="utf-8")
        
        # Split by headings (# or ##)
        sections = self._split_by_headings(content)
        
        pages = []
        for section_num, section in enumerate(sections, start=1):
            pages.append(DocumentPage(
                content=section["content"].strip(),
                page_number=section_num,
                metadata={
                    "section": section_num,
                    "heading": section.get("heading", ""),
                }
            ))
        
        return LoadedDocument(
            doc_id=self._generate_doc_id(file_path),
            source_path=str(file_path.absolute()),
            file_name=file_path.name,
            file_type="markdown",
            pages=pages,
            metadata={"format": "markdown"},
        )
    
    def _split_by_headings(self, content: str) -> list[dict]:
        """Split markdown content by headings."""
        import re
        
        # Pattern to match markdown headings
        heading_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
        
        sections = []
        last_end = 0
        current_heading = ""
        
        for match in heading_pattern.finditer(content):
            # Save previous section if it has content
            if last_end < match.start():
                section_content = content[last_end:match.start()].strip()
                if section_content or sections:  # Skip empty first section
                    sections.append({
                        "heading": current_heading,
                        "content": section_content,
                    })
            
            current_heading = match.group(2)
            last_end = match.end()
        
        # Add the final section
        final_content = content[last_end:].strip()
        if final_content:
            sections.append({
                "heading": current_heading,
                "content": final_content,
            })
        
        # If no headings found, treat entire content as one section
        if not sections:
            sections.append({
                "heading": "",
                "content": content.strip(),
            })
        
        return sections


class TextLoader(BaseLoader):
    """Loader for plain text files."""
    
    def supported_extensions(self) -> list[str]:
        return [".txt", ".text"]
    
    def load(self, file_path: Path) -> LoadedDocument:
        """Load a plain text file as a single page."""
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        content = file_path.read_text(encoding="utf-8")
        
        pages = [DocumentPage(
            content=content.strip(),
            page_number=1,
            metadata={}
        )]
        
        return LoadedDocument(
            doc_id=self._generate_doc_id(file_path),
            source_path=str(file_path.absolute()),
            file_name=file_path.name,
            file_type="text",
            pages=pages,
            metadata={"format": "plain_text"},
        )


# Loader registry
LOADERS: dict[str, BaseLoader] = {
    ".pdf": PDFLoader(),
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".txt": TextLoader(),
    ".text": TextLoader(),
}


def get_loader(file_path: Path) -> BaseLoader:
    """Get the appropriate loader for a file based on extension."""
    ext = file_path.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS.keys())}")
    return loader


def load_document(file_path: str | Path) -> LoadedDocument:
    """Load a document using the appropriate loader."""
    path = Path(file_path)
    loader = get_loader(path)
    return loader.load(path)
