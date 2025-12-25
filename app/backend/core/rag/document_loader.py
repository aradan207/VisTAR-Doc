"""
Document loader for PDF files.
Handles loading, parsing, and chunking of PDF documents for RAG indexing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Generator

from pypdf import PdfReader


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata."""
    
    content: str
    source: str  # Filename
    page: int  # Page number (1-indexed)
    chunk_index: int  # Index of chunk within the page
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentChunk":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            source=data["source"],
            page=data["page"],
            chunk_index=data["chunk_index"],
            metadata=data.get("metadata", {}),
        )


class PDFLoader:
    """
    Loads and chunks PDF documents for RAG indexing.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Initialize PDF loader.

        Args:
            chunk_size: Maximum characters per chunk. Defaults to CHUNK_SIZE env var or 800.
            chunk_overlap: Overlap between chunks. Defaults to CHUNK_OVERLAP env var or 150.
        """
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "150"))

    def load_pdf(self, file_path: str | Path) -> List[DocumentChunk]:
        """
        Load and chunk a single PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            List of DocumentChunk objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        chunks = []
        filename = file_path.name

        try:
            reader = PdfReader(str(file_path))
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if not text or not text.strip():
                    continue

                # Clean up the text
                text = self._clean_text(text)
                
                # Chunk the page text
                page_chunks = self._chunk_text(text, filename, page_num)
                chunks.extend(page_chunks)

        except Exception as e:
            raise RuntimeError(f"Failed to load PDF {file_path}: {e}") from e

        return chunks

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
    ) -> Generator[DocumentChunk, None, None]:
        """
        Load all PDF files from a directory.

        Args:
            directory: Path to directory containing PDFs
            recursive: Whether to search subdirectories

        Yields:
            DocumentChunk objects
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(directory.glob(pattern))

        if not pdf_files:
            raise ValueError(f"No PDF files found in {directory}")

        for pdf_path in pdf_files:
            try:
                chunks = self.load_pdf(pdf_path)
                for chunk in chunks:
                    yield chunk
            except Exception as e:
                print(f"Warning: Failed to load {pdf_path}: {e}")
                continue

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Replace multiple whitespace with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable characters except newlines
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        return text.strip()

    def _chunk_text(
        self,
        text: str,
        source: str,
        page: int,
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            source: Source filename
            page: Page number

        Returns:
            List of DocumentChunk objects
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Get chunk end position
            end = start + self.chunk_size

            # If not at the end, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary (.!?) within last 20% of chunk
                search_start = start + int(self.chunk_size * 0.8)
                best_break = end
                
                for i in range(end, search_start, -1):
                    if i < len(text) and text[i-1] in '.!?\n':
                        best_break = i
                        break
                
                # If no sentence break found, look for word boundary
                if best_break == end:
                    for i in range(end, search_start, -1):
                        if i < len(text) and text[i] in ' \n\t':
                            best_break = i + 1
                            break
                
                end = best_break

            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source=source,
                    page=page,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

            # Move start position with overlap
            start = end - self.chunk_overlap if end - self.chunk_overlap > start else end

            # Prevent infinite loop
            if start >= len(text):
                break

        return chunks
