"""
FAISS vector store for RAG.
Manages document embeddings and similarity search using FAISS.
Attempts GPU acceleration if available, falls back to CPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

from app.backend.core.rag.document_loader import DocumentChunk


class FAISSVectorStore:
    """
    FAISS vector store for document retrieval.
    Uses GPU if available, otherwise CPU.
    """

    def __init__(
        self,
        dimension: int,
        index_path: Optional[str | Path] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
            index_path: Optional path to save/load index. Defaults to INDEX_PATH env var.
            use_gpu: Whether to attempt GPU acceleration. Defaults to True.
        """
        self.dimension = dimension
        self.index_path = Path(index_path or os.getenv(
            "INDEX_PATH",
            Path(__file__).parent.parent.parent.parent.parent / "data" / "faiss_index"
        ))
        self.use_gpu = use_gpu and self._gpu_available()
        
        # FAISS index
        self._index: Optional[faiss.Index] = None
        # GPU resources (if using GPU)
        self._gpu_resources = None
        # Metadata storage (parallel to index vectors)
        self._chunks: List[DocumentChunk] = []
        
        # Initialize or load index
        if self._index_exists():
            self.load()
        else:
            self._create_index()

    def _gpu_available(self) -> bool:
        """Check if GPU is available for FAISS."""
        try:
            num_gpus = faiss.get_num_gpus()
            return num_gpus > 0
        except AttributeError:
            # faiss-cpu doesn't have get_num_gpus
            return False
        except Exception:
            return False

    def _create_index(self) -> None:
        """Create a new FAISS index."""
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        cpu_index = faiss.IndexFlatIP(self.dimension)
        
        if self.use_gpu:
            try:
                # Move index to GPU
                self._gpu_resources = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
                print("FAISS index using GPU acceleration")
            except (AttributeError, Exception) as e:
                print(f"GPU not available, using CPU: {e}")
                self._index = cpu_index
                self.use_gpu = False
        else:
            self._index = cpu_index
            print("FAISS index using CPU")

    def _index_exists(self) -> bool:
        """Check if index files exist."""
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "chunks.json"
        return index_file.exists() and metadata_file.exists()

    def add(self, embeddings: List[List[float]], chunks: List[DocumentChunk]) -> None:
        """
        Add embeddings and their associated chunks to the index.

        Args:
            embeddings: List of embedding vectors
            chunks: List of DocumentChunk objects (must match embeddings length)
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have same length")

        if not embeddings:
            return

        # Normalize embeddings for cosine similarity
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        # Add to index
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (DocumentChunk, score) tuples, sorted by relevance
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Normalize query embedding
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self._index.search(query_vector, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))

        return results

    def save(self) -> None:
        """Save index and metadata to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "chunks.json"

        # Convert GPU index to CPU for saving if needed
        if self.use_gpu:
            try:
                cpu_index = faiss.index_gpu_to_cpu(self._index)
                faiss.write_index(cpu_index, str(index_file))
            except (AttributeError, Exception):
                faiss.write_index(self._index, str(index_file))
        else:
            faiss.write_index(self._index, str(index_file))

        # Save chunk metadata
        chunks_data = [chunk.to_dict() for chunk in self._chunks]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        print(f"Index saved to {self.index_path}")

    def load(self) -> None:
        """Load index and metadata from disk."""
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "chunks.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Index files not found at {self.index_path}")

        # Load FAISS index
        cpu_index = faiss.read_index(str(index_file))
        
        if self.use_gpu:
            try:
                self._gpu_resources = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
                print("FAISS index loaded to GPU")
            except (AttributeError, Exception) as e:
                print(f"GPU not available, using CPU: {e}")
                self._index = cpu_index
                self.use_gpu = False
        else:
            self._index = cpu_index
            print("FAISS index loaded to CPU")

        # Load chunk metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        
        self._chunks = [DocumentChunk.from_dict(d) for d in chunks_data]
        print(f"Loaded {len(self._chunks)} chunks from index")

    def clear(self) -> None:
        """Clear the index."""
        self._create_index()
        self._chunks = []

    @property
    def size(self) -> int:
        """Return number of vectors in index."""
        return self._index.ntotal if self._index else 0
