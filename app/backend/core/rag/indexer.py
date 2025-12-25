"""
Indexer script for building the RAG vector store from PDF documents.

Usage:
    uv run python -m app.backend.core.rag.indexer
    
    Or with custom paths:
    uv run python -m app.backend.core.rag.indexer --manuals-path ./data/manuals --index-path ./data/faiss_index
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent.parent


def build_index(
    manuals_path: Optional[str | Path] = None,
    index_path: Optional[str | Path] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    batch_size: int = 32,
    force_rebuild: bool = False,
) -> None:
    """
    Build FAISS index from PDF documents.

    Args:
        manuals_path: Path to directory containing PDF manuals
        index_path: Path to save FAISS index
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        batch_size: Number of documents to embed at once
        force_rebuild: Force rebuild even if index exists
    """
    from app.backend.core.rag.embeddings import OllamaEmbeddings
    from app.backend.core.rag.document_loader import PDFLoader
    from app.backend.core.rag.vector_store import FAISSVectorStore

    project_root = get_project_root()

    # Resolve paths
    manuals_path = Path(manuals_path or os.getenv(
        "MANUALS_PATH",
        project_root / "data" / "manuals"
    ))
    index_path = Path(index_path or os.getenv(
        "INDEX_PATH",
        project_root / "data" / "faiss_index"
    ))

    print(f"=" * 60)
    print(f"RAG Index Builder")
    print(f"=" * 60)
    print(f"Manuals path: {manuals_path}")
    print(f"Index path: {index_path}")
    print(f"=" * 60)

    # Check if index already exists
    if not force_rebuild and (index_path / "index.faiss").exists():
        print("Index already exists. Use --force to rebuild.")
        print("Loading existing index to verify...")
        embedder = OllamaEmbeddings()
        store = FAISSVectorStore(dimension=embedder.dimension, index_path=index_path)
        print(f"Existing index contains {store.size} vectors")
        return

    # Initialize components
    print("\nInitializing embedding model...")
    embedder = OllamaEmbeddings()
    print(f"Embedding model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.dimension}")

    print("\nInitializing document loader...")
    loader = PDFLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Chunk size: {loader.chunk_size}")
    print(f"Chunk overlap: {loader.chunk_overlap}")

    # Load documents
    print(f"\nLoading PDFs from {manuals_path}...")
    chunks = list(loader.load_directory(manuals_path))
    print(f"Loaded {len(chunks)} chunks from PDFs")

    if not chunks:
        print("No chunks to index. Exiting.")
        return

    # Show source document statistics
    sources = {}
    for chunk in chunks:
        sources[chunk.source] = sources.get(chunk.source, 0) + 1
    
    print(f"\nDocuments processed: {len(sources)}")
    for source, count in sorted(sources.items()):
        print(f"  - {source}: {count} chunks")

    # Generate embeddings
    print(f"\nGenerating embeddings (batch size: {batch_size})...")
    texts = [chunk.content for chunk in chunks]
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  Embedding batch {batch_num}/{total_batches}...", end="\r")
        batch_embeddings = embedder.embed_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    print(f"\n  Generated {len(all_embeddings)} embeddings")

    # Create and populate vector store
    print("\nBuilding FAISS index...")
    store = FAISSVectorStore(
        dimension=embedder.dimension,
        index_path=index_path,
        use_gpu=True,  # Will fallback to CPU if GPU unavailable
    )
    store.clear()  # Start fresh
    store.add(all_embeddings, chunks)
    
    # Save index
    print("\nSaving index...")
    store.save()

    print(f"\n{'=' * 60}")
    print(f"Index built successfully!")
    print(f"Total vectors: {store.size}")
    print(f"Index location: {index_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG index from PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build with defaults (reads from data/manuals, saves to data/faiss_index)
    uv run python -m app.backend.core.rag.indexer

    # Force rebuild existing index
    uv run python -m app.backend.core.rag.indexer --force

    # Custom paths
    uv run python -m app.backend.core.rag.indexer --manuals-path ./docs --index-path ./index
        """,
    )
    parser.add_argument(
        "--manuals-path",
        type=str,
        help="Path to directory containing PDF manuals",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        help="Path to save FAISS index",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Characters per chunk (default: 800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Overlap between chunks (default: 150)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists",
    )

    args = parser.parse_args()

    try:
        build_index(
            manuals_path=args.manuals_path,
            index_path=args.index_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            force_rebuild=args.force,
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
