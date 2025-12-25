# RAG (Retrieval-Augmented Generation) module
# Provides local document search over manufacturing/machine manuals

from app.backend.core.rag.embeddings import OllamaEmbeddings
from app.backend.core.rag.document_loader import PDFLoader, DocumentChunk
from app.backend.core.rag.vector_store import FAISSVectorStore

__all__ = ["OllamaEmbeddings", "PDFLoader", "DocumentChunk", "FAISSVectorStore"]
