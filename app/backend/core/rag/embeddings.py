"""
Ollama-based embeddings wrapper for RAG.
Uses local Ollama server for generating text embeddings.
"""

from __future__ import annotations

import os
from typing import List, Optional

import ollama


class OllamaEmbeddings:
    """
    Wrapper for Ollama embeddings API.
    Uses mxbai-embed-large by default but can be configured via environment.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
    ):
        """
        Initialize Ollama embeddings.

        Args:
            model_name: Ollama embedding model name. Defaults to OLLAMA_EMBED_MODEL env var
                        or 'hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M'
            host: Ollama server URL. Defaults to OLLAMA_HOST env var or 'http://localhost:11434'
        """
        self.model_name = model_name or os.getenv(
            "OLLAMA_EMBED_MODEL",
            "hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M"
        )
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Set host for ollama library
        if self.host:
            os.environ["OLLAMA_HOST"] = self.host

        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension. Computed lazily on first call."""
        if self._dimension is None:
            # Generate a test embedding to determine dimension
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = ollama.embed(model=self.model_name, input=text)
            return response["embeddings"][0]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to embed per API call

        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = ollama.embed(model=self.model_name, input=batch)
                all_embeddings.extend(response["embeddings"])
            except Exception as e:
                raise RuntimeError(f"Failed to generate batch embeddings: {e}") from e
        
        return all_embeddings
