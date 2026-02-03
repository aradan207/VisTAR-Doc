#!/usr/bin/env python3
"""
Test RAG System - Basic Manual Search Functionality

=============================================================================
WHAT THIS FILE TESTS:
=============================================================================

This is a simple smoke test for the manual_search tool. It runs a few
queries and prints the results to the console for manual inspection.

We test:
1. Can the tool connect to the FAISS vector store?
2. Does it return results for common equipment queries?
3. Are the results formatted correctly (source, page, content, score)?

=============================================================================
HOW WE GET THE DATA:
=============================================================================

- The manual_search tool uses a FAISS index stored in data/faiss_index/
- The index contains text chunks from PDF manuals
- Each chunk is embedded using sentence transformers
- Queries are also embedded and compared using cosine similarity

=============================================================================
EXPECTED RESULTS:
=============================================================================

- Each query should return 1-3 results
- Results should include source PDF name and page number
- Content should be relevant to the query
- Higher relevance scores mean better matches

=============================================================================
HOW TO RUN:
=============================================================================

Run directly:
    uv run python tests/test_rag.py

Run with pytest:
    uv run pytest tests/test_rag.py -v

=============================================================================
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from app.backend.api.tools.manual_search import manual_search, ManualSearchArgs


def test_search(query: str, top_k: int = 3):
    """Test the manual_search tool."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    args = ManualSearchArgs(query=query, top_k=top_k)
    result = manual_search(args)
    
    if "error" in result and result["error"]:
        print(f"Error: {result['error']}")
        return
    
    print(f"Total documents indexed: {result['total_documents_indexed']}")
    print(f"Results found: {result['results_count']}")
    print()
    
    for i, r in enumerate(result["results"], 1):
        print(f"Result {i}:")
        print(f"  Source: {r['source']} (Page {r['page']})")
        print(f"  Score: {r['relevance_score']}")
        print(f"  Content: {r['content'][:200]}...")
        print()


if __name__ == "__main__":
    # Test queries related to manufacturing equipment
    test_queries = [
        "How to calibrate a DSC instrument?",
        "What is the temperature range for thermal analysis?",
        "injection molding machine specifications",
        "maintenance procedures for rheometer",
        "DMA test procedures",
    ]
    
    for query in test_queries:
        test_search(query)
