"""
Quick test script for the RAG system.
Run with: uv run python test_rag.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
