#!/usr/bin/env python3
"""
Test the semantic image search functionality.

Tests the updated image_search tool with VLM descriptions and embeddings.
"""

import sys
from pathlib import Path

# Add agentic-rag to path
sys.path.insert(0, str(Path(__file__).parent))

from app.backend.api.tools.image_search import (
    _load_all_data,
    _search_images,
    _semantic_search,
    _extract_machine_names,
    _detect_required_content_type
)


def test_semantic_search():
    """Test semantic search functionality."""
    print("=" * 60)
    print("  TESTING SEMANTIC IMAGE SEARCH")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    _load_all_data()
    
    # Test queries
    test_queries = [
        ("APSX-PIM J7 connector wiring diagram", "APSX-PIM"),
        ("RSA-G2 rheometer sample loading procedure", "RSA-G2"),
        ("BOY 35 injection molding hydraulic system", "BOY-35"),
        ("MILACRON safety warnings", "MILACRON"),
        ("thermal analysis temperature control diagram", None),
    ]
    
    for query, pdf_filter in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"PDF Filter: {pdf_filter}")
        print(f"Machine names extracted: {_extract_machine_names(query)}")
        print(f"Required content type: {_detect_required_content_type(query)}")
        print("-" * 60)
        
        results = _search_images(
            query=query,
            pdf_filter=pdf_filter,
            top_k=5
        )
        
        if results:
            print(f"Found {len(results)} results:")
            for i, r in enumerate(results[:5], 1):
                print(f"\n  {i}. {r['image_name']}")
                print(f"     PDF: {r['pdf_source']}, Page: {r['page_number']}")
                print(f"     Type: {r['content_type']}")
                print(f"     Score: {r['score']}, Semantic: {r['semantic_score']}%")
                desc = r['vlm_description']
                print(f"     Description: {desc[:100]}..." if len(desc) > 100 else f"     Description: {desc}")
        else:
            print("No results found!")
    
    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)


def test_image_search_tool():
    """Test the actual tool function."""
    from app.backend.api.tools.image_search import image_search, ImageSearchArgs
    
    print("\n" + "=" * 60)
    print("  TESTING IMAGE SEARCH TOOL")
    print("=" * 60)
    
    # Test with APSX-PIM query
    args = ImageSearchArgs(
        query="APSX-PIM connector J7 wiring",
        pdf_filter="APSX-PIM",
        top_k=3
    )
    
    result = image_search(args)
    
    print(f"\nQuery: {args.query}")
    print(f"PDF Filter: {args.pdf_filter}")
    print(f"Results count: {result.get('count', 0)}")
    
    if result.get('results'):
        print("\nResults:")
        for r in result['results']:
            print(f"  - {r['image_name']}")
            print(f"    URL: {r['url']}")
            print(f"    Type: {r['content_type']}")
            print(f"    Score: {r['relevance_score']}, Semantic: {r['semantic_match']}")
            print(f"    Description: {r['description'][:80]}...")
    else:
        print(f"Message: {result.get('message', 'No message')}")


if __name__ == "__main__":
    test_semantic_search()
    test_image_search_tool()
