#!/usr/bin/env python3
"""
Test Image Search - Semantic Image Search Functionality

=============================================================================
WHAT THIS FILE TESTS:
=============================================================================

This file tests the internal functions of the image_search tool:
1. Data loading (_load_all_data)
2. Semantic search (_semantic_search)
3. Machine name extraction (_extract_machine_names)
4. Content type detection (_detect_required_content_type)
5. Full search pipeline (_search_images)

=============================================================================
HOW WE GET THE DATA:
=============================================================================

- Images are indexed by vlm-yolo-detector in data/processed/
- image_index.json: metadata for each image (filename, PDF source, page)
- image_embeddings.npy: 384-dimensional vectors from VLM descriptions
- The search embeds the query and finds similar image descriptions

=============================================================================
EXPECTED RESULTS:
=============================================================================

- Queries with machine names should return images from that machine's manual
- Content type keywords (diagram, schematic) should influence results
- PDF filter should restrict results to specific manuals
- Results include image name, PDF source, page number, and relevance score

=============================================================================
HOW TO RUN:
=============================================================================

Run directly:
    uv run python tests/test_image_search.py

Run with pytest:
    uv run pytest tests/test_image_search.py -v

============================================================================="""

import sys
from pathlib import Path

# Add agentic-rag root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
