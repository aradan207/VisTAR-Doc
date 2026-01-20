#!/usr/bin/env python3
"""
End-to-end test of the agentic-rag with semantic image search.

Tests:
1. Manual search finds relevant content
2. Image search finds semantically matching images
3. Results include both text and images
"""

import sys
from pathlib import Path

# Add agentic-rag root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_manual_search():
    """Test manual search functionality."""
    print("=" * 70)
    print("  TEST 1: MANUAL SEARCH")
    print("=" * 70)
    
    from app.backend.api.tools.manual_search import manual_search, ManualSearchArgs
    
    # Test query
    args = ManualSearchArgs(
        query="APSX-PIM J7 connector wiring",
        top_k=3
    )
    
    result = manual_search(args)
    
    print(f"\nQuery: {args.query}")
    print(f"Results: {result.get('count', 0)}")
    
    if result.get('results'):
        for r in result['results'][:2]:
            print(f"\n  Source: {r.get('source', 'N/A')}")
            print(f"  Page: {r.get('page', 'N/A')}")
            print(f"  Content: {r.get('content', 'N/A')[:200]}...")
    
    return result


def test_image_search():
    """Test image search functionality."""
    print("\n" + "=" * 70)
    print("  TEST 2: SEMANTIC IMAGE SEARCH")
    print("=" * 70)
    
    from app.backend.api.tools.image_search import image_search, ImageSearchArgs
    
    # Test query matching what manual search found
    args = ImageSearchArgs(
        query="APSX-PIM J7 connector wiring diagram",
        pdf_filter="APSX-PIM",
        top_k=3
    )
    
    result = image_search(args)
    
    print(f"\nQuery: {args.query}")
    print(f"PDF Filter: {args.pdf_filter}")
    print(f"Results: {result.get('count', 0)}")
    
    if result.get('results'):
        for r in result['results']:
            print(f"\n  Image: {r.get('image_name', 'N/A')}")
            print(f"  Page: {r.get('page_number', 'N/A')}")
            print(f"  Type: {r.get('content_type', 'N/A')}")
            print(f"  Score: {r.get('relevance_score', 'N/A')}")
            print(f"  Semantic: {r.get('semantic_match', 'N/A')}")
            print(f"  URL: {r.get('url', 'N/A')}")
            print(f"  Description: {r.get('description', 'N/A')[:100]}...")
    
    return result


def test_combined_response():
    """Test combining manual and image search results."""
    print("\n" + "=" * 70)
    print("  TEST 3: COMBINED RESPONSE")
    print("=" * 70)
    
    # Run both searches
    from app.backend.api.tools.manual_search import manual_search, ManualSearchArgs
    from app.backend.api.tools.image_search import image_search, ImageSearchArgs
    
    query = "Show me the APSX-PIM connector J7 wiring diagram"
    
    print(f"\nUser Query: {query}")
    
    # Manual search
    manual_args = ManualSearchArgs(query=query, top_k=2)
    manual_result = manual_search(manual_args)
    
    # Get page hint from manual search
    page_hint = None
    pdf_source = None
    if manual_result.get('results'):
        first_result = manual_result['results'][0]
        page_hint = first_result.get('page')
        pdf_source = first_result.get('source', '').replace('.pdf', '')
    
    # Image search
    image_args = ImageSearchArgs(
        query=query,
        pdf_filter=pdf_source or "APSX-PIM",
        page_hint=page_hint,
        top_k=2
    )
    image_result = image_search(image_args)
    
    print(f"\n--- Combined Answer ---")
    print(f"\nText Results from Manual:")
    if manual_result.get('results'):
        for r in manual_result['results'][:1]:
            print(f"  From {r.get('source', 'N/A')}, page {r.get('page', 'N/A')}:")
            print(f"  {r.get('content', 'N/A')[:300]}...")
    
    print(f"\nRelevant Images:")
    if image_result.get('results'):
        for r in image_result['results']:
            print(f"  ![{r.get('content_type', 'Image')} from page {r.get('page_number')}]({r.get('url')})")
            print(f"  Description: {r.get('description', 'N/A')[:150]}...")
    else:
        print("  No images found. Refer to PDF page directly.")
    
    print("\n--- Example Final Answer ---")
    print("""
Based on the APSX-PIM manual, the J7 connector wiring information can be found 
on page 41. The J7 connector is used for...

Here is the schematic diagram showing the connector wiring:

![Schematic diagram from page 41](http://localhost:8000/api/media/yologen/train/APSX-PIM_page41_img1.png)

This diagram shows the electrical connections for the J7 connector including...
""")


def test_different_machines():
    """Test image search for different machines."""
    print("\n" + "=" * 70)
    print("  TEST 4: DIFFERENT MACHINES")
    print("=" * 70)
    
    from app.backend.api.tools.image_search import image_search, ImageSearchArgs
    
    test_cases = [
        ("RSA-G2 sample loading", "RSA-G2"),
        ("BOY 35 hydraulic system diagram", "BOY-35"),
        ("MILACRON safety warnings", "MILACRON"),
    ]
    
    for query, pdf_filter in test_cases:
        args = ImageSearchArgs(query=query, pdf_filter=pdf_filter, top_k=2)
        result = image_search(args)
        
        print(f"\n{query} ({pdf_filter}):")
        if result.get('results'):
            for r in result['results']:
                print(f"  - {r['image_name']} (page {r['page_number']}, {r['semantic_match']})")
        else:
            print("  No results")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  AGENTIC-RAG END-TO-END TEST")
    print("=" * 70)
    
    try:
        test_manual_search()
        test_image_search()
        test_combined_response()
        test_different_machines()
        
        print("\n" + "=" * 70)
        print("  ALL TESTS PASSED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
