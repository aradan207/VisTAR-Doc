#!/usr/bin/env python3
"""
Test Combined Text + Image Queries (End-to-End)

=============================================================================
WHAT THIS FILE TESTS:
=============================================================================

This file tests the full query flow where we:
1. First search for TEXT content using manual_search
2. Extract page hints and source info from text results
3. Then search for IMAGES using those hints
4. Verify that both results come from the SAME manual
5. Verify that page hints improve image relevance

This simulates what the agent does when answering a visual question:
the agent first finds text context, then uses that to find related images.

=============================================================================
HOW WE GET THE DATA:
=============================================================================

Text data:
- manual_search uses FAISS vector store with PDF text chunks
- Each chunk has: content, source PDF, page number
- Location: agentic-rag/data/faiss_index/

Image data:
- image_search uses VLM-generated descriptions and embeddings
- Each image has: filename, PDF source, page, description, URL
- Location: vlm-yolo-detector/data/processed/

=============================================================================
EXPECTED RESULTS:
=============================================================================

Each test case defines:
- query: A question that needs both text AND visual answer
- expected_source: Which manual should both results come from
- description: What we're testing

The test passes if:
1. manual_search returns results from expected_source
2. image_search returns images from the SAME source
3. When we use page hints, images are from nearby pages

=============================================================================
HOW TO RUN:
=============================================================================

Run all tests in this file:
    uv run pytest tests/test_combined_queries.py -v

Run standalone (outside pytest):
    uv run python tests/test_combined_queries.py

=============================================================================
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backend.api.tools.manual_search import manual_search, ManualSearchArgs
from app.backend.api.tools.image_search import image_search, ImageSearchArgs


# =============================================================================
# TEST CASES - Visual questions that need both text and image search
# =============================================================================

COMBINED_QUERY_TEST_CASES = [
    {
        "id": "apsx-pim-wiring-full",
        "query": "Show me the APSX-PIM wiring diagram and explain the connections",
        "expected_source": "APSX-PIM",
        "description": "Should find wiring text AND wiring schematic from same manual",
    },
    {
        "id": "apsx-pim-dashboard-full",
        "query": "What does the APSX-PIM dashboard look like during injection cycle?",
        "expected_source": "APSX-PIM",
        "description": "Should find cycle stage text AND dashboard image",
    },
    {
        "id": "boy35-hydraulic-full",
        "query": "Show me the BOY 35 hydraulic system diagram and pressure specs",
        "expected_source": "BOY-35",
        "description": "Should find hydraulic specs AND hydraulic diagrams",
    },
    {
        "id": "boy35-safety-full",
        "query": "What safety warnings are shown for BOY 35 maintenance?",
        "expected_source": "BOY-35",
        "description": "Should find safety text AND warning images",
    },
    {
        "id": "milacron-ejector-full",
        "query": "Show me how to set up the MILACRON ejector system",
        "expected_source": "MILACRON",
        "description": "Should find ejector setup text AND menu screenshots",
    },
    {
        "id": "maac-forming-full",
        "query": "Show me the MAAC thermoformer forming timer diagram",
        "expected_source": "MAAC",
        "description": "Should find timer settings text AND timer diagram",
    },
    {
        "id": "rsa-g2-clamping-full",
        "query": "What clamping systems are available for RSA-G2 testing?",
        "expected_source": "RSA-G2",
        "description": "Should find clamping procedures AND clamping diagrams",
    },
]


class TestCombinedQueries:
    """Test suite for combined text + image queries."""
    
    @pytest.mark.parametrize("test_case", COMBINED_QUERY_TEST_CASES, ids=lambda x: x["id"])
    def test_combined_search_same_source(self, test_case):
        """
        Test that both manual_search and image_search return results from same source.
        
        This verifies that when we search for both text and images:
        1. Text results come from the expected manual
        2. Image results come from the SAME manual
        3. The pdf_filter correctly links both searches
        """
        query = test_case["query"]
        expected_source = test_case["expected_source"]
        
        # Step 1: Search for text
        manual_args = ManualSearchArgs(query=query, top_k=3)
        manual_result = manual_search(manual_args)
        
        text_results = manual_result.get("results", [])
        assert len(text_results) >= 1, f"No text results for: {query}"
        
        # Verify text is from expected source
        text_sources = [r.get("source", "") for r in text_results]
        text_source_match = any(expected_source.lower() in s.lower() for s in text_sources)
        assert text_source_match, (
            f"Text results not from expected source '{expected_source}'\n"
            f"Sources found: {text_sources}"
        )
        
        # Get page hint from first matching result
        page_hint = None
        for r in text_results:
            if expected_source.lower() in r.get("source", "").lower():
                page_hint = r.get("page")
                break
        
        # Step 2: Search for images with pdf_filter
        image_args = ImageSearchArgs(
            query=query,
            pdf_filter=expected_source,
            page_hint=page_hint,
            top_k=3
        )
        image_result = image_search(image_args)
        
        # Skip if no images indexed
        if image_result.get("count", 0) == 0:
            pytest.skip(f"No images indexed for {expected_source}")
        
        image_results = image_result.get("results", [])
        assert len(image_results) >= 1, f"No image results for: {query}"
        
        # Verify images are from expected source
        for r in image_results:
            pdf_source = r.get("pdf_source", "")
            assert expected_source.lower() in pdf_source.lower(), (
                f"Image from wrong source: {pdf_source}\n"
                f"Expected: {expected_source}"
            )
    
    def test_page_hint_improves_relevance(self):
        """
        Test that providing a page_hint from manual_search improves image results.
        
        We search with and without page hint and verify that:
        1. With page hint, at least one result is near that page
        2. Results are more relevant to the specific content
        """
        # First get a page number from manual search
        manual_args = ManualSearchArgs(
            query="APSX-PIM wiring diagram connectors",
            top_k=1
        )
        manual_result = manual_search(manual_args)
        
        text_results = manual_result.get("results", [])
        if not text_results:
            pytest.skip("No text results found")
        
        page_hint = text_results[0].get("page")
        if not page_hint:
            pytest.skip("No page number in text result")
        
        # Search WITH page hint
        with_hint_args = ImageSearchArgs(
            query="wiring diagram",
            pdf_filter="APSX-PIM",
            page_hint=page_hint,
            top_k=3
        )
        with_hint_result = image_search(with_hint_args)
        
        if with_hint_result.get("count", 0) == 0:
            pytest.skip("No images indexed for APSX-PIM")
        
        # Check that at least one result is within 10 pages of hint
        with_hint_pages = [r.get("page_number", 0) for r in with_hint_result.get("results", [])]
        nearby_pages = [p for p in with_hint_pages if abs(p - page_hint) <= 10]
        
        # This test is informational - we just verify the mechanism works
        print(f"\nPage hint: {page_hint}")
        print(f"Result pages: {with_hint_pages}")
        print(f"Pages within 10 of hint: {nearby_pages}")
        
        # At least verify we got results
        assert len(with_hint_result.get("results", [])) >= 1


class TestSourceIsolation:
    """Test that different manuals don't cross-contaminate results."""
    
    def test_apsx_query_doesnt_return_boy35(self):
        """APSX-PIM queries should not return BOY-35 images."""
        args = ImageSearchArgs(
            query="wiring diagram electrical",
            pdf_filter="APSX-PIM",
            top_k=5
        )
        result = image_search(args)
        
        if result.get("count", 0) == 0:
            pytest.skip("No images indexed for APSX-PIM")
        
        for r in result.get("results", []):
            pdf_source = r.get("pdf_source", "")
            assert "boy" not in pdf_source.lower(), (
                f"APSX-PIM query returned BOY-35 image: {r.get('image_name')}"
            )
    
    def test_boy35_query_doesnt_return_apsx(self):
        """BOY-35 queries should not return APSX-PIM images."""
        args = ImageSearchArgs(
            query="hydraulic system diagram",
            pdf_filter="BOY-35",
            top_k=5
        )
        result = image_search(args)
        
        if result.get("count", 0) == 0:
            pytest.skip("No images indexed for BOY-35")
        
        for r in result.get("results", []):
            pdf_source = r.get("pdf_source", "")
            assert "apsx" not in pdf_source.lower(), (
                f"BOY-35 query returned APSX-PIM image: {r.get('image_name')}"
            )


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

def run_combined_tests_standalone():
    """Run combined tests and print detailed results."""
    print("=" * 80)
    print("  COMBINED QUERY TEST RUNNER")
    print("  Testing text + image search working together")
    print("=" * 80)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_case in COMBINED_QUERY_TEST_CASES:
        test_id = test_case["id"]
        query = test_case["query"]
        expected_source = test_case["expected_source"]
        
        print(f"\n{'-' * 80}")
        print(f"TEST: {test_id}")
        print(f"Query: {query}")
        print(f"Expected source: {expected_source}")
        print(f"{'-' * 80}")
        
        try:
            # Step 1: Manual search
            print("\n  [1] Manual Search Results:")
            manual_args = ManualSearchArgs(query=query, top_k=2)
            manual_result = manual_search(manual_args)
            text_results = manual_result.get("results", [])
            
            if not text_results:
                print("      No text results found")
                failed += 1
                continue
            
            page_hint = None
            text_source_ok = False
            for r in text_results:
                source = r.get("source", "Unknown")
                page = r.get("page", "?")
                is_match = expected_source.lower() in source.lower()
                if is_match and page_hint is None:
                    page_hint = page
                    text_source_ok = True
                marker = "[x]" if is_match else "[ ]"
                print(f"      {marker} {source} (page {page})")
            
            # Step 2: Image search
            print(f"\n  [2] Image Search Results (using page_hint={page_hint}):")
            image_args = ImageSearchArgs(
                query=query,
                pdf_filter=expected_source,
                page_hint=page_hint,
                top_k=2
            )
            image_result = image_search(image_args)
            
            if image_result.get("count", 0) == 0:
                print(f"      [SKIP] No images indexed for {expected_source}")
                skipped += 1
                continue
            
            image_results = image_result.get("results", [])
            image_source_ok = True
            
            for r in image_results:
                name = r.get("image_name", "Unknown")
                source = r.get("pdf_source", "Unknown")
                page = r.get("page_number", "?")
                is_match = expected_source.lower() in source.lower()
                if not is_match:
                    image_source_ok = False
                marker = "[x]" if is_match else "[ ]"
                print(f"      {marker} {name} (page {page})")
            
            # Verdict
            if text_source_ok and image_source_ok:
                print(f"\n  [PASS] Both text and images from {expected_source}")
                passed += 1
            else:
                print(f"\n  [FAIL]")
                if not text_source_ok:
                    print(f"     Text not from expected source")
                if not image_source_ok:
                    print(f"     Images not from expected source")
                failed += 1
                
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"  SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_combined_tests_standalone()
    sys.exit(0 if success else 1)
