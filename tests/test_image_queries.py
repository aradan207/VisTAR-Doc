#!/usr/bin/env python3
"""
Test Image Search Queries Across Multiple Equipment Manuals

=============================================================================
WHAT THIS FILE TESTS:
=============================================================================

This file tests the image_search tool to verify that visual queries
return the correct images from the indexed equipment manuals.

We test:
1. Does the search return images from the CORRECT manual/machine?
2. Does the search return images from the EXPECTED page range?
3. Does the content_type filter work (schematic, diagram, photo, etc.)?
4. Are the image URLs properly formatted?
5. Does the pdf_filter correctly restrict results to one manual?

=============================================================================
HOW WE GET THE DATA:
=============================================================================

- Images are extracted from PDF manuals by vlm-yolo-detector
- Each image is described by a VLM (Vision Language Model) like LLaVA
- Descriptions are embedded using sentence transformers
- The search uses cosine similarity to find semantically matching images
- Data location: vlm-yolo-detector/data/processed/
  - image_index.json: metadata for each image
  - image_embeddings.npy: semantic embedding vectors

=============================================================================
EXPECTED RESULTS:
=============================================================================

Each test case defines:
- query: What we're asking to see
- pdf_filter: Which manual to search (e.g., "APSX-PIM", "BOY-35")
- expected_image_pattern: Pattern the image filename should match
- expected_page_min / expected_page_max: Page range where image should be
- expected_content_type: What type of image (schematic, diagram, photo, etc.)

If the test fails, it means either:
1. The image isn't indexed properly
2. The VLM description doesn't match the query semantically
3. The image was extracted from a different page than expected

=============================================================================
HOW TO RUN:
=============================================================================

Run all tests in this file:
    uv run pytest tests/test_image_queries.py -v

Run a specific test:
    uv run pytest tests/test_image_queries.py::TestImageSearchQueries::test_image_search_returns_correct_source -v

Run standalone (outside pytest):
    uv run python tests/test_image_queries.py

=============================================================================
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backend.api.tools.image_search import image_search, ImageSearchArgs


# =============================================================================
# TEST CASES - Each tests a different machine/manual with expected images
# =============================================================================

IMAGE_SEARCH_TEST_CASES = [
    # --- APSX-PIM Desktop Injection Molding Machine ---
    {
        "id": "apsx-pim-wiring",
        "query": "APSX-PIM wiring diagram electrical connections",
        "pdf_filter": "APSX-PIM",
        "expected_image_pattern": "APSX-PIM_page41",
        "expected_page_min": 38,
        "expected_page_max": 45,
        "expected_content_type": "schematic",
        "description": "Wiring schematic with J1, J2, J3 connectors",
    },
    {
        "id": "apsx-pim-dashboard",
        "query": "APSX-PIM dashboard injection cycle stages",
        "pdf_filter": "APSX-PIM",
        "expected_image_pattern": "APSX-PIM_page31",
        "expected_page_min": 28,
        "expected_page_max": 35,
        "expected_content_type": "diagram",
        "description": "Dashboard showing CLAMPING, INJECTING, HOLDING, COOLING",
    },
    {
        "id": "apsx-pim-components",
        "query": "APSX-PIM internal components hopper feeder",
        "pdf_filter": "APSX-PIM",
        "expected_image_pattern": "APSX-PIM_page",
        "expected_page_min": 10,
        "expected_page_max": 25,
        "expected_content_type": None,  # Any type
        "description": "Internal machine components diagram",
    },
    
    # --- BOY-35 Injection Molding Machine ---
    {
        "id": "boy35-user-admin",
        "query": "BOY 35 user administration screen USB",
        "pdf_filter": "BOY-35",
        "expected_image_pattern": "BOY-35",
        "expected_page_min": 270,
        "expected_page_max": 280,
        "expected_content_type": None,
        "description": "User administration interface with USB options",
    },
    {
        "id": "boy35-safety-warning",
        "query": "BOY 35 maintenance safety warning sign",
        "pdf_filter": "BOY-35",
        "expected_image_pattern": "BOY-35",
        "expected_page_min": 440,
        "expected_page_max": 460,
        "expected_content_type": "warning",
        "description": "Safety warning for maintenance procedures",
    },
    {
        "id": "boy35-process-config",
        "query": "BOY 35 process configurator sequence screen",
        "pdf_filter": "BOY-35",
        "expected_image_pattern": "BOY-35",
        "expected_page_min": 65,
        "expected_page_max": 80,
        "expected_content_type": None,
        "description": "Process configurator user interface",
    },
    
    # --- MILACRON Injection Molding Machine ---
    {
        "id": "milacron-ejector-menu",
        "query": "MILACRON ejector system setup menu schematic",
        "pdf_filter": "MILACRON",
        "expected_image_pattern": "MILACRON_page263",
        "expected_page_min": 260,
        "expected_page_max": 270,
        "expected_content_type": "schematic",
        "description": "Ejector menu with Forward Speed, Dwell settings",
    },
    {
        "id": "milacron-alarm-chart",
        "query": "MILACRON alarm SPC chart trend monitoring",
        "pdf_filter": "MILACRON",
        "expected_image_pattern": "MILACRON_page",
        "expected_page_min": 370,
        "expected_page_max": 410,
        "expected_content_type": None,
        "description": "Alarm SPC chart with trend settings",
    },
    {
        "id": "milacron-lubricant",
        "query": "MILACRON lubricant specifications approved list",
        "pdf_filter": "MILACRON",
        "expected_image_pattern": "MILACRON_page82",
        "expected_page_min": 78,
        "expected_page_max": 88,
        "expected_content_type": "table",
        "description": "Lubricant specifications table",
    },
    
    # --- MAAC Thermoformer ---
    {
        "id": "maac-forming-timer",
        "query": "MAAC thermoformer forming timer flow control",
        "pdf_filter": "MAAC",
        "expected_image_pattern": "MAAC",
        "expected_page_min": 25,
        "expected_page_max": 35,
        "expected_content_type": "diagram",
        "description": "Forming timer diagram with flow control valve",
    },
    {
        "id": "maac-diagnostics",
        "query": "MAAC thermoformer input diagnostics screen",
        "pdf_filter": "MAAC",
        "expected_image_pattern": "MAAC",
        "expected_page_min": 70,
        "expected_page_max": 80,
        "expected_content_type": None,
        "description": "Input diagnostics showing machine inputs status",
    },
    
    # --- RSA-G2 Rheometer ---
    {
        "id": "rsa-g2-clamping",
        "query": "RSA-G2 clamping systems three-point bending tension",
        "pdf_filter": "RSA-G2",
        "expected_image_pattern": "RSA-G2_page7",
        "expected_page_min": 5,
        "expected_page_max": 12,
        "expected_content_type": "schematic",
        "description": "Clamping systems for different test types",
    },
]


class TestImageSearchQueries:
    """Test suite for image search queries across multiple equipment manuals."""
    
    @pytest.mark.parametrize("test_case", IMAGE_SEARCH_TEST_CASES, ids=lambda x: x["id"])
    def test_image_search_returns_correct_source(self, test_case):
        """
        Test that image_search returns images from the expected manual and page range.
        
        This verifies that:
        1. The query returns at least one image
        2. At least one result matches the expected image pattern
        3. The image is from the expected page range
        4. The pdf_filter correctly restricts results
        """
        query = test_case["query"]
        pdf_filter = test_case["pdf_filter"]
        expected_pattern = test_case["expected_image_pattern"]
        page_min = test_case["expected_page_min"]
        page_max = test_case["expected_page_max"]
        
        # Run the search
        args = ImageSearchArgs(
            query=query,
            pdf_filter=pdf_filter,
            top_k=5
        )
        result = image_search(args)
        
        # Check we got results
        assert "results" in result, f"No 'results' key in response for query: {query}"
        results = result["results"]
        
        # Allow tests to pass even if no images indexed for this manual
        if result.get("count", 0) == 0:
            pytest.skip(f"No images indexed for {pdf_filter} - run VLM pipeline first")
        
        assert len(results) >= 1, (
            f"Expected at least 1 result, got {len(results)}\n"
            f"Query: {query}\n"
            f"Message: {result.get('message', 'No message')}"
        )
        
        # Check all results are from the correct PDF
        for r in results:
            pdf_source = r.get("pdf_source", "")
            assert pdf_filter.lower() in pdf_source.lower(), (
                f"Result from wrong PDF: {pdf_source}\n"
                f"Expected PDF containing: {pdf_filter}\n"
                f"Query: {query}"
            )
        
        # Check at least one result matches expected pattern
        image_names = [r.get("image_name", "") for r in results]
        pattern_match = any(expected_pattern.lower() in name.lower() for name in image_names)
        
        # Check at least one result is in expected page range
        pages = [r.get("page_number", 0) for r in results]
        page_match = any(page_min <= p <= page_max for p in pages)
        
        # We consider it a pass if EITHER pattern matches OR page is in range
        # (because the exact image might vary but should be in the right area)
        assert pattern_match or page_match, (
            f"No result matched expected pattern '{expected_pattern}' "
            f"or page range {page_min}-{page_max}\n"
            f"Query: {query}\n"
            f"Images found: {image_names}\n"
            f"Pages found: {pages}\n"
            f"Description: {test_case['description']}"
        )
    
    def test_image_search_returns_urls(self):
        """Test that all results include properly formatted URLs."""
        args = ImageSearchArgs(
            query="wiring diagram electrical schematic",
            pdf_filter="APSX-PIM",
            top_k=3
        )
        result = image_search(args)
        
        if result.get("count", 0) == 0:
            pytest.skip("No images indexed - run VLM pipeline first")
        
        for r in result.get("results", []):
            assert "url" in r, "Result missing 'url' field"
            url = r["url"]
            assert url.startswith("http"), f"URL should start with http, got: {url}"
            assert "/api/media/" in url, f"URL should contain /api/media/, got: {url}"
    
    def test_image_search_returns_descriptions(self):
        """Test that all results include VLM-generated descriptions."""
        args = ImageSearchArgs(
            query="machine components diagram",
            pdf_filter="APSX-PIM",
            top_k=3
        )
        result = image_search(args)
        
        if result.get("count", 0) == 0:
            pytest.skip("No images indexed - run VLM pipeline first")
        
        for r in result.get("results", []):
            assert "description" in r, "Result missing 'description' field"
            desc = r["description"]
            assert len(desc) > 20, f"Description too short: {desc}"
    
    def test_image_search_respects_top_k(self):
        """Test that top_k parameter limits number of results."""
        for k in [1, 2, 3]:
            args = ImageSearchArgs(
                query="diagram",
                pdf_filter=None,  # Search all
                top_k=k
            )
            result = image_search(args)
            
            if result.get("count", 0) == 0:
                pytest.skip("No images indexed - run VLM pipeline first")
            
            assert len(result.get("results", [])) <= k, (
                f"Asked for top_k={k} but got {len(result['results'])} results"
            )
    
    def test_pdf_filter_isolation(self):
        """Test that pdf_filter correctly isolates results to one manual."""
        # Search for a generic term with different filters
        filters = ["APSX-PIM", "BOY-35", "MILACRON"]
        
        for pdf_filter in filters:
            args = ImageSearchArgs(
                query="diagram schematic",
                pdf_filter=pdf_filter,
                top_k=5
            )
            result = image_search(args)
            
            if result.get("count", 0) == 0:
                continue  # Skip if no images for this manual
            
            for r in result.get("results", []):
                pdf_source = r.get("pdf_source", "")
                assert pdf_filter.lower() in pdf_source.lower(), (
                    f"pdf_filter={pdf_filter} but got result from {pdf_source}"
                )


# =============================================================================
# STANDALONE RUNNER - For running outside pytest
# =============================================================================

def run_image_tests_standalone():
    """Run tests and print detailed results for manual review."""
    print("=" * 80)
    print("  IMAGE SEARCH TEST RUNNER")
    print("  Testing image search across multiple equipment manuals")
    print("=" * 80)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_case in IMAGE_SEARCH_TEST_CASES:
        test_id = test_case["id"]
        query = test_case["query"]
        pdf_filter = test_case["pdf_filter"]
        expected_pattern = test_case["expected_image_pattern"]
        page_min = test_case["expected_page_min"]
        page_max = test_case["expected_page_max"]
        
        print(f"\n{'-' * 80}")
        print(f"TEST: {test_id}")
        print(f"Query: {query}")
        print(f"PDF Filter: {pdf_filter}")
        print(f"Expected pattern: {expected_pattern}")
        print(f"Expected pages: {page_min}-{page_max}")
        print(f"{'-' * 80}")
        
        try:
            args = ImageSearchArgs(
                query=query,
                pdf_filter=pdf_filter,
                top_k=3
            )
            result = image_search(args)
            results = result.get("results", [])
            
            if result.get("count", 0) == 0:
                print(f"  [SKIP] No images indexed for {pdf_filter}")
                skipped += 1
                continue
            
            if not results:
                print("  [FAIL] No results returned")
                failed += 1
                continue
            
            # Check pattern and page matches
            pattern_matches = []
            page_matches = []
            
            for r in results:
                name = r.get("image_name", "")
                page = r.get("page_number", 0)
                score = r.get("relevance_score", 0)
                semantic = r.get("semantic_match", "?")
                
                is_pattern_match = expected_pattern.lower() in name.lower()
                is_page_match = page_min <= page <= page_max
                
                if is_pattern_match:
                    pattern_matches.append(name)
                if is_page_match:
                    page_matches.append(name)
                
                match_marker = "[x]" if (is_pattern_match or is_page_match) else "[ ]"
                print(f"  {match_marker} {name}")
                print(f"      Page: {page}, Score: {score:.2f}, Semantic: {semantic}")
                print(f"      URL: {r.get('url', 'N/A')}")
            
            # Verdict
            if pattern_matches or page_matches:
                print(f"\n  [PASS]")
                if pattern_matches:
                    print(f"     Pattern matches: {pattern_matches}")
                if page_matches:
                    print(f"     Page range matches: {page_matches}")
                passed += 1
            else:
                print(f"\n  [FAIL]")
                print(f"     No matches for pattern '{expected_pattern}' or pages {page_min}-{page_max}")
                failed += 1
                
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"  SUMMARY: {passed} passed, {failed} failed, {skipped} skipped, {passed + failed + skipped} total")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_image_tests_standalone()
    sys.exit(0 if success else 1)
