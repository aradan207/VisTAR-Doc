#!/usr/bin/env python3
"""
Test Manual Search Queries Across Multiple Equipment Manuals

=============================================================================
WHAT THIS FILE TESTS:
=============================================================================

This file tests the manual_search tool to verify that text-based queries
return correct and relevant content from the indexed PDF manuals.

We test:
1. Does the search find content from the CORRECT manual/machine?
2. Does the search return the EXPECTED page numbers?
3. Is the content RELEVANT to the query (check key terms are present)?
4. Does filtering by source work correctly?

=============================================================================
HOW WE GET THE DATA:
=============================================================================

- The manual_search tool uses a FAISS vector store to find similar text chunks
- Text chunks come from PDF manuals that were pre-processed and indexed
- Each chunk has: content, source (PDF name), page number, relevance score
- The search uses sentence embeddings to find semantically similar content
- Data location: agentic-rag/data/faiss_index/

=============================================================================
EXPECTED RESULTS:
=============================================================================

Each test case defines:
- query: What we're asking
- expected_source: Which PDF should have the answer (partial match is OK)
- expected_terms: Words that MUST appear in the returned content
- min_results: Minimum number of results we expect to get

If the test fails, it means either:
1. The content isn't indexed properly
2. The embeddings aren't capturing the semantic meaning
3. The expected data changed in the source PDFs

=============================================================================
HOW TO RUN:
=============================================================================

Run all tests in this file:
    uv run pytest tests/test_manual_queries.py -v

Run a specific test:
    uv run pytest tests/test_manual_queries.py::TestManualSearchQueries::test_manual_search_returns_correct_source -v

Run standalone (outside pytest):
    uv run python tests/test_manual_queries.py

=============================================================================
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backend.api.tools.manual_search import manual_search, ManualSearchArgs


# =============================================================================
# TEST CASES - Each tests a different machine/manual
# =============================================================================

MANUAL_SEARCH_TEST_CASES = [
    # --- APSX-PIM Desktop Injection Molding Machine ---
    {
        "id": "apsx-pim-overview",
        "query": "APSX-PIM machine specifications and features",
        "expected_source": "APSX-PIM",
        "expected_terms": ["injection"],
        "min_results": 1,
        "description": "Basic machine overview query",
    },
    {
        "id": "apsx-pim-temperature",
        "query": "APSX-PIM barrel temperature settings and zones",
        "expected_source": "APSX-PIM",
        "expected_terms": ["temperature"],
        "min_results": 1,
        "description": "Temperature control configuration",
    },
    {
        "id": "apsx-pim-clamp",
        "query": "APSX-PIM clamp force and mold closing",
        "expected_source": "APSX-PIM",
        "expected_terms": ["clamp"],
        "min_results": 1,
        "description": "Clamping system details",
    },
    
    # --- BOY-35 Injection Molding Machine ---
    {
        "id": "boy35-hydraulic",
        "query": "BOY 35 hydraulic system pressure settings",
        "expected_source": "BOY-35",
        "expected_terms": ["hydraulic"],
        "min_results": 1,
        "description": "Hydraulic system configuration",
    },
    {
        "id": "boy35-safety",
        "query": "BOY 35 safety gate and emergency stop",
        "expected_source": "BOY-35",
        "expected_terms": ["safety"],
        "min_results": 1,
        "description": "Safety features and emergency procedures",
    },
    {
        "id": "boy35-injection",
        "query": "BOY 35 injection speed and screw position",
        "expected_source": "BOY-35",
        "expected_terms": ["injection"],
        "min_results": 1,
        "description": "Injection parameters",
    },
    
    # --- MILACRON Injection Molding Machine ---
    {
        "id": "milacron-ejector",
        "query": "MILACRON ejector system setup and configuration",
        "expected_source": "MILACRON",
        "expected_terms": ["ejector"],
        "min_results": 1,
        "description": "Ejector system setup",
    },
    
    # --- MAAC Thermoformer ---
    {
        "id": "maac-forming",
        "query": "MAAC thermoformer forming cycle timer settings",
        "expected_source": "MAAC",
        "expected_terms": ["forming"],
        "min_results": 1,
        "description": "Forming cycle configuration",
    },
    
    # --- RSA-G2 Rheometer ---
    {
        "id": "rsa-g2-testing",
        "query": "RSA-G2 tension testing and sample clamping",
        "expected_source": "RSA-G2",
        "expected_terms": ["tension"],
        "min_results": 1,
        "description": "Mechanical testing procedures",
    },
]


class TestManualSearchQueries:
    """Test suite for manual search queries across multiple equipment manuals."""
    
    @pytest.mark.parametrize("test_case", MANUAL_SEARCH_TEST_CASES, ids=lambda x: x["id"])
    def test_manual_search_returns_correct_source(self, test_case):
        """
        Test that manual_search returns results from the expected source manual.
        
        This verifies that:
        1. The query returns results (not empty)
        2. At least one result is from the expected manual
        3. The content contains expected key terms
        """
        query = test_case["query"]
        expected_source = test_case["expected_source"]
        expected_terms = test_case["expected_terms"]
        min_results = test_case["min_results"]
        
        # Run the search
        args = ManualSearchArgs(query=query, top_k=5)
        result = manual_search(args)
        
        # Check we got results
        assert "results" in result, f"No 'results' key in response for query: {query}"
        results = result["results"]
        
        assert len(results) >= min_results, (
            f"Expected at least {min_results} results, got {len(results)}\n"
            f"Query: {query}\n"
            f"Description: {test_case['description']}"
        )
        
        # Check at least one result is from expected source
        sources_found = [r.get("source", "") for r in results]
        source_match = any(expected_source.lower() in s.lower() for s in sources_found)
        
        assert source_match, (
            f"Expected source containing '{expected_source}' not found\n"
            f"Query: {query}\n"
            f"Sources found: {sources_found}\n"
            f"Description: {test_case['description']}"
        )
        
        # Check expected terms appear in at least one result
        all_content = " ".join(r.get("content", "").lower() for r in results)
        
        for term in expected_terms:
            assert term.lower() in all_content, (
                f"Expected term '{term}' not found in any result\n"
                f"Query: {query}\n"
                f"Description: {test_case['description']}"
            )
    
    def test_manual_search_returns_page_numbers(self):
        """Test that all results include valid page numbers."""
        args = ManualSearchArgs(query="injection molding temperature", top_k=5)
        result = manual_search(args)
        
        for r in result.get("results", []):
            assert "page" in r, "Result missing 'page' field"
            assert isinstance(r["page"], int), f"Page should be int, got {type(r['page'])}"
            assert r["page"] > 0, f"Page number should be positive, got {r['page']}"
    
    def test_manual_search_returns_relevance_scores(self):
        """Test that all results include relevance scores."""
        args = ManualSearchArgs(query="safety procedures emergency stop", top_k=5)
        result = manual_search(args)
        
        for r in result.get("results", []):
            assert "relevance_score" in r, "Result missing 'relevance_score' field"
            score = r["relevance_score"]
            assert isinstance(score, (int, float)), f"Score should be numeric, got {type(score)}"
    
    def test_top_k_limits_results(self):
        """Test that top_k parameter correctly limits number of results."""
        for k in [1, 3, 5]:
            args = ManualSearchArgs(query="machine operation manual", top_k=k)
            result = manual_search(args)
            
            assert len(result.get("results", [])) <= k, (
                f"Asked for top_k={k} but got {len(result['results'])} results"
            )


# =============================================================================
# STANDALONE RUNNER - For running outside pytest
# =============================================================================

def run_manual_tests_standalone():
    """Run tests and print detailed results for manual review."""
    print("=" * 80)
    print("  MANUAL SEARCH TEST RUNNER")
    print("  Testing text search across multiple equipment manuals")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_case in MANUAL_SEARCH_TEST_CASES:
        test_id = test_case["id"]
        query = test_case["query"]
        expected_source = test_case["expected_source"]
        expected_terms = test_case["expected_terms"]
        
        print(f"\n{'-' * 80}")
        print(f"TEST: {test_id}")
        print(f"Query: {query}")
        print(f"Expected source: {expected_source}")
        print(f"Expected terms: {expected_terms}")
        print(f"{'-' * 80}")
        
        try:
            args = ManualSearchArgs(query=query, top_k=3)
            result = manual_search(args)
            results = result.get("results", [])
            
            if not results:
                print("  [FAIL] No results returned")
                failed += 1
                continue
            
            # Check source match
            sources = [r.get("source", "") for r in results]
            source_match = any(expected_source.lower() in s.lower() for s in sources)
            
            # Check terms
            all_content = " ".join(r.get("content", "").lower() for r in results)
            terms_found = [t for t in expected_terms if t.lower() in all_content]
            terms_missing = [t for t in expected_terms if t.lower() not in all_content]
            
            # Print results
            for i, r in enumerate(results, 1):
                match_marker = "[x]" if expected_source.lower() in r.get("source", "").lower() else "[ ]"
                print(f"  {match_marker} Result {i}: {r.get('source', 'Unknown')} (page {r.get('page', '?')})")
                print(f"      Score: {r.get('relevance_score', 0):.4f}")
                content_preview = r.get("content", "")[:100].replace("\n", " ")
                print(f"      Content: {content_preview}...")
            
            # Overall verdict
            if source_match and not terms_missing:
                print(f"\n  [PASS]")
                passed += 1
            else:
                print(f"\n  [FAIL]")
                if not source_match:
                    print(f"     - Expected source '{expected_source}' not found in: {sources}")
                if terms_missing:
                    print(f"     - Missing terms: {terms_missing}")
                failed += 1
                
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"  SUMMARY: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_manual_tests_standalone()
    sys.exit(0 if success else 1)
