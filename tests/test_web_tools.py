#!/usr/bin/env python3
"""
Test Web Tools - External Web Search and URL Fetch

=============================================================================
WHAT THIS FILE TESTS:
=============================================================================

This file tests the web-related tools that the agent can use
when local knowledge is insufficient.

We test:
1. web_search: Searches DuckDuckGo for external information
2. fetch_url: Downloads and cleans content from web pages

=============================================================================
HOW WE GET THE DATA:
=============================================================================

- web_search sends requests to DuckDuckGo Lite
- fetch_url downloads HTML from the given URL and extracts text
- Both tools require internet access

=============================================================================
EXPECTED RESULTS:
=============================================================================

- web_search returns a list of search results with titles and URLs
- fetch_url returns cleaned text content from the page
- Both should handle errors gracefully (timeouts, bad URLs, etc.)

=============================================================================
HOW TO RUN:
=============================================================================

Run directly:
    uv run python tests/test_web_tools.py

Run with pytest:
    uv run pytest tests/test_web_tools.py -v

============================================================================="""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backend.api.tools.web import WebSearchArgs, FetchURLArgs, fetch_url, web_search


def test_web_search():
    """Test web search functionality."""
    print("Testing web search...")
    args = WebSearchArgs(query="flight to Madrid")
    result = web_search(args)
    print(result)
    return result


def test_fetch_url():
    """Test URL fetch functionality."""
    print("\nTesting URL fetch...")
    url = "https://www.travelandvoyage.com/post/madrid-ultimate-itinerary"
    url_args = FetchURLArgs(url=url)
    result = fetch_url(url_args)
    print(result)
    return result


if __name__ == "__main__":
    test_web_search()
    test_fetch_url()
