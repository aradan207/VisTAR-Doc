"""
Manual smoke-tests for the HTTP tools.

Run with: uv run python tests/test_web_tools.py
"""

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
