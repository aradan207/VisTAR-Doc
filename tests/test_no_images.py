#!/usr/bin/env python3
"""
Test No Images - Text-Only Query Validation

=============================================================================
WHAT THIS FILE TESTS:
=============================================================================

This file tests that text-only queries do NOT return images.
This is a critical test for the URL hallucination fix.

We test:
1. Questions like "What is X?" should get text-only answers
2. The LLM should NOT invent image URLs
3. The final answer should have ZERO image markdown (no ![...](...) tags)

=============================================================================
HOW WE GET THE DATA:
=============================================================================

- This test calls the live API at http://localhost:8000
- It sends a query via POST /api/agent/chat
- The agent decides which tools to use based on the query
- For "what is" questions, only manual_search should be called

=============================================================================
EXPECTED RESULTS:
=============================================================================

- The final answer should be text only
- No image URLs in the response (no ![...](http://...) markdown)
- The agent should NOT hallucinate URLs based on page numbers

=============================================================================
REQUIREMENTS:
=============================================================================

The backend server must be running:
    ./start.bat

=============================================================================
HOW TO RUN:
=============================================================================

Start server first, then:
    uv run python tests/test_no_images.py

============================================================================="""

import requests
import json
import time


def test_query(prompt):
    print(f"\nQuery: {prompt}")
    url = "http://localhost:8000/api/agent/chat"
    payload = {"message": prompt}
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        data = response.json()
        print(f"Final Answer ({end_time - start_time:.2f}s):")
        print("-" * 50)
        print(data.get("final_answer", "No answer found"))
        print("-" * 50)
        
        # Check if image tags are present
        if "![" in data.get("final_answer", ""):
            print("WARNING: Still found images in the final answer!")
        else:
            print("SUCCESS: No images in the final answer (as expected).")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    test_query("what is the boy 35 evv?")
