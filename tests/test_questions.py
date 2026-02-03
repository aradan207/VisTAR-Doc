#!/usr/bin/env python3
"""
Test Questions - Manual UI Testing Reference

=============================================================================
WHAT THIS FILE TESTS:
=============================================================================

This file provides sample questions for MANUAL testing in the UI.
It also includes a script to verify image_search results directly.

We test:
1. Do visual queries return the expected images?
2. Are images from the correct page in the manual?
3. Does the pdf_filter correctly restrict to one manual?

=============================================================================
HOW WE GET THE DATA:
=============================================================================

- The verify_image_search() function calls image_search directly
- Expected results are based on manual inspection of the VLM index
- The questions are designed for the APSX-PIM and BOY-35 manuals

=============================================================================
EXPECTED RESULTS:
=============================================================================

Each test question defines:
- question: What to ask in the UI
- expected_image: The image filename we expect
- expected_page: The page number in the PDF
- reason: Why this image should match

=============================================================================
HOW TO RUN:
=============================================================================

Run the verification script:
    uv run python tests/test_questions.py

Or test manually:
    1. Start server: ./start.bat
    2. Open http://localhost:3000
    3. Ask the questions and check results

============================================================================="""

import sys
from pathlib import Path

# Add agentic-rag root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# TEST QUESTIONS - Copy these into the UI
# =============================================================================

TEST_QUESTIONS = [
    # --- Question 1: Should return APSX-PIM_page41_img1.png (wiring diagram) ---
    {
        "question": "Show me the APSX-PIM wiring diagram with J1, J2, J3 connectors",
        "expected_image": "APSX-PIM_page41_img1.png",
        "expected_page": 41,
        "reason": "Page 41 has the wiring schematic with labeled connectors J1-Spring Motor, J2-Sensors, J3-Clamp Motor, J9-Switches",
    },
    
    # --- Question 2: Alternative phrasing for page 41 ---
    {
        "question": "What does the electrical connection diagram look like for APSX-PIM?",
        "expected_image": "APSX-PIM_page41_img1.png",
        "expected_page": 41,
        "reason": "The VLM description mentions 'electrical connections and components'",
    },
    
    # --- Question 3: Test with specific connector mention ---
    {
        "question": "Show the APSX-PIM schematic showing the hopper motor and sensors wiring",
        "expected_image": "APSX-PIM_page41_img1.png", 
        "expected_page": 41,
        "reason": "VLM description mentions '6-Hopper Motor' and 'J2-Sensors'",
    },
    
    # --- Question 4: General schematic request ---
    {
        "question": "I need to see the APSX-PIM machine's internal wiring schematic",
        "expected_image": "APSX-PIM_page41_img1.png",
        "expected_page": 41,
        "reason": "This is the main wiring schematic in the manual",
    },
    
    # --- Question 5: Test another machine (BOY-35) ---
    {
        "question": "Show me BOY-35 hydraulic system diagram",
        "expected_image": "Manual_BOY-35-E-VV_unlocked_page*",
        "expected_page": "varies",
        "reason": "Should return BOY-35 images, not APSX-PIM",
    },
    
    # --- Question 6: Test for operational diagrams ---
    {
        "question": "What are the injection cycle stages shown in APSX-PIM dashboard?",
        "expected_image": "APSX-PIM_page31_img1.jpeg or similar",
        "expected_page": 31,
        "reason": "Dashboard shows CLAMPING, INJECTING, HOLDING, COOLING stages",
    },
]

# =============================================================================
# VERIFICATION SCRIPT
# =============================================================================

def verify_image_search():
    """Verify image search returns expected results."""
    from app.backend.api.tools.image_search import image_search, ImageSearchArgs, _load_all_data
    
    # Force reload data
    _load_all_data(force_reload=True)
    
    print("=" * 70)
    print("  IMAGE SEARCH VERIFICATION")
    print("=" * 70)
    
    for i, test in enumerate(TEST_QUESTIONS[:4], 1):  # Test first 4 questions
        print(f"\n{'='*70}")
        print(f"  TEST {i}: {test['question'][:50]}...")
        print(f"  Expected: {test['expected_image']} (page {test['expected_page']})")
        print("=" * 70)
        
        # Extract machine name for filter
        pdf_filter = None
        if "APSX" in test["question"].upper():
            pdf_filter = "APSX-PIM"
        elif "BOY" in test["question"].upper():
            pdf_filter = "BOY-35"
        
        args = ImageSearchArgs(
            query=test["question"],
            pdf_filter=pdf_filter,
            top_k=3
        )
        
        result = image_search(args)
        
        if result.get('results'):
            print("\nResults:")
            for r in result['results']:
                match = "[MATCH]" if test['expected_image'] in r['image_name'] else ""
                print(f"  {match} {r['image_name']}")
                print(f"    Page: {r['page_number']}, Semantic: {r['semantic_match']}")
                print(f"    URL: {r['url']}")
        else:
            print(f"\n  [FAIL] No results found!")
            print(f"    Message: {result.get('message', 'Unknown error')}")
    
    print("\n" + "=" * 70)
    print("  VERIFICATION COMPLETE")
    print("=" * 70)
    print("\nTo test in the UI:")
    print("1. Stop and restart the server: ./start.bat")
    print("2. Go to http://localhost:3000")
    print("3. Try the questions above")
    print("4. Verify images are displayed correctly")


if __name__ == "__main__":
    verify_image_search()
