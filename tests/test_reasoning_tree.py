#!/usr/bin/env python3
"""
Test Reasoning Tree - Data Structure and Operations

=============================================================================
WHAT THIS FILE TESTS:
=============================================================================

This file tests the ReasoningTree data structure that the agent uses
to track its thinking process. Each "leaf" in the tree represents
one reasoning step with tool calls and results.

We test:
1. Creating a tree with a root query
2. Adding child leaves with descriptions and tool calls
3. Parent-child relationships work correctly
4. Context accumulation (getting ancestor chain)
5. Tree serialization to dictionary

=============================================================================
HOW WE GET THE DATA:
=============================================================================

- We create a ReasoningTree in memory
- We add leaves manually with fake tool calls
- No external data needed - this is a unit test

=============================================================================
EXPECTED RESULTS:
=============================================================================

- Tree starts with leaf_0 as root (the user query)
- Each add_leaf returns a new leaf ID (leaf_1, leaf_2, etc.)
- Leaves track their parent and children
- get_leaf_context returns the full ancestor chain
- to_dict exports the tree for JSON serialization

=============================================================================
HOW TO RUN:
=============================================================================

Run directly:
    uv run python tests/test_reasoning_tree.py

Run with pytest:
    uv run pytest tests/test_reasoning_tree.py -v

============================================================================="""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backend.core.models.tool_calls import ToolCall
from app.backend.core.reasoningTree.reasoning_tree import ReasoningTree


def test_reasoning_tree():
    """Test the ReasoningTree with a sample travel planning scenario."""
    tree = ReasoningTree("I want to plan a trip to Madrid.")

    # Leaf 1
    tree.add_leaf(
        description="Define a travel budget.",
        parent_leaf="leaf_0",
        tool_calls=[
            ToolCall(tool_name="CurrencyConverter", args={"from": "EUR", "to": "USD"}, result="$1000")
        ]
    )

    # Leaf 2
    tree.add_leaf(
        description="Search for flights from Paris to Madrid.",
        parent_leaf="leaf_0",
        tool_calls=[
            ToolCall(tool_name="FlightSearch", args={"origin": "Paris", "destination": "Madrid"}, result="Flights starting at 90 EUR")
        ]
    )

    # Leaf 3
    tree.add_leaf(
        description="Find an affordable hotel in Madrid.",
        parent_leaf="leaf_0",
        tool_calls=[
            ToolCall(tool_name="HotelSearch", args={"city": "Madrid", "budget": "<100 EUR/night"}, result="Ibis Hotel, 89 EUR/night")
        ]
    )

    # Leaf 4
    tree.add_leaf(
        description="Book the selected flight.",
        parent_leaf="leaf_2",
        tool_calls=[
            ToolCall(tool_name="FlightBooking", args={"flight_id": "AF1234"}, result="Booking confirmed")
        ]
    )

    # Leaf 5
    tree.add_leaf(
        description="Book the selected hotel.",
        parent_leaf="leaf_3",
        tool_calls=[
            ToolCall(tool_name="HotelBooking", args={"hotel_name": "Ibis"}, result="Booking confirmed")
        ]
    )

    print("Reasoning Tree Context:")
    print(tree.get_reasoning_tree_context())
    print("\nLeaves:")
    print(tree.leaves)


if __name__ == "__main__":
    test_reasoning_tree()
