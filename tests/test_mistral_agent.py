"""
Quick manual script for local debugging of the Mistral integration.

Run with: uv run python tests/test_mistral_agent.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backend.api.tools.web import fetch_url, web_search
from app.backend.core.agent.agent_manager import AgentManager
from app.backend.core.agent.mistralLlm import MistralLLM


def test_mistral_agent():
    """Test the Mistral LLM agent integration."""
    llm = MistralLLM(model_name="codestral-2508")
    llm.register_decorated_tool(web_search)
    llm.register_decorated_tool(fetch_url)

    # Ensure tools are registered
    assert llm.get_tools_spec(), "No tools registered. Ensure tools are properly decorated and registered."

    am = AgentManager("I want to plan a trip to Madrid.", llm)
    am.run()

    print("Reasoning Tree Context:")
    print(am.reasoning_tree.get_reasoning_tree_context())
    print("\nLeaves:")
    print(am.reasoning_tree.leaves)


if __name__ == "__main__":
    test_mistral_agent()
