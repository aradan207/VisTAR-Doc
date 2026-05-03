#!/usr/bin/env python3
"""
Test Image URL Safety

Ensures that only URLs returned by image_search are kept, and that
container-internal hosts are never emitted.
"""

from app.backend.core.agent.agent_manager import AgentManager
from app.backend.core.models.tool_calls import ToolCall


class DummyLLM:
    def generate(self, *args, **kwargs):
        return ""

    def run_tool(self, *args, **kwargs):
        return {}


def _build_manager_with_urls(urls):
    manager = AgentManager(user_input="test", llm=DummyLLM())
    results = {"results": [{"url": url, "description": "image"} for url in urls]}
    tc = ToolCall(tool_name="image_search", args={}, result=results)
    manager.reasoning_tree.add_leaf(
        description="image search",
        parent_leaf="leaf_0",
        tool_calls=[tc],
        result="",
    )
    return manager


def test_strip_hallucinated_urls_keeps_only_tool_results():
    manager = _build_manager_with_urls([
        "/api/media/yologen/train/APSX-PIM_page41_img1.png",
    ])
    answer = (
        "![bad](http://backend:8000/api/media/yologen/train/APSX-PIM_page41_img1.png)\n"
        "![good](/api/media/yologen/train/APSX-PIM_page41_img1.png)"
    )
    cleaned = manager._strip_hallucinated_urls(answer)
    assert "backend:8000" not in cleaned
    assert "/api/media/yologen/train/APSX-PIM_page41_img1.png" in cleaned


def test_strip_hallucinated_urls_allows_absolute_tool_urls():
    public_url = "http://example.com/api/media/yologen/train/APSX-PIM_page41_img1.png"
    manager = _build_manager_with_urls([public_url])
    answer = f"![good]({public_url})"
    cleaned = manager._strip_hallucinated_urls(answer)
    assert public_url in cleaned


def test_strip_hallucinated_urls_allows_relative_when_tool_is_absolute():
    public_url = "http://example.com/api/media/yologen/train/APSX-PIM_page41_img1.png"
    manager = _build_manager_with_urls([public_url])
    answer = "![good](/api/media/yologen/train/APSX-PIM_page41_img1.png)"
    cleaned = manager._strip_hallucinated_urls(answer)
    assert "/api/media/yologen/train/APSX-PIM_page41_img1.png" in cleaned
