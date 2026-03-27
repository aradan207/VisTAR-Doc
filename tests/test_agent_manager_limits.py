import json

from app.backend.core.agent.agent_manager import AgentManager
from app.backend.core.agent.llm import LLM


class StubPlannerLLM(LLM):
    def init_client(self):
        return None

    def generate(self, user_input: str, system_prompt=None) -> str:
        prompt = system_prompt or ""
        if "strict fact-checker" in prompt:
            return "clean final answer"
        if "Question:" in prompt:
            return "final answer"
        if "extracting factual findings" in prompt:
            return "step findings"

        # Planning response: intentionally repetitive to exercise dedupe + budget.
        steps = [
            {"description": "Retrieve APSX-PIM wiring diagram", "tool_calls": []},
            {"description": "Retrieve APSX-PIM wiring diagram", "tool_calls": []},
            {"description": "Retrieve APSX-PIM wiring diagram", "tool_calls": []},
        ]
        return json.dumps(steps)


class InvalidPlanLLM(LLM):
    def init_client(self):
        return None

    def generate(self, user_input: str, system_prompt=None) -> str:
        prompt = system_prompt or ""
        if "strict fact-checker" in prompt:
            return "verified final"
        if "Question:" in prompt:
            return "fallback final"
        if "extracting factual findings" in prompt:
            return "step findings"

        # Invalid planning output should not crash finalization.
        return "not-json"


def test_planner_respects_budget_and_skips_duplicates(monkeypatch):
    monkeypatch.setenv("MAX_TOTAL_NODES", "8")
    monkeypatch.setenv("MAX_CHILDREN_PER_LEAF", "3")
    monkeypatch.setenv("MAX_DEPTH", "6")

    manager = AgentManager("Show APSX-PIM wiring", StubPlannerLLM("stub"))
    result = manager.run()

    metadata = result["metadata"]
    assert metadata["duplicate_steps_skipped"] > 0
    assert metadata["node_count"] <= 9  # planner budget + final answer node
    assert result["final_answer"]


def test_invalid_plan_still_produces_final_answer(monkeypatch):
    monkeypatch.setenv("MAX_TOTAL_NODES", "8")
    monkeypatch.setenv("MAX_CHILDREN_PER_LEAF", "2")
    monkeypatch.setenv("MAX_DEPTH", "4")

    manager = AgentManager("Show APSX-PIM wiring", InvalidPlanLLM("stub"))
    result = manager.run()

    assert result["final_answer"] is not None
    assert result["metadata"]["termination_reason"] is not None
