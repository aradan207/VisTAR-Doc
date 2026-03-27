import json
import re
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, TypeAdapter

from app.backend.core.agent.llm import LLM
from app.backend.core.models.tool_calls import ToolCall
from app.backend.core.reasoningTree.reasoning_tree import ReasoningTree

def strip_json_markdown(response: str) -> str:
    return re.sub(r"^```(?:json)?\n|\n```$", "", response.strip())

class PlannedStep(BaseModel):
    description: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)


class PlanFinal(BaseModel):
    answer: str


class PlanResponse(BaseModel):
    description: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    final: Optional[PlanFinal] = None


class AgentManager:

    def __init__(self, user_input: str, llm: LLM):
        self.user_input = user_input
        self.reasoning_tree = ReasoningTree(user_input)
        self.llm = llm
        self.final_answer: Optional[str] = None
        self.max_total_nodes = max(2, int(os.getenv("MAX_TOTAL_NODES", "80")))
        self.max_children_per_leaf = max(1, int(os.getenv("MAX_CHILDREN_PER_LEAF", "2")))
        self.max_depth = max(1, int(os.getenv("MAX_DEPTH", "5")))

        self.generated_steps_total = 0
        self.duplicate_steps_skipped = 0
        self.max_depth_reached = 0
        self.terminated_by_budget = False
        self.terminated_by_convergence = False
        self.termination_reason: Optional[str] = None
        self._seen_signatures_by_parent: Dict[str, set[str]] = {}

    def _edge_count(self) -> int:
        return sum(len(leaf.child_leaves) for leaf in self.reasoning_tree.leaves.values())

    def _snapshot(self, latest_leaf_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "reasoning_tree": self.reasoning_tree.to_dict(),
            "final_answer": self.final_answer,
            "metadata": {
                "query": self.user_input,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "node_count": len(self.reasoning_tree.leaves),
                "edge_count": self._edge_count(),
                "generated_steps_total": self.generated_steps_total,
                "duplicate_steps_skipped": self.duplicate_steps_skipped,
                "max_depth_reached": self.max_depth_reached,
                "terminated_by_budget": self.terminated_by_budget,
                "terminated_by_convergence": self.terminated_by_convergence,
                "termination_reason": self.termination_reason,
                "max_total_nodes": self.max_total_nodes,
                "max_children_per_leaf": self.max_children_per_leaf,
                "max_depth": self.max_depth,
            },
        }

        if latest_leaf_id and latest_leaf_id in self.reasoning_tree.leaves:
            payload["latest_leaf_id"] = latest_leaf_id
            payload["latest_leaf"] = self.reasoning_tree.leaves[latest_leaf_id].to_dict()

        return payload

    def _emit_update(
        self,
        on_update: Optional[Callable[[Dict[str, Any]], None]],
        latest_leaf_id: Optional[str] = None,
    ) -> None:
        if callable(on_update):
            on_update(self._snapshot(latest_leaf_id=latest_leaf_id))

    def run(self, on_update: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        context = self.reasoning_tree.get_reasoning_tree_context()

        try:
            self.plan(context, parent_leaf_id="leaf_0", on_update=on_update)
        except Exception as exc:
            self.termination_reason = self.termination_reason or f"planning_error: {exc}"

        try:
            self.finalize(on_update=on_update)
        except Exception as exc:
            fallback = f"Unable to produce a complete final answer: {exc}"
            self.final_answer = fallback
            final_leaf_id = self.reasoning_tree.add_leaf(
                description="Final answer",
                parent_leaf="leaf_0",
                tool_calls=[],
                result=fallback,
            )
            self._emit_update(on_update, latest_leaf_id=final_leaf_id)

        return self._snapshot()

    def _can_expand(self) -> bool:
        return len(self.reasoning_tree.leaves) < self.max_total_nodes

    def _reserve_budget_or_terminate(self, reason: str) -> bool:
        if self._can_expand():
            return True
        self.terminated_by_budget = True
        if not self.termination_reason:
            self.termination_reason = reason
        return False

    def _step_signature(self, step: PlanResponse) -> str:
        call_signatures = []
        for tc in step.tool_calls:
            call_signatures.append({
                "tool_name": str(tc.get("tool_name", "")).strip().lower(),
                "args": tc.get("args", {}),
            })

        canonical = {
            "description": (step.description or "").strip().lower(),
            "tool_calls": call_signatures,
        }
        return json.dumps(canonical, sort_keys=True, ensure_ascii=True)

    def _is_duplicate_step(self, parent_leaf_id: str, step: PlanResponse) -> bool:
        sig = self._step_signature(step)
        seen = self._seen_signatures_by_parent.setdefault(parent_leaf_id, set())
        if sig in seen:
            return True
        seen.add(sig)
        return False

    def plan(
        self,
        context: str,
        parent_leaf_id: str,
        max_branch_len: Optional[int] = None,
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        if max_branch_len is None:
            max_branch_len = self.max_depth

        if not self._reserve_budget_or_terminate("node_budget_reached_before_plan"):
            return

        current_depth = self.reasoning_tree.get_branch_depth(parent_leaf_id)
        self.max_depth_reached = max(self.max_depth_reached, current_depth)
        if current_depth >= max_branch_len:
            self.terminated_by_convergence = True
            if not self.termination_reason:
                self.termination_reason = "max_depth_reached"
            return

        response = self.llm.generate(context)
        cleaned = strip_json_markdown(response)
        try:
            data = json.loads(cleaned)
            raw_steps = TypeAdapter(List[PlanResponse]).validate_python(data)
        except Exception as exc:
            if not self.termination_reason:
                self.termination_reason = f"invalid_plan_json: {exc}"
            return

        if not raw_steps:
            self.terminated_by_convergence = True
            if not self.termination_reason:
                self.termination_reason = "planner_returned_empty"
            return

        finals = [step.final for step in raw_steps if step.final and step.final.answer.strip()]
        if finals:
            self.terminated_by_convergence = True
            if not self.termination_reason:
                self.termination_reason = "planner_signaled_final"
            return

        steps = raw_steps[: self.max_children_per_leaf]

        for step in steps:
            if not self._reserve_budget_or_terminate("node_budget_reached_mid_plan"):
                return

            if self._is_duplicate_step(parent_leaf_id, step):
                self.duplicate_steps_skipped += 1
                continue

            self.generated_steps_total += 1
            step_description = (step.description or "Execute retrieval step").strip()

            tool_calls = []
            for tc in step.tool_calls:
                if 'tool_name' in tc:
                    tool_calls.append(ToolCall(
                        tool_name=tc['tool_name'],
                        args=tc.get('args', {})  # Default to empty dict if args missing
                    ))
            
            for call in tool_calls:
                try:
                    call.result = self.llm.run_tool(call.tool_name, call.args)
                except Exception as tool_err:
                    # Store the error as the tool result so the agent can
                    # self-correct or continue with remaining reasoning steps
                    # instead of aborting the entire question.
                    call.result = {
                        "error": f"Tool '{call.tool_name}' failed: {tool_err}",
                        "results": [],
                    }

            FILL_RESULT_PROMPT = """
You are extracting factual findings from tool results for a field technician.

RULES:
1. Extract ALL concrete technical facts, specifications, values, procedures, and component names found in the tool results. Be thorough — a technician needs every relevant detail.
2. Do NOT state which tools were called, whether searches succeeded or failed, or mention document counts.
3. Do NOT use phrases like "the search returned", "no results were found", "based on the tool results", "I was unable to find", or any meta-commentary about the retrieval process.
4. If a tool returned no useful information for an aspect, simply omit that aspect — write nothing about it.
5. If image_search returned results with "url" fields, copy them EXACTLY as ![description](url). Never invent URLs.
6. Do NOT add any fact, value, or specification from your own knowledge that is not present in the tool results.
"""
            tool_results_text = "\n".join(f"{call.tool_name}: {call.result}" for call in tool_calls)

            user_input = f"""
            PREVIOUS CONTEXT:
            {context}

            STEP DESCRIPTION:
            {step_description}

            TOOL RESULTS:
            {tool_results_text}

            What conclusion or synthesis should be recorded for this step?
            """

            result = self.llm.generate(user_input=user_input.strip(), system_prompt=FILL_RESULT_PROMPT.strip())

            new_leaf_id = self.reasoning_tree.add_leaf(
                description=step_description,
                parent_leaf=parent_leaf_id,
                tool_calls=tool_calls,
                result=result
            )

            self._emit_update(on_update, latest_leaf_id=new_leaf_id)

            branch_depth = self.reasoning_tree.get_branch_depth(new_leaf_id)
            self.max_depth_reached = max(self.max_depth_reached, branch_depth)
            if branch_depth >= max_branch_len:
                continue

            if len(self.reasoning_tree.leaves.get(parent_leaf_id, self.reasoning_tree.leaves["leaf_0"]).child_leaves) >= self.max_children_per_leaf:
                continue

            enriched_context = self.reasoning_tree.get_leaf_context(new_leaf_id)
            self.plan(
                enriched_context,
                parent_leaf_id=new_leaf_id,
                max_branch_len=max_branch_len,
                on_update=on_update,
            )

    def finalize(self, on_update: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
        """Generate the final answer based on every terminal leaf."""
        leaves = self.reasoning_tree.get_reasoning_tree_context()

        FINAL_PROMPT = f"""
Question: "{self.user_input}"

You are answering a field technician who needs precise, actionable technical information.

Rules:
1. Answer using ONLY facts, specifications, values, and procedures that appear explicitly in the context below. Include ALL relevant technical details — component names, voltage/pressure/temperature values, connector labels, part numbers, procedure steps — that the context provides.
2. Do NOT add any fact, specification, number, material name, or technical detail from your own training data. If a detail is not explicitly written in the context, do not include it. When uncertain, leave it out.
3. You may use markdown formatting: **bold** key values, bullet lists for specifications, headings for sections. Make the answer easy for a technician to scan.
4. If image URLs from localhost:8000 appear in the context, include them as ![description](url) at the end of your answer. Copy URLs exactly — do not modify or invent URLs.
"""

        final_answer = self.llm.generate(
            user_input=leaves,
            system_prompt=FINAL_PROMPT.strip()
        )

        # Self-critique: second LLM call strips claims not supported by context.
        # Catches hallucinated specs (e.g. invented voltages, connector details)
        # that the model adds from training data despite the prompt constraint.
        CRITIQUE_PROMPT = f"""
You are a strict fact-checker for manufacturing documentation. Your job is critical — incorrect specifications can endanger technicians.

Compare the ANSWER below against the CONTEXT. Remove or correct any claim, number, specification, temperature, pressure, voltage, connector description, or technical detail in the answer that is NOT explicitly stated in the context.

Keep:
- All facts that ARE supported by the context
- All image URLs (![...](...)) unchanged
- Markdown formatting

Remove:
- Any specific value (voltage, temperature, pressure, dimensions) not in the context
- Any component description or function not in the context
- Any material name, standard, or procedure not in the context

Output ONLY the cleaned answer. No commentary, no explanation of what you changed.
"""

        critique_input = f"""ANSWER:
{final_answer}

CONTEXT:
{leaves}"""

        final_answer = self.llm.generate(
            user_input=critique_input,
            system_prompt=CRITIQUE_PROMPT.strip()
        )

        # Post-processing: guarantee image URLs from image_search are in the answer.
        # The model sometimes drops valid images during critique.
        final_answer = self._inject_missing_images(final_answer)

        # Strip any hallucinated URLs that don't match our real API endpoint.
        final_answer = self._strip_hallucinated_urls(final_answer)

        final_leaf_id = self.reasoning_tree.add_leaf(
            description="Final answer",
            parent_leaf="leaf_0",
            tool_calls=[],
            result=final_answer
        )
        self.final_answer = final_answer
        if not self.termination_reason:
            self.termination_reason = "finalized"
        self._emit_update(on_update, latest_leaf_id=final_leaf_id)
        return final_answer

    def _strip_hallucinated_urls(self, answer: str) -> str:
        """
        Remove any image markdown where the URL does not match our real API.

        The model sometimes invents URLs like http://example.com/image1 or
        http://localhost:8000/findings-link that don't correspond to any
        real served image.  Only URLs matching the yologen media endpoint
        are kept.
        """
        import re
        import os

        base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        real_prefix = f"{base_url}/api/media/yologen/"

        def _check_image_md(m: re.Match) -> str:
            url = m.group(2)
            if url.startswith(real_prefix):
                return m.group(0)  # keep — real image
            return ""  # strip — hallucinated

        answer = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", _check_image_md, answer)
        # Clean up leftover blank lines from stripped images
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
        return answer

    def _inject_missing_images(self, answer: str) -> str:
        """
        Walk the reasoning tree and collect every image URL returned by
        image_search tool calls.  Any URL not already present in the answer
        text is appended as Markdown so visual questions always include their
        retrieved images, regardless of whether the LLM chose to keep them.

        At most 3 images are appended to avoid cluttering short answers.
        """
        import re

        collected: list = []  # [(url, description), ...]
        seen_urls: set = set()

        for leaf in self.reasoning_tree.leaves.values():
            for tc in leaf.tool_calls:
                if tc.tool_name != "image_search":
                    continue
                result = tc.result or {}
                if not isinstance(result, dict):
                    continue
                for item in result.get("results", []):
                    url = item.get("url", "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    kwords = item.get("keywords")
                    desc = (
                        item.get("description")
                        or item.get("vlm_description")
                        or (kwords[0] if isinstance(kwords, list) and kwords else "image")
                    )
                    collected.append((url, str(desc)[:80]))

        if not collected:
            return answer

        # Identify URLs already in the answer
        existing_urls = set(re.findall(r"https?://\S+", answer))

        missing = [
            (url, desc)
            for url, desc in collected
            if url not in existing_urls
        ][:3]  # cap at 3 new images

        if not missing:
            return answer

        image_block = "\n".join(f"![{desc}]({url})" for url, desc in missing)
        return f"{answer.rstrip()}\n\n{image_block}"
