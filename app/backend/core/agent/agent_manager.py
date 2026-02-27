import json
import re
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


class AgentManager:

    def __init__(self, user_input: str, llm: LLM):
        self.user_input = user_input
        self.reasoning_tree = ReasoningTree(user_input)
        self.llm = llm
        self.final_answer: Optional[str] = None

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
        self.plan(context, parent_leaf_id="leaf_0", on_update=on_update)
        self.finalize(on_update=on_update)
        return self._snapshot()

    def plan(
        self,
        context: str,
        parent_leaf_id: str,
        max_branch_len: int = 5,
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        response = self.llm.generate(context)
        cleaned = strip_json_markdown(response)
        try:
            data = json.loads(cleaned)
            steps = TypeAdapter(List[PlannedStep]).validate_python(data)
        except Exception:
            return
        if not steps:
            return

        for step in steps:
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
            {step.description}

            TOOL RESULTS:
            {tool_results_text}

            What conclusion or synthesis should be recorded for this step?
            """

            result = self.llm.generate(user_input=user_input.strip(), system_prompt=FILL_RESULT_PROMPT.strip())

            new_leaf_id = self.reasoning_tree.add_leaf(
                description=step.description,
                parent_leaf=parent_leaf_id,
                tool_calls=tool_calls,
                result=result
            )

            self._emit_update(on_update, latest_leaf_id=new_leaf_id)

            branch_depth = self.reasoning_tree.get_branch_depth(new_leaf_id)
            if branch_depth >= max_branch_len:
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
