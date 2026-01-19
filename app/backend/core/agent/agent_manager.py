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
                call.result = self.llm.run_tool(call.tool_name, call.args)

            FILL_RESULT_PROMPT = """
            You are an autonomous reasoning agent summarizing tool results.

            STRICT RULES FOR IMAGE URLs:
            1. If image_search results contain URLs, extract them EXACTLY as written
            2. DO NOT modify, shorten, or make up any URLs
            3. Only use URLs that appear verbatim in "url": "..." fields
            4. Format images as: ![Brief description](exact_url_from_results)
            
            WRONG: ![Image](http://localhost:8000/api/media/yologen/train/APSX-PIM_page37_img1.png)
            (if that exact URL wasn't in the results)
            
            RIGHT: Copy the EXACT url value from the tool results
            
            If no image_search was called or it returned no results, do not include any images.
            
            Respond with a brief summary paragraph. If images were found, include them using the exact URLs from results.
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
You are answering the user's question: "{self.user_input}"

Based on the reasoning tree context below, write a DIRECT answer to the user.

=== CRITICAL IMAGE RULES ===

1. LOOK FOR "url" FIELDS: Search the context for image_search results containing "url": "http://..."

2. COPY URLs EXACTLY: If you find image URLs, copy them CHARACTER FOR CHARACTER
   - CORRECT: Use the exact URL from results like "url": "http://localhost:8000/api/media/yologen/train/APSX-PIM_page41_img1.png"
   - WRONG: Making up or modifying URLs like page37, page20, etc. that weren't in results

3. FORMAT: ![Description based on image_name](exact_url_from_results)

4. NO IMAGES IF NOT FOUND: If image_search returned count: 0 or wasn't called, don't include any image markdown

5. VERIFY BEFORE INCLUDING: Before writing any ![...](...) markdown, confirm that exact URL appeared in the tool results

=== ANSWER FORMAT ===

1. Brief text answer to the user's question
2. Then include images using EXACT urls from image_search results
3. Any additional context or notes

DO NOT invent URLs. DO NOT modify URLs. Only use URLs that appear exactly in the image_search results.
"""

        final_answer = self.llm.generate(
            user_input=leaves,
            system_prompt=FINAL_PROMPT.strip()
        )

        final_leaf_id = self.reasoning_tree.add_leaf(
            description="Final answer",
            parent_leaf="leaf_0",
            tool_calls=[],
            result=final_answer
        )
        self.final_answer = final_answer
        self._emit_update(on_update, latest_leaf_id=final_leaf_id)
        return final_answer
