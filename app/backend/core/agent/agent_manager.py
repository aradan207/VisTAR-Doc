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

=== STEP 1: CHECK WHICH TOOLS WERE CALLED ===

Look at TOOL RESULTS below. Check if "image_search" appears.
- If ONLY "manual_search" was called: Your response must have ZERO images, ZERO URLs, ZERO image suggestions
- If "image_search" was called: Check if it returned results with "url" fields

=== STEP 2: IMAGES ONLY FROM image_search ===

INCLUDE images ONLY if ALL of these are true:
1. The tool "image_search" was actually called (appears in TOOL RESULTS)
2. The results contain "count" > 0
3. The results contain actual "url": "http://..." values

If including images:
- Copy the URL EXACTLY character-for-character from the "url" field
- Format: ![description](exact_url)

=== FORBIDDEN ===

- Do NOT invent URLs based on page numbers from manual_search
- Do NOT guess URL patterns like "machine_page123_img1.png"
- Do NOT suggest "see image at..." unless image_search returned that URL
- Do NOT include any image markdown if image_search wasn't called

Respond with a brief text summary. Only include image markdown if image_search returned actual URLs.
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

=== STEP 1: VERIFY IMAGE_SEARCH WAS CALLED ===

Before writing anything, search the context for "image_search" tool calls.

IF image_search WAS NOT CALLED:
- Your answer must contain ZERO image URLs
- Your answer must contain ZERO ![...](...) markdown
- Your answer must NOT suggest viewing any images
- Just provide a text-only answer based on manual_search results

IF image_search WAS CALLED but returned "count": 0:
- Say "No relevant images were found" in your answer
- Do NOT include any image markdown

=== STEP 2: IF image_search RETURNED RESULTS ===

ONLY if you find actual "url": "http://..." values in image_search results:
1. Copy the URL EXACTLY - character for character
2. Format: ![Brief description](exact_url)
3. Verify the image is from the correct machine (e.g., BOY-35 images for BOY-35 questions)

=== FORBIDDEN - NEVER DO THESE ===

- Do NOT invent URLs like "http://localhost:8000/api/media/yologen/train/MACHINE_page123_img1.png"
- Do NOT construct URLs from page numbers found in manual_search
- Do NOT include image markdown unless image_search returned actual URLs
- Do NOT use "Failed to load:" with made-up URLs

=== ANSWER FORMAT ===

1. Text answer to the user's question
2. IF AND ONLY IF image_search returned URLs: include them with exact URLs
3. Additional context if needed
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
