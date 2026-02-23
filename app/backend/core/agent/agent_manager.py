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
You are extracting factual findings from tool results.

RULES:
1. Extract ONLY concrete technical facts, specifications, and procedures found in the tool results.
2. Do NOT state which tools were called, whether searches succeeded or failed, or mention document counts.
3. Do NOT use phrases like "the search returned", "no results were found", "based on the tool results", "I was unable to find", or any meta-commentary about the retrieval process.
4. If a tool returned no useful information for an aspect, simply omit that aspect — write nothing about it.
5. If image_search returned results with "url" fields, copy them EXACTLY as ![description](url). Never invent URLs.
6. Keep your response concise — 2–4 sentences of factual content only.
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

Write a DIRECT, FACTUAL answer using ONLY the information in the context below.

=== ANSWER RULES ===

1. Start your answer immediately — no preambles like "Based on your request", "Here is a summary", "It seems like", "The context mentions", or "According to the manual".
2. Do NOT mention tools, searches, documents, leaf IDs, tree structures, step descriptions, retrieval metadata, or document counts.
3. Do NOT narrate what was or was not found — just state the facts directly.
4. Do NOT add information the question did not ask for. Match the scope and length of the question.
5. Keep your answer concise: 1–3 sentences for definitions and facts, up to 5 sentences for procedures.
6. CRITICAL: Answer ONLY using facts that appear explicitly in the context. Do NOT add domain knowledge, elaborations, adjacent concepts, or inferences from your training data. If a fact is not stated in the context, do not include it.

=== IMAGES ===

- If the context contains image URLs (http://...) from image_search results, you MUST include them as ![description](exact_url).
- Copy image URLs EXACTLY character-for-character including the file extension (.png). Do NOT shorten or modify URLs.
- ONLY include images that appear in the context. Never invent or construct URLs.
"""

        final_answer = self.llm.generate(
            user_input=leaves,
            system_prompt=FINAL_PROMPT.strip()
        )

        # Self-critique pass: strip claims not supported by the retrieved context.
        # This removes domain-knowledge elaborations the model adds beyond the context.
        CRITIQUE_PROMPT = """
You are a strict fact-checker. Your only job is to remove unsupported claims from an answer.

RULES:
1. Keep every claim that is directly stated or closely paraphrased from the CONTEXT.
2. Remove every claim that adds domain knowledge, elaboration, or inference NOT present in the CONTEXT.
3. Keep all image markdown (![description](url)) unchanged.
4. Do NOT add new content. Do NOT rewrite existing claims. Only remove unsupported ones.
5. Output ONLY the refined answer. No commentary, no preamble, no explanation.
"""
        critique_input = f"""QUESTION: {self.user_input}

CONTEXT (source of truth):
{leaves}

ANSWER TO REVIEW:
{final_answer}

Return only the refined answer with unsupported claims removed."""

        final_answer = self.llm.generate(
            user_input=critique_input.strip(),
            system_prompt=CRITIQUE_PROMPT.strip()
        )

        # Post-processing: guarantee image URLs from image_search are in the answer.
        # The 3B model sometimes drops valid images despite the prompt instruction.
        final_answer = self._inject_missing_images(final_answer)

        final_leaf_id = self.reasoning_tree.add_leaf(
            description="Final answer",
            parent_leaf="leaf_0",
            tool_calls=[],
            result=final_answer
        )
        self.final_answer = final_answer
        self._emit_update(on_update, latest_leaf_id=final_leaf_id)
        return final_answer

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
