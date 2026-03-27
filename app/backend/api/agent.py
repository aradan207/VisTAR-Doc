import asyncio
import json
import math
import os
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.backend.api.tools.manual_search import manual_search
from app.backend.api.tools.image_search import image_search
from app.backend.api.tools.web import fetch_url, web_search
from app.backend.core.agent.agent_manager import AgentManager
from app.backend.core.agent.llm import LLM
from app.backend.core.agent.ollamaLlm import OllamaLLM
from app.backend.core.agent.tool import tool


router = APIRouter(tags=["Agent"])


def _offline_mode_enabled() -> bool:
    """Return True when strict offline mode is enabled.

    OFFLINE_MODE defaults to true so deployments remain local-by-default.
    Set OFFLINE_MODE=false to re-enable web tools in connected environments.
    """
    value = os.getenv("OFFLINE_MODE", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


class AddArgs(BaseModel):
    a: int
    b: int


@tool("add_a_b", AddArgs, "Add two integers and return their sum")
def add_a_b(args: AddArgs) -> dict:
    """Simple example tool used for testing the tool registration pipeline."""
    return {"sum": args.a + args.b}


class AgentRequest(BaseModel):
    query: str
    providerUrl: Optional[str] = None
    modelName: Optional[str] = None
    runBenchmark: Optional[bool] = False


def _format_benchmark_metric(value: Any) -> str:
    """Format benchmark values for the frontend with timeout-safe fallback."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "timeout"
    if math.isnan(num):
        return "timeout"
    return f"{num:.4f}"


def _run_benchmark_for_snapshot(
    *,
    query: str,
    result: Dict[str, Any],
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Run one-sample RAGAS benchmark for an already-computed agent result."""
    from tests.ragas_runner import (
        _clean_answer_for_ragas,
        _extract_contexts,
        _prepare_contexts_for_ragas,
        run_ragas,
    )
    from tests.ragas_config import validate_ragas_prerequisites

    prereq = validate_ragas_prerequisites()
    if not prereq.get("ok", False):
        missing = prereq.get("missing", [])
        detail = "; ".join(str(item) for item in missing) if missing else "Unknown prerequisite failure"
        raise RuntimeError(
            "Benchmark prerequisites not satisfied. "
            f"OFFLINE_MODE={'enabled' if prereq.get('offline_mode') else 'disabled'}. "
            f"Missing: {detail}"
        )

    contexts = _extract_contexts(result)
    prepared_contexts = _prepare_contexts_for_ragas(contexts)
    cleaned_answer = _clean_answer_for_ragas(result.get("final_answer", ""))

    sample = {
        "_id": "dynamic",
        "user_input": query,
        "retrieved_contexts": prepared_contexts,
        "response": cleaned_answer,
        "reference": "dummy ground truth to trigger benchmark bounds",
    }

    enriched = run_ragas(
        [sample],
        max_workers=1,
        on_progress=on_progress,
    )[0]

    return {
        "context_recall": _format_benchmark_metric(enriched.get("context_recall")),
        "faithfulness": _format_benchmark_metric(enriched.get("faithfulness")),
        "factual_correctness": _format_benchmark_metric(enriched.get("factual_correctness")),
        "answer_relevancy": _format_benchmark_metric(enriched.get("answer_relevancy")),
    }


def _build_llm(
    provider_url: Optional[str] = None,
    model_name_override: Optional[str] = None,
) -> LLM:
    """Build the Ollama LLM backend.

    Args:
        provider_url: optional Ollama server URL
        model_name_override: optional model name to use
    """
    model_name = model_name_override or os.getenv("OLLAMA_MODEL", "hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M")
    return OllamaLLM(model_name=model_name, provider_url=provider_url)


def _init_manager(req: AgentRequest) -> AgentManager:
    try:
        llm = _build_llm(
            provider_url=(req.providerUrl or None),
            model_name_override=(req.modelName or None),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to initialize language model: {exc}") from exc

    # Register manual_search first for priority in tool list.
    # In strict offline mode, web tools stay disabled to prevent non-local calls.
    tool_fns = [manual_search, image_search, add_a_b]
    if not _offline_mode_enabled():
        tool_fns.extend([web_search, fetch_url])
    else:
        print("[Agent] OFFLINE_MODE is enabled: web_search and fetch_url are disabled")

    for tool_fn in tool_fns:
        llm.register_decorated_tool(tool_fn)

    return AgentManager(user_input=req.query, llm=llm)


@router.post("/run")
async def run_agent(req: AgentRequest):
    """Run the autonomous research agent for the provided query."""
    manager = _init_manager(req)
    try:
        result = manager.run()
        if req.runBenchmark:
            try:
                result["benchmark_scores"] = _run_benchmark_for_snapshot(
                    query=req.query,
                    result=result,
                )
            except Exception as e:
                print(f"Benchmark failed: {e}")
                result["benchmark_scores"] = {"error": str(e)}
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc


@router.post("/run-stream")
async def run_agent_stream(req: AgentRequest):
    """Stream reasoning tree updates as the agent plans and executes."""

    manager = _init_manager(req)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    loop.call_soon(queue.put_nowait, {"type": "start", "payload": {"query": req.query}})

    def emit_update(snapshot: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "update", "payload": snapshot})

    async def worker() -> None:
        def _run() -> None:
            try:
                result = manager.run(on_update=emit_update)

                metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
                print(
                    "[Agent] Run completed "
                    f"nodes={metadata.get('node_count')} "
                    f"edges={metadata.get('edge_count')} "
                    f"depth={metadata.get('max_depth_reached')} "
                    f"reason={metadata.get('termination_reason')}"
                )

                # Always send final answer as soon as agent reasoning completes.
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "final", "payload": result})

                if req.runBenchmark:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "type": "benchmark_started",
                            "payload": {"benchmark_status": "Running RAGAS evaluation..."},
                        },
                    )

                    def emit_benchmark_progress(stage: str) -> None:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {
                                "type": "benchmark_progress",
                                "payload": {"benchmark_status": stage},
                            },
                        )

                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "type": "benchmark_progress",
                            "payload": {"benchmark_status": "Preparing judge model..."},
                        },
                    )
                    try:
                        result["benchmark_scores"] = _run_benchmark_for_snapshot(
                            query=req.query,
                            result=result,
                            on_progress=emit_benchmark_progress,
                        )
                    except Exception as e:
                        print(f"Benchmark failed: {e}")
                        result["benchmark_scores"] = {"error": str(e)}

                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "type": "benchmark_done",
                            "payload": {
                                "benchmark_scores": result.get("benchmark_scores"),
                            },
                        },
                    )
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "error": str(exc)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})

        await loop.run_in_executor(None, _run)

    asyncio.create_task(worker())

    async def event_generator():
        while True:
            event = await queue.get()
            if event.get("type") == "done":
                break
            yield f"data: {json.dumps(event)}\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
