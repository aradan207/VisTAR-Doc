# Offline Deployment Runbook

This guide describes how to run agentic-rag in air-gapped or internet-restricted environments while preserving text and image retrieval behavior.

## Goals

- Keep Ollama as the inference and embedding provider.
- Keep image retrieval behavior unchanged.
- Prevent runtime internet calls from the agent in offline mode.

## Runtime Offline Controls

- `OFFLINE_MODE=true` disables `web_search` and `fetch_url` tool usage at runtime.
- The default in `.env` is offline-enabled.
- `REQUIRE_IMAGE_RETRIEVAL=true` blocks startup if image artifacts are missing in offline mode.
- `REQUIRE_SEMANTIC_MODEL_CACHE=true` blocks startup if sentence-transformer caches are missing in offline mode.
- `MAX_TOTAL_NODES=80` caps total planning nodes per run to prevent runaway loops.
- `MAX_CHILDREN_PER_LEAF=2` limits fan-out per planning step.
- `MAX_DEPTH=5` keeps branch recursion bounded.
- `YOLOGEN_ROOT` and `YOLOGEN_DATA_ROOT` can override image artifact discovery paths.

## Required Local Artifacts

### Core (required)

- Python dependencies installed from `pyproject.toml`/`uv.lock`
- Node dependencies installed for `app/frontend/agent-frontend`
- Ollama models:
  - `hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M`
  - `hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M` (optional alternative)
  - `hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M`
- Built FAISS index under `data/faiss_index`

### Benchmark/RAGAS (optional)

For benchmark mode (`runBenchmark=true`) and `tests/ragas_runner.py` in offline environments:

- Ollama judge model loaded locally:
  - `hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M` (or your configured `RAGAS_JUDGE_MODEL`)
- Local sentence-transformers cache for:
  - `all-MiniLM-L6-v2`

If these are missing while offline, benchmark mode now fails early with explicit diagnostics.

### Image retrieval (required when `REQUIRE_IMAGE_RETRIEVAL=true`)

From sibling repository `../vlm-yolo-detector/data/processed/`:

- `image_index.json`
- `image_embeddings.npy`
- `embedding_mapping.json`
- `images/` directory

## Online Preparation (connected machine)

1. Clone repositories and run normal install once.
2. Ensure frontend dependencies are installed (`npm install` in `app/frontend/agent-frontend`).
3. Pull required Ollama models.
4. Warm sentence-transformer caches:

```bash
uv run python tests/prepare_offline_semantic_cache.py
```

5. Build FAISS index:

```bash
uv run python -m app.backend.core.rag.indexer
```

6. If image retrieval is required, prepare `vlm-yolo-detector` processed artifacts.
7. Transfer workspace and required model/cache artifacts to offline target.

## Offline Installation (air-gapped target)

Use `install.bat` and choose offline mode (`1`) when prompted.

Offline mode should:

- Skip network-only actions.
- Validate required local artifacts.
- Report actionable missing prerequisites.

## Verification Checklist

1. Start backend and confirm startup logs indicate `OFFLINE_MODE=enabled`.
2. Verify text query returns an answer without image markdown for text-only prompts.
3. Verify image query returns image URLs if image artifacts are present.
4. Verify frontend markdown renders (local `marked` dependency available).

## Suggested Validation Commands

```bash
uv run pytest tests/ --collect-only -q
uv run pytest tests/test_no_images.py -q
uv run pytest tests/test_image_queries.py -q
uv run pytest tests/test_combined_queries.py -q
```

### Benchmark/RAGAS Prerequisite Validation

```bash
uv run python -c "import json; from tests.ragas_config import validate_ragas_prerequisites; print(json.dumps(validate_ragas_prerequisites(), indent=2))"
```

Expected offline-ready output:

- `"ok": true`
- `"missing": []`

### Full Strict Offline Readiness Check

```bash
uv run python tests/offline_readiness_check.py
```

Expected:

- `[Readiness] READY: strict offline prerequisites satisfied`

### One-Command Transfer Bundle

On connected machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/export_offline_bundle.ps1
```

On offline machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/import_offline_bundle.ps1
```

Then run readiness check again before startup.

## Troubleshooting

### Frontend markdown not rendering

- Verify `app/frontend/agent-frontend/node_modules/marked/lib/marked.umd.js` exists.
- Run `npm install` in `app/frontend/agent-frontend` on a connected machine and transfer dependencies as needed.

### No image results

- Verify `../vlm-yolo-detector/data/processed/` contains the required files.
- Check startup warnings from backend for missing image artifacts.

### RAGAS/benchmark in offline mode

- Benchmark is optional for runtime.
- First use of uncached sentence-transformer artifacts may require internet on a connected prep machine.
- Validate prerequisites before long runs using the command above.
- Minimal smoke test:

```bash
uv run python tests/ragas_runner.py --text-only --sample 1
```
