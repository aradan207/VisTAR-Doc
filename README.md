# MDF Agentic Manufacturing Assistant

RAG (Retrieval-Augmented Generation) pipeline with a multi-step reasoning agent. The system integrates planning, tool execution, and local document retrieval to generate comprehensive responses. Provides full transparency by rendering the agent's logic as an interactive Directed Acyclic Graph (DAG) in a web UI.

![UI](media/ui.png)

## Key Features

- **RAG-powered Document Search**: Search through local PDF manuals using semantic embeddings via Ollama. The agent prioritizes local knowledge over web search, with the latter as a fallback.
- **Image Search Integration**: Semantic image search powered by VLM descriptions from the companion vlm-yolo-detector repository.
- **Autonomous Planning Loop**: The AgentManager proposes reasoning steps, executes required tools, and records results in a visual reasoning tree.
- **Tool Registration Framework**: Tools are Pydantic-validated callables decorated with `@tool`, making it easy to add new capabilities at runtime.
- **Interactive Reasoning Tree UI**: Real-time visualization showing the agent's thought process, tool calls, dependencies, and results.
- **Web Research Fallback**: DuckDuckGo Lite search and HTML content extraction for information not in local manuals.
- **Final Report Generation**: After traversing the reasoning tree, the agent synthesizes a professional answer with citations from local documents.
- **Local-First Runtime**: Uses Ollama for generation and primary RAG embeddings, with local sentence-transformer and cross-encoder caches for strict offline agentic semantic workflows.

## System Overview

### RAG Pipeline Details

1. **Document Ingestion**: PDFs in `data/manuals/` are chunked (800 chars with 150 char overlap)
2. **Embedding Generation**: Ollama embeddings (`mxbai-embed-large-v1-gguf:Q4_K_M`) convert text to vectors
3. **Vector Storage**: FAISS index stores embeddings with metadata (source file, page number)
4. **Query Processing**: User queries are embedded and searched against the vector store
5. **Result Ranking**: Top-K most relevant chunks are returned with similarity scores and citations
6. **Tool Priority**: Agent uses `manual_search` FIRST before `web_search`

### Image Search Pipeline

The image search feature requires the vlm-yolo-detector repository:

1. **VLM Descriptions**: Images extracted from PDFs are described using LLaVA via Ollama
2. **Semantic Embeddings**: Descriptions are converted to 384-dim embeddings
3. **FAISS Search**: User queries find relevant images by semantic similarity

## Frontend UI Features

### Reasoning Tree Visualization
- **Interactive Graph**: Pan, zoom, and explore the agent's multi-step reasoning as a DAG
- **Node Types**: 
  - Planning Nodes (blue): Agent's thought process and strategy
  - Tool Call Nodes (green): Executed tools with arguments
  - Result Nodes (yellow): Tool outputs and retrieved information
  - Final Answer Node (purple): Synthesized response
- **Dependencies**: Arrows show which steps depend on previous results
- **Minimap**: Bird's-eye view for navigating large reasoning trees

### Node Inspector Panel
- **Detailed View**: Click any node to see full details (description, tool arguments, results, timestamps)
- **Citations**: For manual_search results, see source document names and page numbers
- **JSON Export**: Copy node data for debugging or documentation

### Tabs Interface
- **Tree Tab**: Main reasoning graph visualization
- **Final Answer Tab**: Formatted final response with citations
- **Raw Data Tab**: Complete JSON response from the API

## Getting Started

This README section explains exactly what the offline scripts do when new users run them, and what changes when your manuals or image artifacts change.

### New User Path

1. **Online local usage only**: run `install.bat` in mode `0`, then run `start.bat`.
2. **Preparing for offline transfer**: run `install.bat` in mode `0`, confirm readiness, then export bundle with `scripts/export_offline_bundle.ps1`.
3. **Receiving on offline machine**: run `install.bat` in mode `1`, import bundle with `scripts/import_offline_bundle.ps1`, run readiness check, then start.

### Prerequisites

- **Python 3.11+** with pip (Python 3.13.5 recommended, any 3.15 version is ideal)
- **Ollama** installed and running (download from ollama.com/download)
- **Node.js 18+** with npm (auto-installed only in online mode)
- **vlm-yolo-detector** repository (optional, for image search)

### Quick Setup (Windows)

For an automated installation:

```bash
git clone https://github.com/Manufacturing-Demonstration-Facility/agentic-rag.git
cd agentic-rag
install.bat
```

`install.bat` supports two modes:

- `0`: Online prepare/install mode
- `1`: Offline install/validation mode

What `install.bat` does in **mode 0**:
1. Installs/updates `uv` and Python dependencies.
2. Installs frontend dependencies (and Node.js if needed).
3. Pulls Ollama models.
4. Attempts semantic cache warmup.
5. Builds FAISS index from `data/manuals`.
6. Runs offline readiness check.
7. Optionally prompts to export an offline bundle.

What `install.bat` does in **mode 1**:
1. Validates pre-existing dependencies and models.
2. Validates required local artifacts for offline execution.
3. Runs readiness checks.
4. Optionally prompts to import an offline bundle.

Mode `1` does not download missing online resources.

### Offline Scripts and Readiness Tools (Why They Exist)

| File | Why it exists | When to run | Must create per user? |
|------|---------------|-------------|-------------------------|
| `scripts/export_offline_bundle.ps1` | Packages offline transfer artifacts (`.cache`, `data/faiss_index`, and sibling `vlm-yolo-detector/data/processed`) | On a connected machine before transferring to air-gapped machine | No, reuse as-is |
| `scripts/import_offline_bundle.ps1` | Unpacks transfer bundle into workspace | On the target offline machine | No, reuse as-is |
| `tests/prepare_offline_semantic_cache.py` | Pre-downloads sentence-transformer and cross-encoder cache needed for strict offline mode | On a connected machine before transfer | No, reuse as-is |
| `tests/offline_readiness_check.py` | Verifies strict prerequisites: FAISS files, Ollama models, image artifacts, semantic cache | After setup, after import, and before startup troubleshooting | No, reuse as-is |

These files are part of the repository and are not generated per user.

Online prep commands:

```bash
uv run python tests/prepare_offline_semantic_cache.py
uv run python tests/offline_readiness_check.py
powershell -ExecutionPolicy Bypass -File scripts/export_offline_bundle.ps1
```

Offline target commands:

```bash
powershell -ExecutionPolicy Bypass -File scripts/import_offline_bundle.ps1
uv run python tests/offline_readiness_check.py
```

Expected readiness success:
- `[Readiness] READY: strict offline prerequisites satisfied`

### Cache Roots for Deterministic Offline Deployments

If you want deterministic local cache locations (recommended for transfer workflows), set these in `.env` **before** running cache warmup and install:

```env
# Repo-local semantic cache roots for deterministic offline deployment.
HF_HOME=C:/Users/mdfar/Repositories/agentic-rag/.cache/huggingface
SENTENCE_TRANSFORMERS_HOME=C:/Users/mdfar/Repositories/agentic-rag/.cache/torch/sentence_transformers
```

When to set these values:
1. **Online prep machine**: set before `tests/prepare_offline_semantic_cache.py` so caches are created in known paths.
2. **Offline target machine**: keep the same values if you imported the same repo-local `.cache` structure.

If you do not set them, default cache discovery still works, but transfer paths are less predictable.

### Manual Setup Steps

#### 1. Clone and Navigate

```bash
git clone https://github.com/Manufacturing-Demonstration-Facility/agentic-rag.git
cd agentic-rag
```

#### 2. Python Dependencies

```bash
pip install uv
uv sync
```

If `uv sync` fails in constrained environments, retry with:

```bash
uv sync --python-preference only-system --native-tls
```

This installs FastAPI, Ollama SDK, FAISS, pypdf, and related dependencies.

#### 3. Frontend Dependencies

```bash
cd app/frontend/agent-frontend
npm install
cd ../../..
```

#### 4. Ollama Models

Pull the required models for LLM reasoning and RAG embeddings:

```bash
# Pull the default LLM model for reasoning (6.1 GB)
ollama pull hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M

# Pull the alternative lighter LLM model (4.4 GB)
ollama pull hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M

# Pull the dedicated RAGAS judge model (4.9 GB)
ollama pull hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M

# Pull the embedding model for RAG (required, 215 MB)
ollama pull hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M
```

**Available LLM Models:**

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| Ministral-3-8B-Instruct-2512 | 6.1 GB | Moderate | Best quality reasoning (default) |
| Mistral-7B-Instruct-v0.3 | 4.4 GB | Fast | Lighter alternative, lower resource usage |
| Meta-Llama-3.1-8B-Instruct | 4.9 GB | Moderate | Dedicated RAGAS judge model |

To switch models, update `OLLAMA_MODEL` in your `.env` file.
For RAGAS judge selection, update `RAGAS_JUDGE_MODEL` in `.env`.

If needed, create your local environment file from the template:

```bash
copy .env.example .env
```

#### 5. Configuration

The `.env` file is already configured with defaults for Ollama:

```env
# LLM Configuration
OLLAMA_MODEL=hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M

# Dedicated RAGAS judge model
RAGAS_JUDGE_MODEL=hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M

# Embedding Model
OLLAMA_EMBED_MODEL=hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M

# Strict offline controls
OFFLINE_MODE=true
REQUIRE_IMAGE_RETRIEVAL=true
REQUIRE_SEMANTIC_MODEL_CACHE=true

# Optional offline cache roots
HF_HOME=C:/Users/mdfar/Repositories/agentic-rag/.cache/huggingface
SENTENCE_TRANSFORMERS_HOME=C:/Users/mdfar/Repositories/agentic-rag/.cache/torch/sentence_transformers

# Planner safety controls
MAX_TOTAL_NODES=80
MAX_CHILDREN_PER_LEAF=2
MAX_DEPTH=5

# RAG Configuration
MANUALS_PATH=data/manuals
INDEX_PATH=data/faiss_index
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

Notes:
1. Keep `OFFLINE_MODE=true` for strict local operation (web tools disabled).
2. Keep `REQUIRE_IMAGE_RETRIEVAL=true` if image retrieval is a hard requirement.
3. Keep `REQUIRE_SEMANTIC_MODEL_CACHE=true` to block accidental network fetches in offline mode.

#### 6. Add Your PDF Manuals

Place your PDF files in the `data/manuals/` directory:

```bash
cp /path/to/your/manuals/*.pdf data/manuals/
```

#### 7. Build the FAISS Index

Generate embeddings and create the searchable vector database:

```bash
uv run python -m app.backend.core.rag.indexer
```

This will:
- Load all PDFs from `data/manuals/`
- Chunk documents (800 chars with 150 char overlap)
- Generate embeddings using Ollama
- Build FAISS index with vectors
- Save to `data/faiss_index/`

**Force rebuild** (if you add/update manuals):

```bash
uv run python -m app.backend.core.rag.indexer --force
```

### Setting Up Image Search (Optional)

For image search functionality, set up the vlm-yolo-detector repository:

```bash
cd ..
git clone https://github.com/morkev/vlm-yolo-detector.git
cd vlm-yolo-detector
install.bat
```

The image search tool will automatically find embeddings at `../vlm-yolo-detector/data/processed/`.

Required image artifacts in that folder:
1. `image_index.json`
2. `image_embeddings.npy`
3. `embedding_mapping.json`
4. `images/`

### When Manuals or Image Data Change

If you keep the current data, you do not need to regenerate offline scripts. The scripts are reusable utilities.

If you add or replace PDF manuals:
1. Update files in `data/manuals/`.
2. Rebuild index: `uv run python -m app.backend.core.rag.indexer --force`.
3. Run readiness check again.
4. If you deploy offline, export a fresh bundle and re-import on target machines.

If you regenerate image artifacts in the companion repository:
1. Rebuild artifacts in `../vlm-yolo-detector/data/processed/`.
2. Confirm required image files listed above exist.
3. Re-export offline bundle and re-import on offline targets.

What does **not** change when data changes:
1. `scripts/export_offline_bundle.ps1`
2. `scripts/import_offline_bundle.ps1`
3. `tests/offline_readiness_check.py`
4. `tests/prepare_offline_semantic_cache.py`

You rerun them, you do not regenerate them.

### Running the Application

#### Option A: Use the Startup Script (Windows)

```bash
start.bat
```

This launches both backend and frontend in separate terminal windows.

#### Option B: Manual Start

**Terminal 1 - Backend**:
```bash
uv run --python python uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend**:
```bash
cd app/frontend/agent-frontend
npm start
```

### Access the Application

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OLLAMA_MODEL | Ollama model name | Ministral-3-8B-Instruct-2512 |
| OLLAMA_HOST | Ollama server URL | http://localhost:11434 |
| OLLAMA_EMBED_MODEL | Ollama embedding model for RAG | mxbai-embed-large-v1-gguf:Q4_K_M |
| OFFLINE_MODE | Disables web tools for strict local mode | true |
| REQUIRE_IMAGE_RETRIEVAL | Block startup if image artifacts are missing | true |
| REQUIRE_SEMANTIC_MODEL_CACHE | Block startup if semantic cache is missing | true |
| HF_HOME | HuggingFace cache root override | repo `.cache/huggingface` when set |
| SENTENCE_TRANSFORMERS_HOME | Sentence-transformers cache root override | repo `.cache/torch/sentence_transformers` when set |
| YOLOGEN_ROOT | Optional override to companion repo root | unset |
| YOLOGEN_DATA_ROOT | Optional override to processed image data root | unset |
| MANUALS_PATH | Path to PDF documents directory | data/manuals |
| INDEX_PATH | Path to FAISS index storage | data/faiss_index |
| CHUNK_SIZE | Characters per document chunk | 800 |
| CHUNK_OVERLAP | Overlap between chunks | 150 |
| MAX_TOTAL_NODES | Planner hard cap per run | 80 |
| MAX_CHILDREN_PER_LEAF | Planner branch fan-out cap | 2 |
| MAX_DEPTH | Planner depth cap | 5 |

### Manual Search Priority

The agent is configured to prioritize local documents over web search:

1. **Automatic Prioritization**: The system prompt instructs the LLM to use manual_search FIRST for questions about equipment, specifications, procedures, troubleshooting, and manufacturing processes.

2. **Fallback Behavior**: Only if manual_search returns no relevant results will the agent use web_search.

3. **Citation Tracking**: Results from manual_search include source document filename, page number, relevance score, and text excerpt.

## Extending the Agent with Custom Tools

Tools are Python functions decorated with `@tool`:

```python
from pydantic import BaseModel, Field
from app.backend.core.agent.tool import tool

class ConnectorPinoutArgs(BaseModel):
    machine_model: str = Field(..., description="Machine model, for example APSX-PIM")
    connector_id: str = Field(..., description="Connector label, for example J1, J2, or J3")

@tool(
    "lookup_connector_pinout",
    ConnectorPinoutArgs,
    "Lookup local connector pinout details for manufacturing troubleshooting and wiring support",
)
def lookup_connector_pinout(args: ConnectorPinoutArgs) -> dict:
    pinout_db = {
        "APSX-PIM": {
            "J1": {"role": "power and communication", "manual_page": 41},
            "J2": {"role": "motor drives and limits", "manual_page": 41},
            "J3": {"role": "temperature and heater outputs", "manual_page": 41},
        }
    }

    model_key = args.machine_model.upper()
    connector_key = args.connector_id.upper()
    connectors = pinout_db.get(model_key, {})
    info = connectors.get(connector_key)

    if info is None:
        return {
            "error": f"No pinout found for {args.machine_model} {args.connector_id}",
            "available_connectors": sorted(connectors.keys()),
            "results": [],
        }

    return {
        "machine_model": model_key,
        "connector_id": connector_key,
        "results": [
            {
                "role": info["role"],
                "manual_page": info["manual_page"],
                "source": "local_pinout_catalog",
            }
        ],
    }
```

Register the tool on the LLM wrapper before running the agent:

```python
llm.register_decorated_tool(lookup_connector_pinout)
```

## Testing

Run tests from the project root:

```bash
# Test RAG search functionality
uv run python tests/test_rag.py

# Test image search
uv run python tests/test_image_search.py

# Test end-to-end flow
uv run python tests/test_e2e.py
```

## Troubleshooting

### Index Not Found Error

**Problem**: FileNotFoundError: Index files not found at data/faiss_index

**Solution**: Build the index first:
```bash
uv run python -m app.backend.core.rag.indexer
```

### Ollama Connection Error

**Problem**: Failed to generate embedding: connection refused

**Solution**: 
1. Check Ollama is running: `ollama list`
2. Start Ollama service if needed
3. Verify embedding model is pulled: `ollama pull hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M`

### Empty Search Results

**Problem**: manual_search returns 0 results for known topics

**Solutions**:
1. Verify PDFs are in `data/manuals/`
2. Rebuild index: `uv run python -m app.backend.core.rag.indexer --force`
3. Run test script: `uv run python tests/test_rag.py`

### Image Search Not Working

**Problem**: image_search returns no results

**Solutions**:
1. Verify vlm-yolo-detector is cloned alongside this repository
2. Check that `../vlm-yolo-detector/data/processed/image_embeddings.npy` exists
3. Run `install.bat` in the vlm-yolo-detector directory if embeddings are missing

## Second Required Repository

- **vlm-yolo-detector**: Companion repository for image extraction and semantic search. Required for image search functionality.
  - Repository: https://github.com/morkev/vlm-yolo-detector
  - Contains: Image extraction scripts, VLM description generation, semantic embeddings
