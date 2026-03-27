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
- **Swappable LLM Providers**: Supports Ollama backends that run entirely offline for both LLM inference and embeddings.

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

For offline or air-gapped deployment, follow [media/offline.md](media/offline.md).

### Want To Use It Offline Later? Do This Once While You Are Online

If you are connected right now and want smooth offline use later, run these once:

```bash
uv run python tests/prepare_offline_semantic_cache.py
uv run python tests/offline_readiness_check.py
```

If readiness is green, you can create a transfer bundle:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/export_offline_bundle.ps1
```

On the offline machine, import and validate:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/import_offline_bundle.ps1
uv run python tests/offline_readiness_check.py
```

Expected result:
- `[Readiness] READY: strict offline prerequisites satisfied`

### Prerequisites

- **Python 3.11+** with pip (Python 3.13 recommended)
- **Ollama** installed and running (download from ollama.com/download)
- **Node.js 18+** with npm (will be installed automatically if missing on Windows)
- **vlm-yolo-detector** repository (optional, for image search)

### Quick Setup (Windows)

For a fully automated installation:

```bash
git clone https://github.com/Manufacturing-Demonstration-Facility/agentic-rag.git
cd agentic-rag
install.bat
```

`install.bat` supports two modes:

- `0`: Online prepare/install mode
- `1`: Offline install/validation mode

At the end of install, the script can optionally run:
- Online mode: `powershell -ExecutionPolicy Bypass -File scripts/export_offline_bundle.ps1`
- Offline mode: `powershell -ExecutionPolicy Bypass -File scripts/import_offline_bundle.ps1`

This will automatically:
1. Install uv package manager and Python dependencies
2. Install Node.js (if needed) and frontend dependencies
3. Pull all required Ollama models
4. Clone vlm-yolo-detector repository (for image search)
5. Build the FAISS index from PDF manuals
6. Verify the installation

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

This installs FastAPI, Ollama SDK, FAISS (vector database), pypdf, and other dependencies.

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

# Pull the embedding model for RAG (required, 215 MB)
ollama pull hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M
```

**Available LLM Models:**

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| Ministral-3-8B-Instruct-2512 | 6.1 GB | Moderate | Best quality reasoning (default) |
| Mistral-7B-Instruct-v0.3 | 4.4 GB | Fast | Lighter alternative, lower resource usage |

To switch models, update `OLLAMA_MODEL` in your `.env` file.

If needed, create your local environment file from the template:

```bash
copy .env.example .env
```

#### 5. Configuration

The `.env` file is already configured with defaults for Ollama:

```env
# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_MODEL=hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M

# Embedding Model
OLLAMA_EMBED_MODEL=hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M

# Strict offline controls
OFFLINE_MODE=true
REQUIRE_IMAGE_RETRIEVAL=true
REQUIRE_SEMANTIC_MODEL_CACHE=true

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

### Running the Application

#### Option A: Use the Startup Script (Windows)

```bash
start.bat
```

This launches both backend and frontend in separate terminal windows.

#### Option B: Manual Start

**Terminal 1 - Backend**:
```bash
uv run uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
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
| LLM_PROVIDER | LLM backend: ollama, openai, or mistral | ollama |
| OLLAMA_MODEL | Ollama model name | Ministral-3-8B-Instruct-2512 |
| OLLAMA_HOST | Ollama server URL | http://localhost:11434 |
| OLLAMA_EMBED_MODEL | Ollama embedding model for RAG | mxbai-embed-large-v1-gguf:Q4_K_M |
| MANUALS_PATH | Path to PDF documents directory | data/manuals |
| INDEX_PATH | Path to FAISS index storage | data/faiss_index |
| CHUNK_SIZE | Characters per document chunk | 800 |
| CHUNK_OVERLAP | Overlap between chunks | 150 |

### Manual Search Priority

The agent is configured to prioritize local documents over web search:

1. **Automatic Prioritization**: The system prompt instructs the LLM to use manual_search FIRST for questions about equipment, specifications, procedures, troubleshooting, and manufacturing processes.

2. **Fallback Behavior**: Only if manual_search returns no relevant results will the agent use web_search.

3. **Citation Tracking**: Results from manual_search include source document filename, page number, relevance score, and text excerpt.

## Extending the Agent with Custom Tools

Tools are Python functions decorated with `@tool`:

```python
from pydantic import BaseModel
from app.backend.core.agent.tool import tool

class WeatherArgs(BaseModel):
    location: str

@tool("get_weather", WeatherArgs, "Retrieve the current weather for a city")
def get_weather(args: WeatherArgs) -> dict:
    return {"conditions": "sunny", "location": args.location}
```

Register the tool on the LLM wrapper before running the agent:

```python
llm.register_decorated_tool(get_weather)
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

## Project Structure

```
agentic-rag/
├── app/
│   ├── backend/
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── api/
│   │   │   ├── agent.py            # Agent endpoint and tool registration
│   │   │   └── tools/              # Tool implementations
│   │   │       ├── image_search.py # Semantic image search
│   │   │       ├── manual_search.py# RAG document search
│   │   │       └── web.py          # Web search fallback
│   │   └── core/
│   │       ├── agent/              # Agent logic and LLM wrappers
│   │       ├── models/             # Pydantic models
│   │       ├── rag/                # RAG pipeline components
│   │       └── reasoningTree/      # Reasoning tree implementation
│   └── frontend/
│       └── agent-frontend/         # Web UI
├── data/
│   ├── faiss_index/                # Built FAISS index
│   └── manuals/                    # PDF documents
├── tests/                          # Test suite
├── install.bat                     # Automated installation script
├── start.bat                       # Application startup script
└── pyproject.toml                  # Python dependencies
```

## Related Repositories

- **vlm-yolo-detector**: Companion repository for image extraction and semantic search. Required for image search functionality.
  - Repository: https://github.com/morkev/vlm-yolo-detector
  - Contains: Image extraction scripts, VLM description generation, semantic embeddings
