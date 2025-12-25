# MDF's Agentic Manufacturing Assistant

**RAG (Retrieval-Augmented Generation)** pipeline with a multi-step reasoning agent. The system has a complex planning, tool integration, and local document retrieval, to generate a final response. Provides full transparency by rendering the agent's logic as an interactive Directed Acyclic Graph (DAG) in a web UI.

![UI](media/ui.png)

## Key Features

- **RAG-powered Document Search**: Search through local PDF manuals (e.g., manufacturing equipment, technical documentation) using semantic embeddings via Ollama. The agent automatically prioritizes local knowledge over web search, but has the later as a fallback.
- **Autonomous Planning Loop**: The `AgentManager` asks the LLM to propose reasoning steps, executes required tools, and records results in a visual reasoning tree.
- **Tool Registration Framework**: Tools are Pydantic-validated callables decorated with `@tool`, making it trivial to add new capabilities at runtime.
- **Interactive Reasoning Tree UI**: Real-time visualization showing the agent's thought process, tool calls, dependencies, and results in an explorable graph.
- **Web Research Helpers**: Web search uses DuckDuckGo Lite search and HTML content extraction for information not in local manuals.
- **Final Report Generation**: After traversing the reasoning tree, the agent synthesizes a professional answer with citations from local documents.
- **Swappable LLM Providers**: You can swap the LLM provider, but as of now we are keeping Ollama backends via environment variables, which can run entirely offline for both LLM inference and embeddings.

## System Overview

will add a diagram here later

### RAG Pipeline Details

1. **Document Ingestion**: PDFs in `data/manuals/` are chunked (800 chars with 150 char overlap)
2. **Embedding Generation**: Ollama embeddings (`mxbai-embed-large-v1-gguf:Q4_K_M`) convert text to vectors
3. **Vector Storage**: FAISS index stores embeddings with metadata (source file, page number)
4. **Query Processing**: User queries are embedded and searched against the vector store
5. **Result Ranking**: Top-K most relevant chunks are returned with similarity scores and citations
6. **Tool Priority**: Agent is instructed to use `manual_search` FIRST before `web_search`

## Frontend UI Features

The interactive web UI provides a comprehensive view of the agent's reasoning process:

### Reasoning Tree Visualization
- **Interactive Graph**: Pan, zoom, and explore the agent's multi-step reasoning as a directed acyclic graph (DAG)
- **Node Types**: 
  - **Planning Nodes** (blue): Show the agent's thought process and strategy
  - **Tool Call Nodes** (green): Display executed tools with arguments
  - **Result Nodes** (yellow): Show tool outputs and retrieved information
  - **Final Answer Node** (purple): Contains the synthesized response
- **Dependencies**: Arrows show which steps depend on previous results
- **Minimap**: Bird's-eye view for navigating large reasoning trees

### Node Inspector Panel
- **Detailed View**: Click any node to see full details (description, tool arguments, results, timestamps)
- **Citations**: For `manual_search` results, see source document names and page numbers
- **JSON Export**: Copy node data for debugging or documentation

### Tabs Interface
- **Tree Tab**: Main reasoning graph visualization
- **Final Answer Tab**: Formatted final response with citations
- **Raw Data Tab**: Complete JSON response from the API

### Visual Indicators
- **Color Coding**: Nodes colored by type (planning, tool execution, results, final answer)
- **Status Icons**: Success/error indicators for tool executions
- **Relevance Scores**: For RAG results, visual representation of similarity scores

## Getting Started

### Prerequisites

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **Ollama** (for offline operation)

### Setup Steps

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

If using Ollama (recommended for offline operation), pull the required models:

```bash
# Pull the LLM model for reasoning
ollama pull hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M

# Pull the embedding model for RAG
ollama pull hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M
```

Verify models are installed:
```bash
ollama list
```

#### 5. Configuration

The `.env` file is already configured with sensible defaults for Ollama:

```env
# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_MODEL=hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M

# Embedding Model (for RAG)
OLLAMA_EMBED_MODEL=hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M

# RAG Configuration
MANUALS_PATH=data/manuals
INDEX_PATH=data/faiss_index
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

For OpenAI or Mistral, update:
```env
LLM_PROVIDER=openai  # or mistral
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o
```

### Setting Up Your Document Knowledge Base

#### 6. Add Your PDF Manuals

Place your PDF files in the `data/manuals/` directory:

```bash
# Example: Copy your manuals
cp /path/to/your/manuals/*.pdf data/manuals/
```

The system supports technical documentation, equipment manuals, procedures, specifications, etc.

#### 7. Build the FAISS Index

Generate embeddings and create the searchable vector database:

```bash
uv run python -m app.backend.core.rag.indexer
```

This will:
- Load all PDFs from `data/manuals/`
- Chunk documents (800 chars with 150 char overlap)
- Generate embeddings using Ollama
- Build FAISS index with ~5,000+ vectors (for 30 manuals)
- Save to `data/faiss_index/`

**Time estimate**: ~2-5 minutes for 30 PDFs depending on your hardware.

**Force rebuild** (if you add/update manuals):
```bash
uv run python -m app.backend.core.rag.indexer --force
```

**Verify index**:
```bash
uv run python test_rag.py
```

### Running the Application

#### Option A: Use the Startup Script (Windows)

```bash
start.bat
```

This launches both backend and frontend in separate terminal windows.

#### Option B: Manual Start

**Terminal 1 - Backend**:
```bash
uv run uvicorn app.backend.main:app --reload
```

**Terminal 2 - Frontend**:
```bash
cd app/frontend/agent-frontend
npm start
```

### Access the Application

- **Frontend UI**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

## Configuration Reference

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| **LLM Configuration** |
| `LLM_PROVIDER` | LLM backend: `ollama`, `openai`, or `mistral` | `ollama` | Yes |
| `OLLAMA_MODEL` | Ollama model name | `hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M` | If using Ollama |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` | No |
| `OPENAI_API_KEY` | OpenAI API key | – | If using OpenAI |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o` | If using OpenAI |
| `MISTRAL_API_KEY` | Mistral API key | – | If using Mistral |
| `MISTRAL_MODEL` | Mistral model name | `mistral-medium-2508` | If using Mistral |
| **RAG Configuration** |
| `OLLAMA_EMBED_MODEL` | Ollama embedding model for RAG | `hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M` | Yes |
| `MANUALS_PATH` | Path to PDF documents directory | `data/manuals` | No |
| `INDEX_PATH` | Path to FAISS index storage | `data/faiss_index` | No |
| `CHUNK_SIZE` | Characters per document chunk | `800` | No |
| `CHUNK_OVERLAP` | Overlap between chunks | `150` | No |

### Manual Search Priority

The agent is configured to **prioritize local documents over web search**:

1. **Automatic Prioritization**: The system prompt instructs the LLM to use `manual_search` FIRST for any questions about:
   - Equipment, machines, instruments
   - Specifications and technical details
   - Procedures and maintenance
   - Troubleshooting
   - Manufacturing processes

2. **Fallback Behavior**: Only if `manual_search` returns no relevant results (or user explicitly requests web information) will the agent use `web_search`.

3. **Citation Tracking**: Results from `manual_search` include:
   - Source document filename
   - Page number
   - Relevance score (0-1, based on cosine similarity)
   - Relevant text excerpt

Example agent behavior:
```
User: "How do I calibrate the DSC-2500?"

Agent reasoning:
1. Use manual_search("DSC-2500 calibration") → Returns 5 chunks from TA-Instruments-Discovery-DSC-Brochure-EN.pdf
2. Synthesize answer citing pages 10, 15, 23
3. (No web search needed)
```

## Extending the Agent with Custom Tools

Tools are simple Python functions decorated with `@tool`. The decorator captures metadata (name, schema, description) so the agent can advertise and execute the tool safely.

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

The example `add_a_b` tool in `app/backend/api/agent.py` demonstrates how minimal a tool can be.

## Usage Examples

### Example 1: Technical Question with RAG

**Query**: "What is the temperature range of the Discovery DSC?"

**Agent Process**:
1. Calls `manual_search("Discovery DSC temperature range")`
2. Retrieves chunks from `TA-Instruments-Discovery-DSC-Brochure-EN.pdf`
3. Synthesizes answer: "The Discovery DSC operates from -90°C to 550°C (source: TA-Instruments-Discovery-DSC-Brochure-EN.pdf, page 6)"

### Example 2: Multi-Step Research

**Query**: "Compare the specifications of injection molding machines in our manuals"

**Agent Process**:
1. Calls `manual_search("injection molding specifications")` 
2. Finds results from `Manual_BOY-35-E-VV_unlocked.pdf` and `Manual_MAAC-Thermoformer.pdf`
3. Extracts key specs (clamp force, injection volume, etc.)
4. Compares and presents structured comparison table

### Example 3: Fallback to Web

**Query**: "What are the latest industry standards for plastic testing ISO updated in 2024?"

**Agent Process**:
1. Calls `manual_search("ISO plastic testing standards 2024")` → No relevant results (manuals outdated)
2. Falls back to `web_search("ISO plastic testing standards 2024")`
3. Combines local manual info with current web data

## Testing and Utilities

### Test Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `test_rag.py` | Test RAG search with sample queries | `uv run python test_rag.py` |
| `test_tool.py` | Test web search and URL fetch tools | `uv run python test_tool.py` |
| `test.py` | Verify reasoning tree serialization | `uv run python test.py` |

### RAG Index Management

**View index statistics**:
```bash
uv run python -m app.backend.core.rag.indexer
```

**Rebuild with custom settings**:
```bash
uv run python -m app.backend.core.rag.indexer \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --force
```

**Test specific query**:
```python
from app.backend.api.tools.manual_search import manual_search, ManualSearchArgs

result = manual_search(ManualSearchArgs(
    query="rheometer calibration procedure",
    top_k=5
))
print(result)
```

## Troubleshooting

### Index Not Found Error

**Problem**: `FileNotFoundError: Index files not found at data/faiss_index`

**Solution**: Build the index first:
```bash
uv run python -m app.backend.core.rag.indexer
```

### Ollama Connection Error

**Problem**: `Failed to generate embedding: connection refused`

**Solution**: 
1. Check Ollama is running: `ollama list`
2. Start Ollama service if needed
3. Verify embedding model is pulled: `ollama pull hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M`

### Empty Search Results

**Problem**: `manual_search` returns 0 results for known topics

**Solutions**:
1. Verify PDFs are in `data/manuals/`
2. Rebuild index: `uv run python -m app.backend.core.rag.indexer --force`
3. Try test script: `uv run python test_rag.py`
4. Check chunk size (may need adjustment for your documents)

### GPU Not Available (FAISS)

**Problem**: `GPU not available, using CPU`

**Status**: This is expected with `faiss-cpu`. The system will work fine on CPU.

**For GPU acceleration** (optional, requires CUDA):
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

## Advanced Topics

### Adding New Tools

Create a new tool in `app/backend/api/tools/`:

```python
from pydantic import BaseModel, Field
from app.backend.core.agent.tool import tool

class MyToolArgs(BaseModel):
    param: str = Field(..., description="Parameter description")

@tool("my_tool", MyToolArgs, "Tool description for LLM")
def my_tool(args: MyToolArgs) -> dict:
    result = process(args.param)
    return {"result": result}
```

Register in `app/backend/api/agent.py`:
```python
from app.backend.api.tools.my_tool import my_tool

for tool_fn in (manual_search, my_tool, web_search, fetch_url, add_a_b):
    llm.register_decorated_tool(tool_fn)
```

### Customizing Chunk Size

Larger chunks preserve more context but reduce retrieval precision:

```env
CHUNK_SIZE=1200        # Larger chunks (default: 800)
CHUNK_OVERLAP=200      # More overlap (default: 150)
```

Rebuild index after changing:
```bash
uv run python -m app.backend.core.rag.indexer --force
```

### Using Different Embedding Models

Any Ollama embedding model can be used:

```env
# Alternative embedding models
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_EMBED_MODEL=mxbai-embed-large
```

Pull the model:
```bash
ollama pull nomic-embed-text
```

Rebuild index:
```bash
uv run python -m app.backend.core.rag.indexer --force
```

## Performance Optimization

### For Large Document Collections (100+ PDFs)

1. **Increase batch size** for faster indexing:
   ```bash
   uv run python -m app.backend.core.rag.indexer --batch-size 64
   ```

2. **Use GPU FAISS** for faster search (requires CUDA)

3. **Consider FAISS IVF index** for 100K+ chunks (modify `vector_store.py`)

### For Better Retrieval Quality

1. **Adjust top_k** in manual searches (default: 5):
   ```python
   ManualSearchArgs(query="...", top_k=10)  # Return more results
   ```

2. **Fine-tune chunk size** based on your document structure

3. **Add metadata filtering** (requires code modification to filter by document type, date, etc.)

