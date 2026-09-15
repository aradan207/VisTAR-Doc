# Docker Deployment Guide

This guide documents the Docker workflow for VisTAR-Doc. It replaces the older Docker docs and provides a single source of truth.

## Overview

- Backend runs in a container (FastAPI on port 8000).
- Frontend runs in a container (Nginx on port 3000).
- Ollama runs on the host machine (not in Docker).
- Image artifacts are mounted from the sibling repo: ../vlm-yolo-detector/data/processed.
- API and media URLs are emitted as relative paths in Docker (via `RELATIVE` sentinel).
- In local dev (no Docker), image URLs default to `http://localhost:8000` so the browser resolves them correctly.

## Prerequisites

1. Docker Desktop installed and running.
2. Ollama running on the host.
3. Repository layout:

   Repositories/
   ├── VisTAR-Doc/
   └── vlm-yolo-detector/

4. Image artifacts exist in ../vlm-yolo-detector/data/processed:
   - image_index.json
   - image_embeddings.npy
   - embedding_mapping.json
   - images/

5. FAISS index exists in data/faiss_index.

## One-Time Setup

Copy the Docker environment template:

```bash
copy .env.docker.example .env.docker
```

This creates .env.docker with safe defaults. Do not commit .env.docker.

## Start and Stop

Recommended (checks prerequisites and prints status):

```bash
docker-start.bat
```

Stop:

```bash
docker-stop.bat
```

## Manual Commands

Start:

```bash
docker compose --env-file .env.docker up -d backend frontend
```

Stop:

```bash
docker compose --env-file .env.docker down
```

Logs:

```bash
docker compose --env-file .env.docker logs -f
```

Status:

```bash
docker compose --env-file .env.docker ps
```

## Access

- UI: http://localhost:3000
- API: http://localhost:8000

## Configuration (.env.docker)

Common variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| DOCKER_OLLAMA_HOST | http://host.docker.internal:11434 | Ollama host reachable from containers |
| DOCKER_OFFLINE_MODE | false | Disable web tools (strict offline) |
| DOCKER_REQUIRE_OLLAMA | true | Fail startup if Ollama is unreachable |
| DOCKER_REQUIRE_IMAGE_RETRIEVAL | true | Require image artifacts on startup |
| DOCKER_REQUIRE_SEMANTIC_MODEL_CACHE | false | Require sentence-transformer cache |
| DOCKER_API_PUBLIC_BASE_URL | RELATIVE | Image URL mode: RELATIVE for nginx proxy, or an absolute URL for LAN |
| DOCKER_API_BASE_URL | http://backend:8000 | Internal Docker URL for validation |

### Image URL Behavior

- **Docker (default)**: `DOCKER_API_PUBLIC_BASE_URL=RELATIVE` — backend emits relative URLs like `/api/media/yologen/...` which nginx proxies to the backend container.
- **Local dev (no Docker)**: When `API_PUBLIC_BASE_URL` is unset, the backend defaults to `http://localhost:8000` so images resolve correctly even when the frontend is served on a different port (e.g. 3000).
- **LAN access**: Set `DOCKER_API_PUBLIC_BASE_URL` to the host IP (e.g. `http://192.168.x.x:8000`).

### LAN Access

1. Find host IP (ipconfig).
2. Set in .env.docker:

```env
DOCKER_API_PUBLIC_BASE_URL=http://192.168.x.x:8000
```

3. Restart:

```bash
docker compose --env-file .env.docker restart
```

## Strict Offline Mode

To block network requests in containers:

```env
DOCKER_OFFLINE_MODE=true
DOCKER_REQUIRE_SEMANTIC_MODEL_CACHE=true
```

Prepare caches first on a connected machine:

```bash
uv run python tests/prepare_offline_semantic_cache.py
```

## Regenerate Image Artifacts (Optional)

If manuals or image extraction logic change:

```bash
docker compose --env-file .env.docker --profile preprocess run --rm vlm-preprocess
```

This rebuilds artifacts under ../vlm-yolo-detector/data/processed.

## Troubleshooting

### Backend is unhealthy

```bash
docker compose --env-file .env.docker logs backend
```

Common causes:
- Ollama unreachable: start Ollama or fix DOCKER_OLLAMA_HOST.
- Missing artifacts: confirm ../vlm-yolo-detector/data/processed contents.
- Missing semantic cache: set DOCKER_REQUIRE_SEMANTIC_MODEL_CACHE=false.

### Frontend cannot connect

- Check backend is healthy: docker compose --env-file .env.docker ps
- Check logs: docker compose --env-file .env.docker logs frontend
- Verify ports 3000 and 8000 are free.

### Docker Desktop UI

You can start from Docker Desktop Compose, but it does not check prerequisites. Use docker-start.bat when possible.

## Architecture

Host (Windows 11)
├── Ollama (11434)
└── Docker Desktop
    ├── backend (FastAPI, :8000)
    │   ├── mounts: ./data/faiss_index, ./data/manuals, ./.cache
    │   └── mounts: ../vlm-yolo-detector/data/processed (read-only)
    └── frontend (Nginx, :3000)
        └── proxies /api to backend

## Quick Command Reference

- Start: docker-start.bat
- Stop: docker-stop.bat
- Status: docker compose --env-file .env.docker ps
- Logs: docker compose --env-file .env.docker logs -f
- Rebuild: docker compose --env-file .env.docker build
