@echo off
REM Start the Docker Compose stack for agentic-rag
REM Requires: Docker Desktop running, .env.docker file, Ollama running on host

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo Starting agentic-rag Docker Compose Stack
echo ============================================================
echo.

REM Check if Docker is running
docker ps >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if .env.docker exists
if not exist .env.docker (
    echo WARNING: .env.docker not found. Creating from .env.docker.example...
    copy .env.docker.example .env.docker
    if errorlevel 1 (
        echo ERROR: Failed to copy .env.docker.example
        pause
        exit /b 1
    )
)

REM Check if Ollama is reachable
echo Checking Ollama connectivity...
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5 -UseBasicParsing; if ($r.StatusCode -eq 200) { Write-Host 'OK: Ollama is reachable'; exit 0 } } catch { Write-Host 'WARNING: Ollama at http://localhost:11434 is not reachable. Make sure Ollama is running on host.'; exit 1 }"
if errorlevel 1 (
    echo.
    echo To start Ollama on your host machine:
    echo   ollama serve
    echo.
    echo You can run this in a separate terminal/PowerShell window.
)

echo.
echo Building Docker images...
docker compose --env-file .env.docker build
if errorlevel 1 (
    echo ERROR: Docker build failed
    pause
    exit /b 1
)

echo.
echo Starting backend and frontend containers...
docker compose --env-file .env.docker up -d backend frontend
if errorlevel 1 (
    echo ERROR: Docker compose up failed
    pause
    exit /b 1
)

echo.
echo Waiting for services to be healthy...
timeout /t 3 /nobreak

echo.
docker compose --env-file .env.docker ps

echo.
echo ============================================================
echo Stack Status
echo ============================================================
echo.
docker compose --env-file .env.docker ps

echo.
echo API:  http://localhost:8000
echo UI:   http://localhost:3000
echo.
echo For logs:
echo   docker compose --env-file .env.docker logs -f backend
echo   docker compose --env-file .env.docker logs -f frontend
echo.
echo To stop:
echo   docker compose --env-file .env.docker down
echo.
