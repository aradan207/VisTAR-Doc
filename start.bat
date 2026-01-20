@echo off
echo Starting Agentic RAG Agent...
echo.

REM Use system Python to avoid SSL certificate issues when downloading Python
REM Running without --reload to ensure embeddings persist correctly
start "Agentic Backend" cmd /k "cd /d %~dp0 && uv run --python python uvicorn app.backend.main:app --host 0.0.0.0 --port 8000"
timeout /t 3 >nul
start "Agentic Frontend" cmd /k "cd /d %~dp0app\frontend\agent-frontend && npm start"
echo.
echo Agentic RAG Agent started.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit this window...
pause >nul
