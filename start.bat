@echo off
echo Starting Agentic RAG Agent...
echo.
start "Agentic Backend" cmd /k "cd /d %~dp0 && uv run uvicorn app.backend.main:app --reload"
timeout /t 3 >nul
start "Agentic Frontend" cmd /k "cd /d %~dp0app\frontend\agent-frontend && npm start"
echo.
echo Agentic RAG Agent started with RAG capability.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit this window...
pause >nul
