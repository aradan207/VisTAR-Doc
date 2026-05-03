@echo off
REM Stop the Docker Compose stack for agentic-rag

echo.
echo ============================================================
echo Stopping agentic-rag Docker Compose Stack
echo ============================================================
echo.

docker compose --env-file .env.docker down

echo.
echo Stopped. Run docker-start.bat to start again.
echo.
