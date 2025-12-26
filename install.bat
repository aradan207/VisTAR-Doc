@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Agentic RAG - Automated Installation Script
echo ============================================================
echo.

:: Change to the script's directory
cd /d "%~dp0"

:: Check for Python
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%
echo.

:: Check for Ollama
echo [2/7] Checking Ollama installation...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Ollama is not installed or not in PATH.
    echo Please install Ollama from https://ollama.com/download
    echo After installation, restart this script.
    pause
    exit /b 1
)
for /f "tokens=4" %%i in ('ollama --version 2^>^&1') do set OLLAMA_VERSION=%%i
echo Found Ollama %OLLAMA_VERSION%
echo.

:: Install uv package manager
echo [3/7] Installing uv package manager...
pip install uv --quiet
if errorlevel 1 (
    echo ERROR: Failed to install uv package manager.
    pause
    exit /b 1
)
echo uv installed successfully.
echo.

:: Install Python dependencies
echo [4/7] Installing Python dependencies...
uv sync
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies.
    pause
    exit /b 1
)
echo Python dependencies installed successfully.
echo.

:: Check for Node.js and install frontend dependencies
echo [5/7] Setting up frontend dependencies...
where npm >nul 2>&1
if errorlevel 1 (
    echo Node.js/npm not found. Attempting to install Node.js...
    echo.
    
    :: Try winget first
    winget install OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements >nul 2>&1
    if errorlevel 1 (
        echo Winget installation failed. Downloading Node.js installer...
        
        :: Download Node.js installer
        powershell -Command "Invoke-WebRequest -Uri 'https://nodejs.org/dist/v22.16.0/node-v22.16.0-x64.msi' -OutFile '%TEMP%\nodejs.msi' -UseBasicParsing"
        if errorlevel 1 (
            echo ERROR: Failed to download Node.js installer.
            echo Please install Node.js manually from https://nodejs.org/
            pause
            exit /b 1
        )
        
        echo Installing Node.js (this may require administrator privileges)...
        start /wait msiexec /i "%TEMP%\nodejs.msi" /qn /norestart
        if errorlevel 1 (
            echo Silent install failed. Launching interactive installer...
            start /wait "" "%TEMP%\nodejs.msi"
        )
        
        :: Clean up
        del "%TEMP%\nodejs.msi" >nul 2>&1
    )
    
    :: Refresh PATH
    echo Refreshing PATH environment...
    for /f "tokens=*" %%a in ('powershell -Command "[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')"') do set "PATH=%%a"
)

:: Verify npm is available
where npm >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm still not found after installation.
    echo Please close this window, open a new terminal, and run install.bat again.
    echo Or install Node.js manually from https://nodejs.org/
    pause
    exit /b 1
)

:: Install frontend dependencies
echo Installing npm packages for frontend...
cd app\frontend\agent-frontend
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install frontend dependencies.
    cd ..\..\..
    pause
    exit /b 1
)
cd ..\..\..
echo Frontend dependencies installed successfully.
echo.

:: Pull Ollama models
echo [6/7] Pulling Ollama models (this may take several minutes)...
echo.

echo Pulling Mistral 7B Instruct (4.4 GB)...
ollama pull hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M
if errorlevel 1 (
    echo WARNING: Failed to pull Mistral 7B model. You can try manually later.
)

echo.
echo Pulling Ministral 8B Instruct (6.1 GB) - Best quality model...
ollama pull hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M
if errorlevel 1 (
    echo WARNING: Failed to pull Ministral 8B model. You can try manually later.
)

echo.
echo Pulling mxbai-embed-large embedding model (215 MB)...
ollama pull hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q4_K_M
if errorlevel 1 (
    echo ERROR: Failed to pull embedding model. RAG will not work without this.
    pause
    exit /b 1
)

echo.
echo All Ollama models pulled successfully.
echo.

:: Build FAISS index
echo [7/7] Building FAISS index from PDF manuals...
.\.venv\Scripts\python.exe -m app.backend.core.rag.indexer
if errorlevel 1 (
    echo WARNING: Failed to build FAISS index.
    echo You can build it manually later with: uv run python -m app.backend.core.rag.indexer
)
echo.

:: Verify installation
echo ============================================================
echo  Verifying Installation
echo ============================================================
echo.

echo Checking Ollama models...
ollama list
echo.

echo Testing RAG functionality...
.\.venv\Scripts\python.exe test_rag.py
echo.

echo ============================================================
echo  Installation Complete!
echo ============================================================
echo.
echo To start the application, run: start.bat
echo.
echo Or start manually:
echo   Backend:  uv run uvicorn app.backend.main:app --reload
echo   Frontend: cd app\frontend\agent-frontend ^&^& npm start
echo.
echo Available LLM models:
echo   - hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M (default, best quality)
echo   - hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M (lighter alternative)
echo.
echo To switch models, edit the OLLAMA_MODEL value in .env
echo.
pause
