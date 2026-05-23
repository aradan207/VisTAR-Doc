@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Agentic RAG - Automated Installation Script
echo ============================================================
echo.

:: Change to the script's directory
cd /d "%~dp0"

:: Use repo-local caches for deterministic offline portability.
set "CACHE_ROOT=%CD%\.cache"
set "HF_HOME=%CACHE_ROOT%\huggingface"
set "SENTENCE_TRANSFORMERS_HOME=%CACHE_ROOT%\torch\sentence_transformers"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%SENTENCE_TRANSFORMERS_HOME%" mkdir "%SENTENCE_TRANSFORMERS_HOME%"
echo Cache root: %CACHE_ROOT%
echo HF_HOME: %HF_HOME%
echo SENTENCE_TRANSFORMERS_HOME: %SENTENCE_TRANSFORMERS_HOME%
echo.

echo Select installation mode:
echo   0 = Online prepare/install mode
echo   1 = Offline install/validation mode
set /p INSTALL_MODE=Enter mode [0]: 
if "%INSTALL_MODE%"=="" set INSTALL_MODE=0

if not "%INSTALL_MODE%"=="0" if not "%INSTALL_MODE%"=="1" (
    echo ERROR: Invalid mode selection. Enter 0 or 1.
    pause
    exit /b 1
)

if "%INSTALL_MODE%"=="1" (
    set "ONLINE_MODE=0"
    echo Running in OFFLINE mode.
) else (
    set "ONLINE_MODE=1"
    echo Running in ONLINE mode.
)
echo.

:: Check for Python
echo [1/6] Checking Python installation...
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

:: Install/validate uv package manager
echo [2/6] Preparing uv package manager...
if "%ONLINE_MODE%"=="1" (
    pip install uv --quiet
    if errorlevel 1 (
        echo ERROR: Failed to install uv package manager.
        pause
        exit /b 1
    )
    echo uv installed successfully.
) else (
    uv --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv is required in offline mode but was not found.
        echo Install uv on a connected machine and retry.
        pause
        exit /b 1
    )
    echo Found uv in offline mode.
)
echo.

:: Install/validate Python dependencies
echo [3/6] Installing or validating Python dependencies...
if "%ONLINE_MODE%"=="1" (
    echo Using system Python installation...
    uv sync --python-preference only-system
    if errorlevel 1 (
        echo First attempt failed. Trying with native TLS...
        uv sync --python-preference only-system --native-tls
        if errorlevel 1 (
            echo.
            echo ERROR: Failed to install Python dependencies.
            echo.
            echo Troubleshooting tips:
            echo   1. Check your internet connection
            echo   2. Try running: uv sync --python-preference only-system --native-tls
            echo   3. If behind a corporate proxy, configure proxy settings
            echo.
            pause
            exit /b 1
        )
    )
    echo Python dependencies installed successfully.

    echo Warming sentence-transformers caches for strict offline mode...
    if exist ".\.venv\Scripts\python.exe" (
        .\.venv\Scripts\python.exe tests\prepare_offline_semantic_cache.py
        if errorlevel 1 (
            echo WARNING: Failed to fully warm semantic model caches.
            echo You can retry with: .\.venv\Scripts\python.exe tests\prepare_offline_semantic_cache.py
        )
    ) else (
        echo WARNING: .venv missing; skipped semantic cache warm-up.
    )
) else (
    if exist ".venv\Scripts\python.exe" (
        echo Found existing virtual environment at .venv\Scripts\python.exe
    ) else (
        echo WARNING: .venv not found. Offline mode cannot fetch Python dependencies.
        echo Prepare dependencies in online mode first.
    )
)
echo.

:: Check for Node.js and install frontend dependencies
echo [4/6] Setting up frontend dependencies...
if "%ONLINE_MODE%"=="1" (
    where npm >nul 2>&1
    if errorlevel 1 (
        echo Node.js/npm not found. Attempting to install Node.js...
        echo.
        call :install_nodejs
        if errorlevel 1 (
            pause
            exit /b 1
        )
    )

    where npm >nul 2>&1
    if errorlevel 1 (
        echo ERROR: npm still not found after installation.
        echo Please close this window, open a new terminal, and run install.bat again.
        echo Or install Node.js manually from https://nodejs.org/
        pause
        exit /b 1
    )

    echo Installing npm packages for frontend...
    cd app\frontend\agent-frontend
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install frontend dependencies.
        cd ..\..\..
        pause
        exit /b 1
    )
    if not exist "node_modules\marked\lib\marked.umd.js" (
        echo ERROR: marked runtime asset is missing after npm install.
        echo Expected: app\frontend\agent-frontend\node_modules\marked\lib\marked.umd.js
        cd ..\..\..
        pause
        exit /b 1
    )
    cd ..\..\..
    echo Frontend dependencies installed successfully.
) else (
    if exist "app\frontend\agent-frontend\node_modules\marked\lib\marked.umd.js" (
        echo Found local marked runtime asset for frontend markdown rendering.
    ) else (
        echo WARNING: Frontend marked runtime asset is missing.
        echo Missing: app\frontend\agent-frontend\node_modules\marked\lib\marked.umd.js
        echo Prepare frontend dependencies in online mode first.
    )
)
echo.

:: Check vlm-yolo-detector image artifacts
echo [5/6] Checking vlm-yolo-detector image artifacts...
set "PARENT_DIR=%~dp0.."
if "%ONLINE_MODE%"=="1" (
    if exist "%PARENT_DIR%\vlm-yolo-detector" (
        echo Found existing vlm-yolo-detector repository.
    ) else (
        echo Cloning vlm-yolo-detector repository...
        cd "%PARENT_DIR%"
        git clone https://github.com/morkev/vlm-yolo-detector.git
        if errorlevel 1 (
            echo WARNING: Failed to clone vlm-yolo-detector.
            echo Image search will not work. Clone manually to %PARENT_DIR%\vlm-yolo-detector
        ) else (
            echo Repository cloned. Run install.bat in vlm-yolo-detector to set up image data.
        )
        cd "%~dp0"
    )
) else (
    if not exist "%PARENT_DIR%\vlm-yolo-detector\data\processed" (
        echo WARNING: vlm-yolo-detector processed folder not found.
        echo Expected: %PARENT_DIR%\vlm-yolo-detector\data\processed
    )
)

if exist "%PARENT_DIR%\vlm-yolo-detector\data\processed\image_index.json" (
    echo Found image_index.json
) else (
    echo WARNING: Missing image_index.json
)
if exist "%PARENT_DIR%\vlm-yolo-detector\data\processed\image_embeddings.npy" (
    echo Found image_embeddings.npy
) else (
    echo WARNING: Missing image_embeddings.npy
)
if exist "%PARENT_DIR%\vlm-yolo-detector\data\processed\embedding_mapping.json" (
    echo Found embedding_mapping.json
) else (
    echo WARNING: Missing embedding_mapping.json
)
echo.

:: Build FAISS index
echo [6/6] Building FAISS index from PDF manuals...
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe -m app.backend.core.rag.indexer
    if errorlevel 1 (
        echo WARNING: Failed to build FAISS index.
        echo You can build it manually later with: uv run python -m app.backend.core.rag.indexer
    )
) else (
    echo WARNING: Python virtual environment missing; skipping FAISS build.
)
echo.

:: Verify installation
echo ============================================================
echo  Verifying Installation
echo ============================================================
echo.

echo Testing RAG functionality...
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe tests\test_rag.py
) else (
    echo Skipped: .venv\Scripts\python.exe not found
)
echo.

echo Running offline readiness check...
if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe tests\offline_readiness_check.py
    if errorlevel 1 (
        echo WARNING: Offline readiness check reported issues.
        echo Review the output above. Non-critical warnings can be ignored.
    )
) else (
    echo WARNING: Skipped readiness check because .venv is missing.
)
echo.

echo ============================================================
echo  Installation Complete!
echo ============================================================
echo.
echo To start the application, run: start.bat
echo.
echo IMPORTANT: Make sure Ollama models are pulled before starting.
echo   See README.md for the required model pull commands.
echo.
echo Or start manually:
echo   Backend:  uv run uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
echo   Frontend: cd app\frontend\agent-frontend ^&^& npm start
echo.
pause
goto :eof

:: ============================================================
:: Subroutine: Install Node.js
:: ============================================================
:install_nodejs
:: Try winget first
winget install OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements >nul 2>&1
if not errorlevel 1 goto :refresh_path

echo Winget installation failed. Downloading Node.js installer...

:: Download Node.js installer using PowerShell
set "NODE_MSI=%TEMP%\nodejs.msi"
powershell -Command "Invoke-WebRequest -Uri 'https://nodejs.org/dist/v22.16.0/node-v22.16.0-x64.msi' -OutFile '%NODE_MSI%' -UseBasicParsing"
if errorlevel 1 (
    echo ERROR: Failed to download Node.js installer.
    echo Please install Node.js manually from https://nodejs.org/
    exit /b 1
)

echo Installing Node.js (this may require administrator privileges)...
start /wait msiexec /i "%NODE_MSI%" /qn /norestart
if errorlevel 1 (
    echo Silent install failed. Launching interactive installer...
    start /wait "" "%NODE_MSI%"
)

:: Clean up
del "%NODE_MSI%" >nul 2>&1

:refresh_path
:: Refresh PATH environment
echo Refreshing PATH environment...
for /f "usebackq tokens=*" %%a in (powershell -Command "[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')") do set "PATH=%%a"
exit /b 0
