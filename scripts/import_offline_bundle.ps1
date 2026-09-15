param(
    [string]$BundleZip = "../vistar-doc-offline-bundle.zip"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $repoRoot
$zipPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BundleZip))

if (-not (Test-Path $zipPath)) {
    throw "Bundle not found: $zipPath"
}

Push-Location $workspaceRoot
try {
    tar -xf $zipPath
} finally {
    Pop-Location
}

Write-Host "Offline bundle import completed into workspace: $workspaceRoot"
Write-Host "Next: run 'uv run python tests/offline_readiness_check.py' from the repository root."
