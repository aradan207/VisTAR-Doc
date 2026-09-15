param(
    [string]$OutputZip = "../vistar-doc-offline-bundle.zip"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $repoRoot
 
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputZip))
if (Test-Path $outputPath) {
    Remove-Item -Path $outputPath -Force
}

# Derive the checkout folder name instead of hardcoding it, so the bundle is
# still complete when the repository is cloned under a different directory name.
$repoName = Split-Path -Leaf $repoRoot

$includes = @(
    "$repoName/.cache",
    "$repoName/data/faiss_index",
    "vlm-yolo-detector/data/processed"
)

foreach ($item in $includes) {
    $abs = Join-Path $workspaceRoot $item
    if (Test-Path $abs) {
        Write-Host "Including: $item"
    } else {
        Write-Host "Missing (skipped): $item"
    }
}

Push-Location $workspaceRoot
try {
    tar -a -c -f $outputPath @includes
} finally {
    Pop-Location
}

Write-Host "Offline bundle created: $outputPath"
