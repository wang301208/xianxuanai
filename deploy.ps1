#Requires -Version 5.0
param(
    [string] $Service = "forge",
    [int] $Port = 8000,
    [switch] $NoStart,
    [switch] $CheckOnly,
    [switch] $SkipSystemDeps,
    [switch] $SkipDeps,
    [switch] $SkipAssets,
    [switch] $FetchAssets,
    [string] $AssetsManifest = "",
    [switch] $Yes,
    [string] $MinPythonVersion = "3.10",
    [string] $PythonWingetId = "Python.Python.3.11"
)

$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
if (-not $repoRoot) {
    $repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

Write-Host "==> Running setup..."

$setupScript = Join-Path $repoRoot "scripts\\setup.ps1"
if (-not (Test-Path $setupScript -PathType Leaf)) {
    Write-Error "Missing setup script: $setupScript"
    exit 1
}

$setupParams = @{
    MinPythonVersion = $MinPythonVersion
    PythonWingetId   = $PythonWingetId
}
if ($CheckOnly) { $setupParams.CheckOnly = $true }
if ($SkipSystemDeps) { $setupParams.SkipSystemDeps = $true }
if ($SkipDeps) { $setupParams.SkipDeps = $true }
if ($SkipAssets) { $setupParams.SkipAssets = $true }
if ($FetchAssets) { $setupParams.FetchAssets = $true }
if ($AssetsManifest) { $setupParams.AssetsManifest = $AssetsManifest }
if ($Yes) { $setupParams.Yes = $true }

& $setupScript @setupParams

if ($NoStart) {
    Write-Host "==> Setup complete (-NoStart)."
    exit 0
}

if ($CheckOnly) {
    Write-Host "==> Setup ran in -CheckOnly mode; skipping start."
    exit 0
}

$envTemplate = Join-Path $repoRoot "config\\.env.template"
$envFile = Join-Path $repoRoot ".env"
if (-not (Test-Path $envFile -PathType Leaf)) {
    if (Test-Path $envTemplate -PathType Leaf) {
        Copy-Item -Path $envTemplate -Destination $envFile -Force
        Write-Host "==> Created .env from config/.env.template. Please edit .env and add your API keys."
    } else {
        Write-Host "==> Warning: config/.env.template not found; create .env manually if needed."
    }
}

switch ($Service) {
    "forge" {
        $env:PORT = "$Port"
        Write-Host "==> Starting Forge server on http://localhost:$Port (Ctrl+C to stop)"
        Push-Location $repoRoot
        try {
            & poetry run python -m forge
        } finally {
            Pop-Location
        }
    }
    default {
        Write-Error "Unknown service '$Service' (supported: forge)"
        exit 1
    }
}
