#Requires -Version 5.0
param(
    [string] $MinPythonVersion = "3.10",
    [string] $PythonWingetId = "Python.Python.3.11",
    [switch] $CheckOnly,
    [switch] $SkipDeps,
    [switch] $SkipSystemDeps,
    [switch] $SkipAssets,
    [switch] $FetchAssets,
    [string] $AssetsManifest = "",
    [switch] $Yes
)

$ErrorActionPreference = 'Stop'

if ($env:OS -ne "Windows_NT") {
    Write-Error "setup.ps1 should be run on Windows."
    exit 1
}

Write-Host "OS: Windows ($([Environment]::OSVersion.VersionString)), PowerShell: $($PSVersionTable.PSVersion)"

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $gpuLine = $null
    $cudaVersion = $null

    try {
        $gpuLine = (& nvidia-smi -L 2>$null | Select-Object -First 1)
    } catch {
        $gpuLine = $null
    }

    try {
        $out = & nvidia-smi 2>$null
        if ($out -match "CUDA Version:\s*(\d+\.\d+)") {
            $cudaVersion = $Matches[1]
        }
    } catch {
        $cudaVersion = $null
    }

    $msg = "NVIDIA GPU: detected"
    if ($gpuLine) { $msg += " ($gpuLine)" }
    if ($cudaVersion) { $msg += ", CUDA: $cudaVersion" }
    Write-Host $msg
} else {
    Write-Host "NVIDIA GPU: not detected (nvidia-smi not found)"
}

function Write-StatusOk {
    param([string] $Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-StatusWarn {
    param([string] $Message)
    Write-Host "⚠️ $Message" -ForegroundColor Yellow
}

function Write-StatusInfo {
    param([string] $Message)
    Write-Host "ℹ️ $Message" -ForegroundColor Cyan
}

function Get-Executable {
    param([string] $Name)
    return Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue
}

function Ensure-SystemTool {
    param(
        [string] $Executable,
        [string] $Label,
        [string] $WingetId = "",
        [string] $HelpUrl = "",
        [switch] $Optional
    )

    $cmd = Get-Executable -Name $Executable
    if ($cmd) {
        Write-StatusOk "$Label 已安装 ($($cmd.Source))"
        return $true
    }

    Write-StatusWarn "$Label 未找到"
    if ($CheckOnly -or $SkipSystemDeps) {
        return $false
    }

    $winget = Get-Executable -Name "winget"
    if ($winget -and $WingetId) {
        Write-StatusInfo "正在安装 ${Label} (winget: ${WingetId})..."
        try {
            & winget install --id $WingetId -e --source winget --silent --accept-package-agreements --accept-source-agreements
        } catch {
            Write-StatusWarn "winget 安装失败：$($_.Exception.Message)"
        }

        $cmd = Get-Executable -Name $Executable
        if ($cmd) {
            Write-StatusOk "$Label 已安装 ($($cmd.Source))"
            return $true
        }
    }

    if ($HelpUrl) {
        Write-Host "  Install: $HelpUrl"
    }
    return $false
}

if (-not $SkipSystemDeps) {
    Write-Host "Checking system tools..."
    Ensure-SystemTool -Executable "git" -Label "git" -WingetId "Git.Git" -HelpUrl "https://git-scm.com/download/win" | Out-Null
    Ensure-SystemTool -Executable "ffmpeg" -Label "ffmpeg" -WingetId "Gyan.FFmpeg" -HelpUrl "https://ffmpeg.org/download.html" -Optional | Out-Null
    Ensure-SystemTool -Executable "curl.exe" -Label "curl" -Optional | Out-Null

    $docker = Get-Executable -Name "docker"
    if ($docker) {
        Write-StatusOk "docker 已安装"
    } else {
        Write-StatusWarn "docker 未找到（如需本地 compose 部署，请安装 Docker Desktop）"
    }
} else {
    Write-Host "Skipping system dependency check (-SkipSystemDeps)."
}

function Resolve-PythonInvoker {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @("python")
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return @("py", "-3")
    }

    return $null
}

function Get-PythonVersion {
    param([string[]] $PythonInvoker)
    $raw = & $PythonInvoker[0] @($PythonInvoker | Select-Object -Skip 1) --version 2>&1
    if ($raw -match "Python\s+(\d+)\.(\d+)\.(\d+)") {
        return [version]::new($Matches[1], $Matches[2], $Matches[3])
    }
    if ($raw -match "Python\s+(\d+)\.(\d+)") {
        return [version]::new($Matches[1], $Matches[2], 0)
    }
    throw "Unable to parse Python version from output: $raw"
}

function Install-Python {
    Write-Host "Python not found (or below required version). Attempting installation via winget..."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Error "winget is not available. Install Python $PythonWingetId manually from https://www.python.org/downloads/"
        exit 1
    }

    try {
        winget install --id $PythonWingetId -e --source winget --silent --accept-package-agreements --accept-source-agreements
    } catch {
        Write-Error "Failed to install Python via winget. Please install it manually from https://www.python.org/downloads/"
        exit 1
    }
}

function Install-Poetry {
    Write-Host "Poetry not found. Attempting installation..."
    $pythonInvoker = Resolve-PythonInvoker
    if (-not $pythonInvoker) {
        Write-Error "Python is required to install Poetry, but no Python executable was found."
        exit 1
    }

    try {
        (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | & $pythonInvoker[0] @($pythonInvoker | Select-Object -Skip 1) -
    } catch {
        Write-Error "Failed to install Poetry. See https://python-poetry.org/docs/#installation for help."
        exit 1
    }

    $env:Path = "${env:APPDATA}\\Python\\Scripts;${env:USERPROFILE}\\.local\\bin;${env:Path}"
}

function Detect-DependencyManager {
    param([string] $RepoRoot)

    $environmentYml = Join-Path $RepoRoot "environment.yml"
    $environmentYaml = Join-Path $RepoRoot "environment.yaml"
    $requirementsTxt = Join-Path $RepoRoot "requirements.txt"
    $pyprojectToml = Join-Path $RepoRoot "pyproject.toml"

    if (Test-Path $environmentYml -PathType Leaf) {
        return [pscustomobject]@{ Manager = "conda"; Manifest = $environmentYml }
    }
    if (Test-Path $environmentYaml -PathType Leaf) {
        return [pscustomobject]@{ Manager = "conda"; Manifest = $environmentYaml }
    }
    if (Test-Path $requirementsTxt -PathType Leaf) {
        return [pscustomobject]@{ Manager = "pip"; Manifest = $requirementsTxt }
    }
    if (Test-Path $pyprojectToml -PathType Leaf) {
        return [pscustomobject]@{ Manager = "poetry"; Manifest = $pyprojectToml }
    }

    return [pscustomobject]@{ Manager = "none"; Manifest = $null }
}

function Install-ProjectDeps {
    param(
        [string] $RepoRoot,
        [string[]] $PythonInvoker
    )

    $detected = Detect-DependencyManager -RepoRoot $RepoRoot
    $manager = $detected.Manager
    $manifest = $detected.Manifest

    switch ($manager) {
        "conda" {
            $conda = Get-Command conda -ErrorAction SilentlyContinue
            if (-not $conda) {
                Write-Error "Detected conda environment file ($manifest) but 'conda' is not on PATH."
                exit 1
            }

            Write-Host "Installing dependencies via conda (manifest: $manifest)..."
            & conda env update -f $manifest --prune
            if ($LASTEXITCODE -ne 0) {
                Write-Host "conda env update failed; attempting create..."
                & conda env create -f $manifest
                if ($LASTEXITCODE -ne 0) {
                    Write-Error "Conda dependency installation failed."
                    exit 1
                }
            }
        }
        "pip" {
            if (-not $PythonInvoker) {
                Write-Error "Detected requirements.txt but no Python executable was found."
                exit 1
            }

            $pythonExe = $PythonInvoker[0]
            $pythonArgs = @()
            if ($PythonInvoker.Count -gt 1) {
                $pythonArgs = $PythonInvoker[1..($PythonInvoker.Count - 1)]
            }

            Write-Host "Installing dependencies via pip (manifest: $manifest)..."
            & $pythonExe @pythonArgs -m pip install --upgrade pip | Out-Null
            & $pythonExe @pythonArgs -m pip install -r $manifest
            if ($LASTEXITCODE -ne 0) {
                Write-Error "pip dependency installation failed."
                exit 1
            }
        }
        "poetry" {
            $poetryCmd = Get-Command poetry -ErrorAction SilentlyContinue
            if (-not $poetryCmd) {
                Write-Error "Detected pyproject.toml but 'poetry' is not on PATH."
                exit 1
            }

            Write-Host "Installing dependencies via Poetry (manifest: $manifest)..."
            Push-Location $RepoRoot
            try {
                & poetry install --no-interaction
                if ($LASTEXITCODE -ne 0) {
                    Write-Error "Poetry dependency installation failed."
                    exit 1
                }
            } finally {
                Pop-Location
            }
        }
        "none" {
            Write-Host "No dependency manifest found in repo root (pyproject.toml / requirements.txt / environment.yml)."
        }
        default {
            Write-Error "Unknown dependency manager '$manager'."
            exit 1
        }
    }
}

$min = [version]$MinPythonVersion
$pythonInvoker = Resolve-PythonInvoker
if ($pythonInvoker) {
    $current = Get-PythonVersion -PythonInvoker $pythonInvoker
    Write-Host "Python found: $current"
} else {
    Write-Host "Python not found."
    $current = $null
}

if (-not $pythonInvoker -or $current -lt $min) {
    if ($CheckOnly) {
        Write-Error "Python $MinPythonVersion+ is required."
        exit 1
    }

    Install-Python
    $pythonInvoker = Resolve-PythonInvoker
    if (-not $pythonInvoker) {
        Write-Error "Python installation finished but python is still not on PATH. Restart your terminal and re-run this script."
        exit 1
    }

    $current = Get-PythonVersion -PythonInvoker $pythonInvoker
    Write-Host "Python ready: $current"
    if ($current -lt $min) {
        Write-Error "Python $MinPythonVersion+ is required, but found $current."
        exit 1
    }
}

$poetry = Get-Command poetry -ErrorAction SilentlyContinue
if ($poetry) {
    $poetryVersion = & poetry --version 2>&1
    Write-Host "Poetry found: $poetryVersion"
} else {
    if ($CheckOnly) {
        Write-Error "Poetry is required but was not found."
        exit 1
    }
    Install-Poetry
    $poetry = Get-Command poetry -ErrorAction SilentlyContinue
    if ($poetry) {
        $poetryVersion = & poetry --version 2>&1
        Write-Host "Poetry installed: $poetryVersion"
    } else {
        Write-Warning "Poetry installer completed but 'poetry' is still not on PATH in this session."
    }
}

if (-not $CheckOnly -and -not $SkipDeps) {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    Install-ProjectDeps -RepoRoot $repoRoot -PythonInvoker $pythonInvoker
} elseif ($SkipDeps) {
    Write-Host "Skipping dependency installation (-SkipDeps)."
} else {
    Write-Host "check-only mode: skipping dependency installation."
}

if ($SkipAssets) {
    Write-Host "Skipping asset check (-SkipAssets)."
} else {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    if (-not $AssetsManifest) {
        $AssetsManifest = Join-Path $repoRoot "config\\assets.json"
    } elseif (-not [System.IO.Path]::IsPathRooted($AssetsManifest)) {
        $AssetsManifest = Join-Path $repoRoot $AssetsManifest
    }

    $assetsScript = Join-Path $PSScriptRoot "fetch_assets.py"
    if (-not (Test-Path $assetsScript -PathType Leaf)) {
        Write-Host "Assets helper not found at $assetsScript (skipping)."
    } elseif (-not (Test-Path $AssetsManifest -PathType Leaf)) {
        Write-Host "Assets manifest not found at $AssetsManifest (skipping)."
    } else {
        if (-not $pythonInvoker) {
            Write-Error "Python is required to check/fetch assets, but no Python executable was found."
            exit 1
        }

        $pythonExe = $pythonInvoker[0]
        $pythonArgs = @()
        if ($pythonInvoker.Count -gt 1) {
            $pythonArgs = $pythonInvoker[1..($pythonInvoker.Count - 1)]
        }

        if ($FetchAssets -and -not $CheckOnly) {
            Write-Host "Fetching model/data assets (manifest: $AssetsManifest)..."
            $cmdArgs = @($assetsScript, "fetch", "--manifest", $AssetsManifest)
            if ($Yes) {
                $cmdArgs += "--yes"
            }
            & $pythonExe @pythonArgs @cmdArgs
            if ($LASTEXITCODE -ne 0) {
                exit $LASTEXITCODE
            }
        } else {
            Write-Host "Checking model/data assets (manifest: $AssetsManifest)..."
            & $pythonExe @pythonArgs $assetsScript "check" "--manifest" $AssetsManifest | Out-Host
        }
    }
}

Write-Host "Environment setup complete."
