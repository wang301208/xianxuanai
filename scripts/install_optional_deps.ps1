param(
    [string[]] $Bundle = @("core"),
    [string] $Python = $env:PYTHON,
    [ValidateSet("auto", "cpu", "cu118", "cu121", "cu124")]
    [string] $Torch = "auto"
)

function Show-Usage {
    Write-Host @"
Usage: install_optional_deps.ps1 [-Bundle core,nlp] [-Python python.exe] [-Torch auto|cpu|cu118|cu121|cu124]

Bundles:
  core   -> psutil coverage pytest jsonschema structlog
  nlp    -> scikit-learn sentence-transformers numpy pandas transformers
  vision -> torch torchvision onnxruntime tensorrt
  devops -> boto3 google-cloud-logging google-cloud-storage docker redis
  qa     -> selenium duckduckgo-search playsound pypdf readability-lxml
  all    -> installs every bundle above

Torch variants (only affects the vision bundle):
  auto  -> choose CUDA wheel index when NVIDIA GPU is detected (via nvidia-smi)
  cpu   -> install CPU wheels from PyPI
  cu118 -> use PyTorch CUDA 11.8 wheel index
  cu121 -> use PyTorch CUDA 12.1 wheel index
  cu124 -> use PyTorch CUDA 12.4 wheel index
"@
}

if ($args -contains "-h" -or $args -contains "--help") {
    Show-Usage
    exit 0
}

if (-not $Python) {
    $Python = "python"
}

$packages = @{
    core   = @("psutil", "coverage", "pytest", "jsonschema", "structlog")
    nlp    = @("scikit-learn", "sentence-transformers", "numpy", "pandas", "transformers")
    vision = @("torch", "torchvision", "onnxruntime", "tensorrt")
    devops = @("boto3", "google-cloud-logging", "google-cloud-storage", "docker", "redis")
    qa     = @("selenium", "duckduckgo-search", "playsound", "pypdf", "readability-lxml")
}

function Resolve-TorchVariant {
    param([string] $Requested)
    if ($Requested -ne "auto") {
        return $Requested
    }

    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidia) {
        return "cpu"
    }

    try {
        $out = & nvidia-smi 2>$null
    } catch {
        return "cpu"
    }

    if ($out -match "CUDA Version:\\s*(\\d+)\\.(\\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -gt 12 -or ($major -eq 12 -and $minor -ge 4)) {
            return "cu124"
        }
        if ($major -eq 12 -and $minor -ge 1) {
            return "cu121"
        }
        if ($major -gt 11 -or ($major -eq 11 -and $minor -ge 8)) {
            return "cu118"
        }
    }

    return "cpu"
}

function Get-PackagesForBundle {
    param([string] $Name)
    if ($Name -eq "all") {
        return $packages.Keys | ForEach-Object { $packages[$_] } | Select-Object -Unique
    }
    if (-not $packages.ContainsKey($Name)) {
        Write-Error "Unknown bundle '$Name'"
        exit 1
    }
    return $packages[$Name]
}

$selected = @{}
foreach ($bundleName in $Bundle) {
    foreach ($pkg in (Get-PackagesForBundle -Name $bundleName)) {
        $selected[$pkg] = $true
    }
}

if ($selected.Count -eq 0) {
    Write-Host "Nothing to install (empty bundle selection)."
    exit 0
}

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Python executable '$Python' not found on PATH."
    exit 1
}

$torchPackages = @("torch", "torchvision")
$needsTorch = $false
foreach ($torchPkg in $torchPackages) {
    if ($selected.ContainsKey($torchPkg)) {
        $needsTorch = $true
        $null = $selected.Remove($torchPkg)
    }
}

$installList = $selected.Keys
Write-Host "Using Python executable: $Python"

& $Python -m pip install --upgrade pip | Out-Null
if ($needsTorch) {
    $resolved = Resolve-TorchVariant -Requested $Torch
    Write-Host "Installing PyTorch (variant: $resolved)..."
    if ($resolved -eq "cpu") {
        & $Python -m pip install torch torchvision
    } else {
        & $Python -m pip install --extra-index-url "https://download.pytorch.org/whl/$resolved" torch torchvision
    }
}

if ($installList.Count -gt 0) {
    Write-Host "Installing packages: $($installList -join ', ')"
    & $Python -m pip install @installList
}

Write-Host "Optional dependencies installed successfully."
