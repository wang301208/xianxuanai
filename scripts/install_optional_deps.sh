#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install optional dependency bundles so advanced agents do not silently skip work.

Usage:
  install_optional_deps.sh --bundle core,nlp [--python python3.11] [--torch auto]

Bundles:
  core    -> psutil coverage pytest jsonschema structlog
  nlp     -> scikit-learn sentence-transformers numpy pandas transformers
  vision  -> torch torchvision onnxruntime tensorrt
  devops  -> boto3 google-cloud-logging google-cloud-storage docker redis
  qa      -> selenium duckduckgo-search playsound pypdf readability-lxml
  all     -> installs every bundle above

Torch variants (only affects the vision bundle):
  auto    -> choose CUDA wheel index when NVIDIA GPU is detected (via nvidia-smi)
  cpu     -> install CPU wheels from PyPI
  cu118   -> use PyTorch CUDA 11.8 wheel index
  cu121   -> use PyTorch CUDA 12.1 wheel index
  cu124   -> use PyTorch CUDA 12.4 wheel index

Environment variables:
  PYTHON   Override Python executable (default: python3)
  TORCH_VARIANT  Override torch variant (default: auto)
EOF
}

PYTHON_BIN="${PYTHON:-python3}"
BUNDLE_LIST="core"
TORCH_VARIANT="${TORCH_VARIANT:-auto}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      [[ $# -ge 2 ]] || { echo "error: --bundle requires a value" >&2; usage; exit 1; }
      BUNDLE_LIST="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "error: --python requires a value" >&2; usage; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --torch)
      [[ $# -ge 2 ]] || { echo "error: --torch requires a value" >&2; usage; exit 1; }
      TORCH_VARIANT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage
      exit 1
      ;;
  esac
done

declare -A PACKAGES
PACKAGES[core]="psutil coverage pytest jsonschema structlog"
PACKAGES[nlp]="scikit-learn sentence-transformers numpy pandas transformers"
PACKAGES[vision]="torch torchvision onnxruntime tensorrt"
PACKAGES[devops]="boto3 google-cloud-logging google-cloud-storage docker redis"
PACKAGES[qa]="selenium duckduckgo-search playsound pypdf readability-lxml"

resolve_torch_variant() {
  local requested="$1"
  case "$requested" in
    auto|cpu|cu118|cu121|cu124) ;;
    *)
      echo "error: invalid torch variant '$requested' (expected auto|cpu|cu118|cu121|cu124)" >&2
      exit 1
      ;;
  esac

  if [[ "$requested" != "auto" ]]; then
    echo "$requested"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi

  local cuda_version major minor
  cuda_version="$(
    nvidia-smi 2>/dev/null \
      | grep -Eo 'CUDA Version: [0-9]+\\.[0-9]+' \
      | head -n1 \
      | awk '{print $3}' \
      || true
  )"
  if [[ -z "$cuda_version" ]]; then
    echo "cpu"
    return
  fi

  IFS='.' read -r major minor <<<"$cuda_version"
  minor="${minor:-0}"

  if [[ "$major" -gt 12 || ( "$major" -eq 12 && "$minor" -ge 4 ) ]]; then
    echo "cu124"
    return
  fi
  if [[ "$major" -eq 12 && "$minor" -ge 1 ]]; then
    echo "cu121"
    return
  fi
  if [[ "$major" -gt 11 || ( "$major" -eq 11 && "$minor" -ge 8 ) ]]; then
    echo "cu118"
    return
  fi
  echo "cpu"
}

select_packages() {
  local bundle="$1"
  if [[ "$bundle" == "all" ]]; then
    echo "${PACKAGES[core]} ${PACKAGES[nlp]} ${PACKAGES[vision]} ${PACKAGES[devops]} ${PACKAGES[qa]}"
    return
  fi
  if [[ -z "${PACKAGES[$bundle]+x}" ]]; then
    echo "error: unknown bundle '$bundle'" >&2
    exit 1
  fi
  echo "${PACKAGES[$bundle]}"
}

IFS=',' read -ra BUNDLES <<< "$BUNDLE_LIST"
declare -A SEEN
INSTALL_LIST=()
for bundle in "${BUNDLES[@]}"; do
  for pkg in $(select_packages "$bundle"); do
    if [[ -z "${SEEN[$pkg]+x}" ]]; then
      INSTALL_LIST+=("$pkg")
      SEEN[$pkg]=1
    fi
  done
done

if [[ ${#INSTALL_LIST[@]} -eq 0 ]]; then
  echo "Nothing to install (empty bundle list)." >&2
  exit 0
fi

echo "Using Python executable: $PYTHON_BIN"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found on PATH" >&2
  exit 1
fi

NEEDS_TORCH=false
FILTERED_LIST=()
for pkg in "${INSTALL_LIST[@]}"; do
  case "$pkg" in
    torch|torchvision)
      NEEDS_TORCH=true
      ;;
    *)
      FILTERED_LIST+=("$pkg")
      ;;
  esac
done

"$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
if [[ "$NEEDS_TORCH" == true ]]; then
  TORCH_RESOLVED="$(resolve_torch_variant "$TORCH_VARIANT")"
  echo "Installing PyTorch (variant: $TORCH_RESOLVED)..."
  if [[ "$TORCH_RESOLVED" == "cpu" ]]; then
    "$PYTHON_BIN" -m pip install torch torchvision
  else
    "$PYTHON_BIN" -m pip install --extra-index-url "https://download.pytorch.org/whl/$TORCH_RESOLVED" torch torchvision
  fi
fi

if [[ ${#FILTERED_LIST[@]} -gt 0 ]]; then
  echo "Installing packages: ${FILTERED_LIST[*]}"
  "$PYTHON_BIN" -m pip install "${FILTERED_LIST[@]}"
fi

echo "Optional dependencies installed successfully."
