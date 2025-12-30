#!/usr/bin/env bash
set -euo pipefail

# AutoGPT 项目环境设置脚本
#
# 主要功能:
#   - 检测操作系统/架构
#   - 检测 Python 版本（默认要求 >= 3.10）
#   - 缺失或版本过低时，使用 pyenv 安装指定 Python（默认 3.11.5）
#   - 安装 Poetry
#   - 自动安装项目依赖（poetry/pip/conda）
#   - 检查/下载模型与数据资源（assets）
#   - 检查/安装系统工具依赖（system deps）
#
# 用法:
#   ./setup.sh [--check-only] [--skip-system-deps] [--skip-deps] [--skip-assets] [--fetch-assets] [--assets-manifest config/assets.json] [--yes] [--min-python 3.10] [--python-version 3.11.5]
#
# 环境变量:
#   MIN_PYTHON_VERSION   最低 Python 版本要求（默认 3.10）
#   PYTHON_VERSION       需要安装的 Python 版本（默认 3.11.5）

MIN_PYTHON_VERSION="${MIN_PYTHON_VERSION:-3.10}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11.5}"
CHECK_ONLY=false
INSTALL_DEPS=true
INSTALL_SYSTEM_DEPS=true
SKIP_ASSETS=false
FETCH_ASSETS=false
ASSETS_YES=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSETS_MANIFEST="${ASSETS_MANIFEST:-$REPO_ROOT/config/assets.json}"

usage() {
  cat <<'EOF'
Set up Python + Poetry for this repo.

Usage:
  ./setup.sh [--check-only] [--skip-system-deps] [--skip-deps] [--skip-assets] [--fetch-assets] [--assets-manifest config/assets.json] [--yes] [--min-python 3.10] [--python-version 3.11.5]

Environment variables:
  MIN_PYTHON_VERSION   Minimum Python version required (default: 3.10)
  PYTHON_VERSION       Python version to install via pyenv when needed (default: 3.11.5)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=true
      shift
      ;;
    --skip-system-deps)
      INSTALL_SYSTEM_DEPS=false
      shift
      ;;
    --skip-deps)
      INSTALL_DEPS=false
      shift
      ;;
    --skip-assets)
      SKIP_ASSETS=true
      shift
      ;;
    --fetch-assets)
      FETCH_ASSETS=true
      shift
      ;;
    --assets-manifest)
      [[ $# -ge 2 ]] || { echo "error: --assets-manifest requires a value" >&2; usage; exit 1; }
      ASSETS_MANIFEST="$2"
      shift 2
      ;;
    --yes)
      ASSETS_YES=true
      shift
      ;;
    --min-python)
      [[ $# -ge 2 ]] || { echo "error: --min-python requires a value" >&2; usage; exit 1; }
      MIN_PYTHON_VERSION="$2"
      shift 2
      ;;
    --python-version)
      [[ $# -ge 2 ]] || { echo "error: --python-version requires a value" >&2; usage; exit 1; }
      PYTHON_VERSION="$2"
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

if [[ "$ASSETS_MANIFEST" != /* ]]; then
  ASSETS_MANIFEST="$REPO_ROOT/$ASSETS_MANIFEST"
fi

detect_platform() {
  local sys arch
  sys="$(uname -s 2>/dev/null || echo unknown)"
  arch="$(uname -m 2>/dev/null || echo unknown)"
  echo "Detected OS: ${sys} (${OSTYPE:-unknown}), arch: ${arch}"
}

detect_cuda() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cuda_version gpu_line
    gpu_line="$(nvidia-smi -L 2>/dev/null | head -n1 || true)"
    cuda_version="$(
      nvidia-smi 2>/dev/null \
        | grep -Eo 'CUDA Version: [0-9]+\\.[0-9]+' \
        | head -n1 \
        | awk '{print $3}' \
        || true
    )"
    echo "NVIDIA GPU: detected${gpu_line:+ (${gpu_line})}${cuda_version:+, CUDA: ${cuda_version}}"
    return 0
  fi
  echo "NVIDIA GPU: not detected (nvidia-smi not found)"
}

is_windows_like() {
  case "${OSTYPE:-}" in
    cygwin*|msys*|win32*) return 0 ;;
  esac
  case "$(uname -s 2>/dev/null || true)" in
    CYGWIN*|MINGW*|MSYS*) return 0 ;;
  esac
  return 1
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command '$cmd' not found on PATH" >&2
    exit 1
  fi
}

status_ok() {
  echo "✅ $1"
}

status_warn() {
  echo "⚠️ $1"
}

status_info() {
  echo "ℹ️ $1"
}

have_privileges() {
  [[ "$(id -u)" -eq 0 ]] || command -v sudo >/dev/null 2>&1
}

run_privileged() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

detect_package_manager() {
  if command -v apt-get >/dev/null 2>&1; then
    echo "apt"
    return 0
  fi
  if command -v dnf >/dev/null 2>&1; then
    echo "dnf"
    return 0
  fi
  if command -v yum >/dev/null 2>&1; then
    echo "yum"
    return 0
  fi
  if command -v pacman >/dev/null 2>&1; then
    echo "pacman"
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    echo "brew"
    return 0
  fi
  echo "none"
}

APT_UPDATED=false

install_packages() {
  local pm="$1"
  shift

  case "$pm" in
    apt|dnf|yum|pacman)
      if ! have_privileges; then
        status_warn "sudo/root is required to install system packages automatically."
        return 1
      fi
      ;;
  esac

  case "$pm" in
    apt)
      if [[ "$APT_UPDATED" != true ]]; then
        status_info "Updating apt package index..."
        run_privileged apt-get update
        APT_UPDATED=true
      fi
      run_privileged apt-get install -y --no-install-recommends "$@"
      ;;
    dnf)
      run_privileged dnf install -y "$@"
      ;;
    yum)
      run_privileged yum install -y "$@"
      ;;
    pacman)
      run_privileged pacman -Sy --noconfirm "$@"
      ;;
    brew)
      brew install "$@"
      ;;
    *)
      return 1
      ;;
  esac
}

ensure_cmd() {
  local cmd="$1"
  local pkg="$2"
  local required="${3:-false}"
  local pm="$4"

  if command -v "$cmd" >/dev/null 2>&1; then
    status_ok "$cmd 已安装"
    return 0
  fi

  if [[ "$CHECK_ONLY" == true ]]; then
    status_warn "$cmd 未找到"
    if [[ "$required" == true ]]; then
      return 1
    fi
    return 0
  fi

  if [[ "$INSTALL_SYSTEM_DEPS" != true ]]; then
    status_warn "$cmd 未找到"
    if [[ "$required" == true ]]; then
      status_warn "$cmd is required, but system-deps installation is disabled (--skip-system-deps)."
      return 1
    fi
    return 0
  fi

  if [[ "$pm" == "none" ]]; then
    status_warn "$cmd 未找到"
    status_warn "No supported package manager found; cannot auto-install '$cmd'."
    if [[ "$required" == true ]]; then
      return 1
    fi
    return 0
  fi

  status_warn "$cmd 未找到，正在安装（$pm）..."
  if install_packages "$pm" "$pkg"; then
    if command -v "$cmd" >/dev/null 2>&1; then
      status_ok "$cmd 已安装"
      return 0
    fi
    status_warn "$cmd 安装完成但仍不可用（PATH 未更新？）"
    if [[ "$required" == true ]]; then
      return 1
    fi
    return 0
  fi

  status_warn "$cmd 安装失败"
  if [[ "$required" == true ]]; then
    return 1
  fi
  return 0
}

ensure_system_deps() {
  if [[ "$INSTALL_SYSTEM_DEPS" != true ]]; then
    echo "Skipping system dependency check (--skip-system-deps)."
    return 0
  fi

  echo "Checking system tools..."
  local pm
  pm="$(detect_package_manager)"
  if [[ "$pm" == "none" ]]; then
    status_warn "No supported package manager found (apt/dnf/yum/pacman/brew). Auto-install disabled."
  else
    status_info "Using package manager: $pm"
  fi

  ensure_cmd curl curl true "$pm"
  ensure_cmd git git true "$pm"

  # Optional tools from Dockerfiles and common workflows.
  ensure_cmd ffmpeg ffmpeg false "$pm"
  ensure_cmd wget wget false "$pm"
  ensure_cmd jq jq false "$pm"

  if command -v docker >/dev/null 2>&1; then
    status_ok "docker 已安装"
    if docker compose version >/dev/null 2>&1; then
      status_ok "docker compose 已安装"
    else
      status_warn "docker compose 未找到（如需本地 compose 部署，请安装 Docker Compose v2）"
    fi
  else
    status_warn "docker 未找到（如需本地 compose 部署，请先安装 Docker）"
  fi
}

detect_dep_manager() {
  if [[ -f "$REPO_ROOT/environment.yml" ]]; then
    echo "conda:$REPO_ROOT/environment.yml"
    return 0
  fi
  if [[ -f "$REPO_ROOT/environment.yaml" ]]; then
    echo "conda:$REPO_ROOT/environment.yaml"
    return 0
  fi
  if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
    echo "pip:$REPO_ROOT/requirements.txt"
    return 0
  fi
  if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "poetry:$REPO_ROOT/pyproject.toml"
    return 0
  fi
  echo "none:"
}

install_project_deps() {
  if [[ "$CHECK_ONLY" == true ]]; then
    echo "check-only mode: skipping dependency installation."
    return 0
  fi
  if [[ "$INSTALL_DEPS" != true ]]; then
    echo "Skipping dependency installation (--skip-deps)."
    return 0
  fi

  local manager manifest
  IFS=':' read -r manager manifest < <(detect_dep_manager)
  case "$manager" in
    conda)
      require_cmd conda
      echo "Installing dependencies via conda (manifest: $manifest)..."
      if conda env update -f "$manifest" --prune; then
        echo "Conda environment updated."
      else
        echo "conda env update failed; attempting create..."
        conda env create -f "$manifest"
      fi
      ;;
    pip)
      require_cmd python3
      echo "Installing dependencies via pip (manifest: $manifest)..."
      python3 -m pip install --upgrade pip >/dev/null
      python3 -m pip install -r "$manifest"
      ;;
    poetry)
      require_cmd poetry
      echo "Installing dependencies via Poetry (manifest: $manifest)..."
      (cd "$REPO_ROOT" && poetry install --no-interaction)
      ;;
    none)
      echo "No dependency manifest found in repo root (pyproject.toml / requirements.txt / environment.yml)."
      ;;
    *)
      echo "error: unknown dependency manager '$manager'" >&2
      exit 1
      ;;
  esac
}

manage_assets() {
  if [[ "$SKIP_ASSETS" == true ]]; then
    echo "Skipping asset check (--skip-assets)."
    return 0
  fi

  local assets_script="$SCRIPT_DIR/fetch_assets.py"
  if [[ ! -f "$assets_script" ]]; then
    echo "Assets helper not found at $assets_script (skipping)."
    return 0
  fi

  if [[ ! -f "$ASSETS_MANIFEST" ]]; then
    echo "Assets manifest not found at $ASSETS_MANIFEST (skipping)."
    return 0
  fi

  if [[ "$CHECK_ONLY" == true || "$FETCH_ASSETS" != true ]]; then
    echo "Checking model/data assets (manifest: $ASSETS_MANIFEST)..."
    python3 "$assets_script" check --manifest "$ASSETS_MANIFEST" || true
    return 0
  fi

  echo "Fetching model/data assets (manifest: $ASSETS_MANIFEST)..."
  local -a cmd
  cmd=(fetch --manifest "$ASSETS_MANIFEST")
  if [[ "$ASSETS_YES" == true ]]; then
    cmd+=(--yes)
  fi
  python3 "$assets_script" "${cmd[@]}"
}

parse_version() {
  local ver="$1"
  local major minor patch
  IFS='.' read -r major minor patch <<<"$ver"
  patch="${patch:-0}"
  echo "$major" "$minor" "$patch"
}

python_satisfies_min() {
  local python_bin="$1"
  local major minor patch
  read -r major minor patch < <(parse_version "$MIN_PYTHON_VERSION")
  "$python_bin" -c "import sys; sys.exit(0 if sys.version_info >= (${major}, ${minor}, ${patch}) else 1)" \
    >/dev/null 2>&1
}

ensure_pyenv_ready() {
  export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
  export PATH="$PYENV_ROOT/bin:$PATH"

  if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)" >/dev/null 2>&1 || true
    return 0
  fi

  if [[ "$CHECK_ONLY" == true ]]; then
    echo "pyenv not found (check-only mode)." >&2
    exit 1
  fi

  require_cmd curl
  require_cmd git

  echo "pyenv not found; installing via https://pyenv.run ..."
  curl -fsSL https://pyenv.run | bash

  export PATH="$PYENV_ROOT/bin:$PATH"
  if ! command -v pyenv >/dev/null 2>&1; then
    echo "error: pyenv installation completed but 'pyenv' is still not on PATH." >&2
    echo "Please restart your shell and ensure PYENV_ROOT/bin is in PATH." >&2
    exit 1
  fi
  eval "$(pyenv init -)" >/dev/null 2>&1 || true
}

ensure_python() {
  if command -v python3 >/dev/null 2>&1; then
    local current_version
    current_version="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
    echo "Python found: ${current_version:-unknown}"
    if python_satisfies_min python3; then
      return 0
    fi
    echo "Python version is below ${MIN_PYTHON_VERSION}; will attempt pyenv install (${PYTHON_VERSION})."
  else
    echo "python3 not found; will attempt pyenv install (${PYTHON_VERSION})."
  fi

  if [[ "$CHECK_ONLY" == true ]]; then
    echo "check-only mode: skipping Python installation." >&2
    exit 1
  fi

  ensure_pyenv_ready

  if pyenv versions --bare 2>/dev/null | grep -qx "$PYTHON_VERSION"; then
    echo "pyenv already has Python ${PYTHON_VERSION}."
  else
    echo "Installing Python ${PYTHON_VERSION} via pyenv..."
    pyenv install "$PYTHON_VERSION"
  fi

  pyenv global "$PYTHON_VERSION"
  eval "$(pyenv init -)" >/dev/null 2>&1 || true

  if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 still not found after pyenv install." >&2
    echo "Ensure your shell initializes pyenv (adds \$PYENV_ROOT/shims to PATH)." >&2
    exit 1
  fi
}

ensure_poetry() {
  if command -v poetry >/dev/null 2>&1; then
    echo "Poetry found: $(poetry --version 2>/dev/null || echo unknown)"
    return 0
  fi

  if [[ "$CHECK_ONLY" == true ]]; then
    echo "Poetry not found (check-only mode)." >&2
    exit 1
  fi

  require_cmd curl
  echo "Poetry not found; installing..."
  curl -sSL https://install.python-poetry.org | python3 -

  export PATH="$HOME/.local/bin:$HOME/.poetry/bin:$PATH"
  if ! command -v poetry >/dev/null 2>&1; then
    echo "warning: Poetry installed but not found on PATH in this shell." >&2
    echo "You may need to add ~/.local/bin (or ~/.poetry/bin) to PATH and restart your shell." >&2
    return 0
  fi
  echo "Poetry installed: $(poetry --version 2>/dev/null || echo unknown)"
}

detect_platform
if is_windows_like; then
  echo "This script is for Linux/macOS (and WSL). For native Windows, run: scripts/setup.ps1" >&2
  exit 1
fi

detect_cuda

ensure_system_deps

ensure_python
ensure_poetry
install_project_deps
manage_assets

echo "Environment setup complete."
