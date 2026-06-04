#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  Hamlet QA Failure Analysis - Setup"
echo "========================================"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hamlet-qa}"
MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-$HOME/miniforge3}"

print_miniforge_help() {
    cat <<EOF
Miniforge/conda was not found. This repo uses conda with Miniforge as the
environment manager.

Install Miniforge on Linux servers with:

  curl -L -o Miniforge3-Linux-x86_64.sh \\
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "$MINIFORGE_PREFIX"
  export PATH="$MINIFORGE_PREFIX/bin:\$PATH"
  bash setup.sh

On macOS, install Miniforge with Homebrew or the official installer, then rerun:

  brew install --cask miniforge
  bash setup.sh

After setup completes, activate with:

  source "$MINIFORGE_PREFIX/etc/profile.d/conda.sh"
  conda activate $CONDA_ENV_NAME
EOF
}

find_conda_cmd() {
    if command -v mamba >/dev/null 2>&1; then
        command -v mamba
    elif command -v conda >/dev/null 2>&1; then
        command -v conda
    elif [ -x "$MINIFORGE_PREFIX/bin/mamba" ]; then
        printf '%s\n' "$MINIFORGE_PREFIX/bin/mamba"
    elif [ -x "$MINIFORGE_PREFIX/bin/conda" ]; then
        printf '%s\n' "$MINIFORGE_PREFIX/bin/conda"
    else
        return 1
    fi
}

conda_base_for() {
    local conda_cmd="$1"
    if command -v conda >/dev/null 2>&1; then
        conda info --base 2>/dev/null || true
        return
    fi
    if [ -f "$MINIFORGE_PREFIX/etc/profile.d/conda.sh" ]; then
        printf '%s\n' "$MINIFORGE_PREFIX"
        return
    fi
    local conda_bin_dir
    conda_bin_dir="$(dirname "$conda_cmd")"
    (cd "$conda_bin_dir/.." && pwd)
}

CONDA_CMD="$(find_conda_cmd || true)"
if [ -z "$CONDA_CMD" ]; then
    print_miniforge_help
    exit 1
fi

CONDA_BASE="$(conda_base_for "$CONDA_CMD")"
echo "Using $CONDA_CMD to manage environment: $CONDA_ENV_NAME"

if "$CONDA_CMD" env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
    echo "[1/3] Conda environment exists; updating from environment.yml ..."
else
    echo "[1/3] Creating conda environment: $CONDA_ENV_NAME ..."
    "$CONDA_CMD" create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION" pip
fi

echo "[2/3] Syncing dependencies from environment.yml ..."
"$CONDA_CMD" env update -n "$CONDA_ENV_NAME" -f environment.yml --prune

ACTUAL_PYTHON="$("$CONDA_CMD" run -n "$CONDA_ENV_NAME" python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
if [ "$ACTUAL_PYTHON" != "$PYTHON_VERSION" ]; then
    echo "Environment Python is $ACTUAL_PYTHON, expected $PYTHON_VERSION."
    exit 1
fi

echo "[3/3] Preparing local directories ..."
mkdir -p data runs

echo ""
echo "Setup complete."
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Activate with:"
    echo "  source \"$CONDA_BASE/etc/profile.d/conda.sh\""
    echo "  conda activate $CONDA_ENV_NAME"
else
    echo "Activate with: conda activate $CONDA_ENV_NAME"
fi
echo "Run tests with:"
echo "  conda run -n $CONDA_ENV_NAME python -m unittest discover -s tests"
