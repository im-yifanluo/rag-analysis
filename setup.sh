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

python_minor() {
    "$1" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'
}

ensure_python_version() {
    local actual
    actual="$("$@" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
    if [ "$actual" != "$PYTHON_VERSION" ]; then
        echo "Selected Python is $actual, but this repo expects Python $PYTHON_VERSION."
        return 1
    fi
}

install_pip_dependencies() {
    local python_cmd=("$@")
    "${python_cmd[@]}" -m pip install --upgrade pip --quiet
    "${python_cmd[@]}" -m pip install -r requirements.txt --quiet
}

print_miniforge_help() {
    cat <<EOF
Python $PYTHON_VERSION was not found, and conda/mamba was not found either.

Recommended fix on the server:

  curl -L -o Miniforge3-Linux-x86_64.sh \\
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "$MINIFORGE_PREFIX"
  export PATH="$MINIFORGE_PREFIX/bin:\$PATH"
  bash setup.sh

After setup completes, activate with:

  source "$MINIFORGE_PREFIX/etc/profile.d/conda.sh"
  conda activate $CONDA_ENV_NAME

If your cluster provides Miniforge as a module instead, load that module and
rerun: bash setup.sh
EOF
}

CONDA_CMD=""
if command -v mamba >/dev/null 2>&1; then
    CONDA_CMD="mamba"
elif command -v conda >/dev/null 2>&1; then
    CONDA_CMD="conda"
elif [ -x "$MINIFORGE_PREFIX/bin/mamba" ]; then
    CONDA_CMD="$MINIFORGE_PREFIX/bin/mamba"
elif [ -x "$MINIFORGE_PREFIX/bin/conda" ]; then
    CONDA_CMD="$MINIFORGE_PREFIX/bin/conda"
elif [ -x "$HOME/mambaforge/bin/mamba" ]; then
    CONDA_CMD="$HOME/mambaforge/bin/mamba"
elif [ -x "$HOME/mambaforge/bin/conda" ]; then
    CONDA_CMD="$HOME/mambaforge/bin/conda"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
fi

CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
elif [ -x "$MINIFORGE_PREFIX/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$MINIFORGE_PREFIX"
elif [ -n "$CONDA_CMD" ]; then
    CONDA_BIN_DIR="$(dirname "$CONDA_CMD")"
    CONDA_BASE="$(cd "$CONDA_BIN_DIR/.." && pwd)"
fi

if [ -n "$CONDA_CMD" ]; then
    echo "Using $CONDA_CMD to manage environment: $CONDA_ENV_NAME"
    if "$CONDA_CMD" env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
        echo "[1/3] Conda environment already exists."
    else
        echo "[1/3] Creating conda environment with Python $PYTHON_VERSION ..."
        "$CONDA_CMD" env create -f environment.yml
    fi

    echo "[2/3] Installing dependencies with pip inside conda env ..."
    ensure_python_version "$CONDA_CMD" run -n "$CONDA_ENV_NAME" python
    install_pip_dependencies "$CONDA_CMD" run -n "$CONDA_ENV_NAME" python

    echo "[3/3] Preparing local directories ..."
    mkdir -p data runs

    echo ""
    echo "Setup complete."
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo "Activate with:"
        echo "  source \"$CONDA_BASE/etc/profile.d/conda.sh\""
        echo "  conda activate $CONDA_ENV_NAME"
        echo "Run tests with: python -m unittest discover -s tests"
    else
        echo "Could not locate conda.sh for shell activation."
        echo "Run commands through the env with:"
        echo "  $CONDA_CMD run -n $CONDA_ENV_NAME python -m unittest discover -s tests"
    fi
    exit 0
fi

PYTHON_BIN=""
for candidate in python3.12 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if [ "$(python_minor "$candidate")" = "$PYTHON_VERSION" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    print_miniforge_help
    exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment ..."
    "$PYTHON_BIN" -m venv venv
else
    echo "[1/3] Virtual environment already exists."
fi

ensure_python_version venv/bin/python

echo "[2/3] Installing dependencies ..."
install_pip_dependencies venv/bin/python

echo "[3/3] Preparing local directories ..."
mkdir -p data runs

echo ""
echo "Setup complete."
echo "Activate with: source venv/bin/activate"
echo "Run tests with: python -m unittest discover -s tests"
