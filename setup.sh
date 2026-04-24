#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  Hamlet QA Failure Analysis - Setup"
echo "========================================"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "No suitable Python interpreter found."
    exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
case "$PYTHON_VERSION" in
    3.10|3.11|3.12)
        ;;
    *)
        echo "Selected interpreter: $PYTHON_BIN ($PYTHON_VERSION)"
        echo "Please use Python 3.10, 3.11, or 3.12 for the vLLM stack."
        exit 1
        ;;
esac

echo "Using Python interpreter: $PYTHON_BIN ($PYTHON_VERSION)"

if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment ..."
    "$PYTHON_BIN" -m venv venv
else
    echo "[1/3] Virtual environment already exists."
fi

source venv/bin/activate

echo "[2/3] Installing dependencies ..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo "[3/3] Preparing local directories ..."
mkdir -p data runs

echo ""
echo "Setup complete."
echo "Activate with: source venv/bin/activate"
echo "Run tests with: python -m unittest discover -s tests"
