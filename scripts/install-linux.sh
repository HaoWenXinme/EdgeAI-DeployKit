#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
  ROOT="$SCRIPT_DIR"
else
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-"$ROOT/.venv"}"
WITH_TENSORFLOW=0
WITH_PYTORCH=0
WITH_ML=0
WITH_LLM=0
WITH_FRONTEND=1

usage() {
  cat <<'EOF'
Usage: scripts/install-linux.sh [options]

Options:
  --with-pytorch       Install PyTorch conversion dependencies.
  --with-tensorflow     Install TensorFlow/Keras conversion dependencies.
  --with-ml             Install sklearn/xgboost/lightgbm conversion dependencies.
  --with-llm            Install llama-cpp-python for GGUF chat inference.
  --no-frontend         Skip pnpm/Corepack frontend dependency install.
  --python PATH         Python executable to use. Default: python3.
  --venv PATH           Virtualenv directory. Default: .venv.
  -h, --help            Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-tensorflow)
      WITH_TENSORFLOW=1
      shift
      ;;
    --with-pytorch)
      WITH_PYTORCH=1
      shift
      ;;
    --with-ml)
      WITH_ML=1
      shift
      ;;
    --with-llm)
      WITH_LLM=1
      shift
      ;;
    --no-frontend)
      WITH_FRONTEND=0
      shift
      ;;
    --python)
      PYTHON_BIN="${2:?missing python path}"
      shift 2
      ;;
    --venv)
      VENV_DIR="${2:?missing venv path}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
  echo "Install Python 3.9+ first, then rerun this script." >&2
  exit 1
fi

echo "[EdgeAI] project root: $ROOT"
echo "[EdgeAI] python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable + " " + sys.version.split()[0])')"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[pdf]"

if [ "$WITH_PYTORCH" -eq 1 ]; then
  PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
  python -m pip install --index-url "$PYTORCH_INDEX_URL" torch torchvision
else
  echo "[EdgeAI] PyTorch conversion deps skipped. Use --with-pytorch to enable .pt/.pth conversion."
fi

if [ "$WITH_TENSORFLOW" -eq 1 ]; then
  python -m pip install tensorflow-cpu tf2onnx h5py
else
  echo "[EdgeAI] TensorFlow conversion deps skipped. Use --with-tensorflow to enable .h5/.keras/SavedModel conversion."
fi

if [ "$WITH_ML" -eq 1 ]; then
  python -m pip install ".[traditional-ml]"
else
  echo "[EdgeAI] Traditional ML deps skipped. Use --with-ml to enable sklearn/xgboost/lightgbm conversion."
fi

if [ "$WITH_LLM" -eq 1 ]; then
  python -m pip install ".[llm]"
else
  echo "[EdgeAI] LLM runtime deps skipped. Use --with-llm or install llama.cpp to enable GGUF chat."
fi

if [ "$WITH_FRONTEND" -eq 1 ]; then
  if command -v corepack >/dev/null 2>&1; then
    corepack enable || true
    (cd product-ui && corepack pnpm install --frozen-lockfile)
  elif command -v pnpm >/dev/null 2>&1; then
    (cd product-ui && pnpm install --frozen-lockfile)
  else
    echo "[WARN] pnpm/Corepack not found. Install Node.js 20+ with Corepack, then run:"
    echo "       cd product-ui && corepack pnpm install --frozen-lockfile"
  fi
fi

mkdir -p inputs/models inputs/images outputs/packages outputs/logs reports

python scripts/linux_doctor.py --json outputs/linux_doctor.json

cat <<EOF

[EdgeAI] Linux install complete.

Start backend:
  source "$VENV_DIR/bin/activate"
  ./start-backend.sh

Start UI:
  ./start-ui.sh

Or start both in one command:
  scripts/start-linux.sh --lan

Doctor report:
  outputs/linux_doctor.json
EOF
