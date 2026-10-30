#!/usr/bin/env bash
# One-command, project-scoped installation. No containers or service autostart.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
download_only=0
notebooks=1
for arg in "$@"; do
  case "$arg" in
    --download-only) download_only=1 ;;
    --no-notebooks) notebooks=0 ;;
    -h|--help)
      printf '%s\n' 'Usage: ./setup.sh [--download-only] [--no-notebooks]' 'Default: install locked dependencies, download and verify public inputs, train missing models, test, execute notebooks 11-15.' 'No service is automatically started.'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
if [[ "$(uname -s)" == Darwin ]]; then
  major="$(sw_vers -productVersion | cut -d. -f1)"
  if [[ "$(uname -m)" != arm64 || "$major" -lt 14 ]]; then
    printf '%s\n' 'This macOS lockfile requires Apple Silicon and macOS 14 or later.' >&2
    exit 1
  fi
  if [[ ! -f /opt/homebrew/opt/libomp/lib/libomp.dylib ]]; then
    printf '%s\n' 'Install the OpenMP runtime first: brew install libomp' >&2
    exit 1
  fi
fi
mkdir -p logs/ipython logs/matplotlib
export UV_CACHE_DIR="$ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
export IPYTHONDIR="$ROOT/logs/ipython"
export MPLCONFIGDIR="$ROOT/logs/matplotlib"
export JUPYTER_DATA_DIR="$ROOT/.venv/share/jupyter"
export JUPYTER_RUNTIME_DIR="$ROOT/logs/jupyter-runtime"
usable_uv() {
  command -v uv >/dev/null 2>&1 || return 1
  local version major minor patch
  version="$(uv --version | cut -d' ' -f2)"
  IFS=. read -r major minor patch <<< "$version"
  [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ && "$patch" =~ ^[0-9]+$ ]] || return 1
  (( major > 0 || minor > 13 || (minor == 13 && patch >= 6) ))
}
if usable_uv; then
  UV="$(command -v uv)"
else
  command -v python3 >/dev/null 2>&1 || { printf '%s\n' 'Install Python 3 or uv first.' >&2; exit 1; }
  if [[ ! -x .setup-tools/bin/uv ]]; then
    python3 -m venv .setup-tools
    .setup-tools/bin/python -m pip install --index-url https://pypi.org/simple 'uv==0.13.6'
  fi
  UV="$ROOT/.setup-tools/bin/uv"
fi
"$UV" sync --frozen --group dev --group notebooks
PY="$ROOT/.venv/bin/python"
"$PY" download_public_assets.py
"$PY" -m reclab.competition prepare
if [[ "$download_only" == 1 ]]; then
  printf '%s\n' 'Public inputs and reference sources verified. Training was not run.'
  exit 0
fi
"$PY" scripts/bootstrap.py
if ! "$PY" - <<'PY'
from pathlib import Path
import json
base = Path('artifacts/xmrec')
try:
    latest = json.loads((base / 'latest.json').read_text())['run_id']
    receipt = json.loads((base / latest / 'receipt.json').read_text())
    assert receipt['status'] == 'PASS'
except (OSError, ValueError, KeyError, AssertionError):
    raise SystemExit(1)
PY
then
  "$PY" -m reclab.competition run --tag clone-setup
fi
"$PY" -m ipykernel install --prefix "$ROOT/.venv" --name reclab --display-name 'Recommendation Lab (Python 3.12)'
"$PY" -m compileall -q reclab scripts tests
"$PY" -m pytest -q
"$PY" scripts/verify_restored_lab.py
if [[ "$notebooks" == 1 ]]; then
  "$PY" scripts/execute_notebooks.py --only 11_XMRec_Start 12_XMRec_Features 13_XMRec_Graph 14_XMRec_Ranking 15_XMRec_Evaluation
fi
printf '%s\n' 'Setup verified. Run ./lab.sh up, open http://127.0.0.1:18580, then click the Notebook course link.'
