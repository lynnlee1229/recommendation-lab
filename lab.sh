#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
case "${1:-help}" in
  setup)
    shift
    exec ./setup.sh "$@"
    ;;
  up|start)
    .venv/bin/python scripts/manage.py start
    ;;
  status|stop)
    .venv/bin/python scripts/manage.py "$1"
    ;;
  test)
    .venv/bin/python -m pytest -q
    ;;
  train)
    shift
    .venv/bin/python -m reclab.cli train "$@"
    ;;
  notebook)
    .venv/bin/python scripts/manage.py start
    .venv/bin/python -c 'from reclab.notebook import ensure_server; ensure_server(); print("Notebook ready: http://127.0.0.1:18580/notebooks")'
    ;;
  notebook-status)
    .venv/bin/python -c 'import json; from reclab.notebook import status; print(json.dumps(status()))'
    ;;
  notebook-stop)
    .venv/bin/python -c 'import json; from reclab.notebook import stop; print(json.dumps(stop()))'
    ;;
  *)
    printf '%s\n' 'setup: download, prepare and train the first four experiments' 'up: start localhost workbench without opening a browser' 'status | stop: inspect or stop the dashboard' 'notebook: start the project JupyterLab with token authentication' 'notebook-status | notebook-stop: inspect or stop JupyterLab and its kernels' 'test: run numerical and real-data acceptance checks' 'train --config configs/logq.json: create a new model version'
    ;;
esac
