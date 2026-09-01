#!/usr/bin/env bash
# Start / stop / check the Calibration Iteration Explorer (http://127.0.0.1:5058).
#
#   ./calibration_explorer/run.sh            start in the background (log: /tmp/calibration_explorer.log)
#   ./calibration_explorer/run.sh fg         start in the foreground (Ctrl-C to stop)
#   ./calibration_explorer/run.sh stop       stop a running instance
#   ./calibration_explorer/run.sh status     is it up, and is the data loaded?
#
# Interpreter resolution: EXPLORER_PYTHON if set; else `uv run --extra explorer`
# (the canonical environment); else the first python found carrying flask+pandas+
# sklearn (historically ~/anaconda3/bin/python). The repo's *base* uv venv does NOT
# include Flask — the explorer's dependencies are the opt-in `explorer` extra.
set -euo pipefail
cd "$(dirname "$0")"
PORT=5058
LOG=/tmp/calibration_explorer.log

runner() {
  if [ -n "${EXPLORER_PYTHON:-}" ]; then
    echo "$EXPLORER_PYTHON app.py"
  elif command -v uv >/dev/null 2>&1 \
      && (cd .. && uv run --extra explorer python -c 'import flask' >/dev/null 2>&1); then
    echo "uv run --extra explorer python calibration_explorer/app.py"
  else
    for c in "$HOME/anaconda3/bin/python" python3 python; do
      if "$c" -c 'import flask, pandas, sklearn' >/dev/null 2>&1; then
        echo "$c app.py"; return
      fi
    done
    echo ""
  fi
}

case "${1:-start}" in
  stop)
    lsof -ti :$PORT | xargs kill 2>/dev/null && echo "stopped" || echo "nothing on :$PORT" ;;
  status)
    curl -s --max-time 3 "http://127.0.0.1:$PORT/api/status" \
      | python3 -c 'import json,sys;d=json.load(sys.stdin);print("up · ready:",d["ready"],"·",d["message"])' \
      2>/dev/null || echo "not running (start with: ./calibration_explorer/run.sh)" ;;
  start|fg)
    if lsof -ti :$PORT >/dev/null 2>&1; then
      echo "already running → http://127.0.0.1:$PORT"; exit 0
    fi
    CMD=$(runner)
    [ -z "$CMD" ] && { echo "no interpreter with flask+pandas+sklearn found." >&2
      echo "fix: uv sync --extra explorer --python 3.13   (or set EXPLORER_PYTHON)" >&2; exit 1; }
    # uv commands must run from the repo root; plain-python from this directory.
    case "$CMD" in uv\ *) cd ..;; esac
    echo "starting: $CMD"
    if [ "${1:-start}" = fg ]; then
      exec $CMD
    fi
    nohup $CMD > "$LOG" 2>&1 &
    echo "→ http://127.0.0.1:$PORT   (log: $LOG)"
    echo "data loads in the background ~1–2 min; the page polls until ready."
    echo "optional: pre-warm every configuration with calibration_explorer/warm_cache.py" ;;
  *) echo "usage: run.sh [start|fg|stop|status]"; exit 1 ;;
esac
