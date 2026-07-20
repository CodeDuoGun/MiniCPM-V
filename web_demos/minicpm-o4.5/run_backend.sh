#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "${ENV_FILE:-$SCRIPT_DIR/.env}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE:-$SCRIPT_DIR/.env}"
    set +a
fi

cd "$PROJECT_DIR"
UVICORN_LOG_LEVEL="$(printf '%s' "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"
exec python -m uvicorn --app-dir "$SCRIPT_DIR" app.server:app \
    --host 0.0.0.0 --port "${PORT:-32560}" --log-level "$UVICORN_LOG_LEVEL"
