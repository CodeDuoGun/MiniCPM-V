#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f key.pem ] || [ ! -f cert.pem ]; then
    bash make_ssl_cert.sh
fi

pnpm run dev
