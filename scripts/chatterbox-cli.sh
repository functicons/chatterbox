#!/usr/bin/env bash
# Chatterbox TTS CLI (Resemble AI). Usage:
#   ./scripts/chatterbox-cli.sh "Hello, world." -o out.wav
#   ./scripts/chatterbox-cli.sh "Hello" -r myvoice.wav -e 1.2     # voice clone + emotion
#   echo "Hello" | ./scripts/chatterbox-cli.sh - -o out.wav
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/.venv/bin/python" "$SCRIPT_DIR/_chatterbox_cli.py" "$@"
