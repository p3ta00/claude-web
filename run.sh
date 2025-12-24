#!/bin/bash
# Claude Web - Run Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for venv
if [ ! -d "venv" ]; then
    echo "[!] Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Check for Claude CLI
if ! command -v claude &> /dev/null; then
    echo "[!] Claude Code CLI not found. Run ./setup.sh for instructions."
    exit 1
fi

# Activate venv and run
source venv/bin/activate

# Create data dir if missing
mkdir -p backend/data

echo ""
echo "[+] Starting Claude Web..."
echo "[*] Open http://localhost:8888 in your browser"
echo "[*] Press Ctrl+C to stop"
echo ""

cd backend
python main.py
