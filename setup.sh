#!/bin/bash
# Claude Web - Setup Script
# CLI Wrapper for Claude Code - Uses your subscription, no API costs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗    ██╗    ██╗███████╗██████╗ "
echo " ██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝    ██║    ██║██╔════╝██╔══██╗"
echo " ██║     ██║     ███████║██║   ██║██║  ██║█████╗      ██║ █╗ ██║█████╗  ██████╔╝"
echo " ██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝      ██║███╗██║██╔══╝  ██╔══██╗"
echo " ╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗    ╚███╔███╔╝███████╗██████╔╝"
echo "  ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝     ╚══╝╚══╝ ╚══════╝╚═════╝ "
echo ""
echo "  Mobile-friendly Claude Interface"
echo "  Uses Claude Code CLI - No API costs!"
echo ""

# Check for Claude CLI
echo "[*] Checking for Claude Code CLI..."
if ! command -v claude &> /dev/null; then
    echo ""
    echo "[!] Claude Code CLI not found!"
    echo ""
    echo "    Install it first:"
    echo "    npm install -g @anthropic-ai/claude-code"
    echo ""
    echo "    Then run 'claude' to authenticate with your account."
    echo ""
    exit 1
fi

CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
echo "[+] Found Claude Code: $CLAUDE_VERSION"

# Check if authenticated (quick test)
echo ""
echo "[*] Checking Claude Code authentication..."
if ! timeout 10 claude --print --dangerously-skip-permissions "echo test" &>/dev/null; then
    echo ""
    echo "[!] Claude Code may not be authenticated or timed out."
    echo ""
    echo "    Run 'claude' in your terminal and complete the login flow."
    echo "    This will open your browser to authenticate."
    echo ""
    read -p "    Press Enter to continue anyway, or Ctrl+C to cancel..."
else
    echo "[+] Claude Code is authenticated!"
fi

# Check Python
echo ""
echo "[*] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "[+] Found $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "[*] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[+] Created virtual environment"
else
    echo "[+] Virtual environment exists"
fi

source venv/bin/activate

# Install/upgrade dependencies
echo ""
echo "[*] Installing dependencies..."
pip install --upgrade pip -q
pip install -q -r requirements.txt

# Create data directory
echo ""
echo "[*] Setting up data directory..."
mkdir -p backend/data
echo "[+] Data directory ready"

echo ""
echo "[+] Setup complete!"
echo ""
echo "============================================"
echo "  To start Claude Web:"
echo ""
echo "    ./run.sh"
echo ""
echo "  Then open: http://localhost:8888"
echo ""
echo "  Features:"
echo "    - Mobile-friendly interface"
echo "    - Chat history & sessions"
echo "    - Image upload (paste or attach)"
echo "    - Rename & delete conversations"
echo "    - Real-time streaming responses"
echo "============================================"
echo ""
