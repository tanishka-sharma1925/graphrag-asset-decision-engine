#!/bin/bash
cd "$(dirname "$0")"

echo "Starting HVAC dashboard backend..."
echo "Working directory: $(pwd)"

if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated .venv"
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo ""
    echo "ERROR: GROQ_API_KEY is not set."
    echo "Add this line to ~/.zshrc, then reopen this script:"
    echo '  export GROQ_API_KEY="your-key-here"'
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

echo "GROQ_API_KEY is set. Launching server.py..."
echo "Leave this window open. Press Ctrl+C or close this window to stop the server."
echo ""

python3 server.py

read -p "Server stopped. Press Enter to close this window..."
