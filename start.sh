#!/usr/bin/env bash
# ==============================================================================
# Start script for Multimodal RAG
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "backend/app/main.py" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "multimodal-rag/backend/app/main.py" ]; then
    PROJECT_DIR="$SCRIPT_DIR/multimodal-rag"
else
    echo "[ERROR] Could not find backend/app/main.py"
    exit 1
fi

cd "$PROJECT_DIR"

# Locate python venv
if [ -f "venv/Scripts/python.exe" ]; then
    PYTHON_BIN="venv/Scripts/python.exe"
elif [ -f "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
else
    PYTHON_BIN="python"
fi

echo "========================================================"
echo "       Starting Multimodal RAG Application"
echo "========================================================"

# Start backend
echo "[1/3] Starting FastAPI Backend on http://127.0.0.1:8000..."
cd "$PROJECT_DIR/backend"
"$PYTHON_BIN" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Trap signals to kill child processes on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start frontend
echo "[2/3] Starting React Frontend on http://127.0.0.1:5173..."
cd "$PROJECT_DIR/frontend"
npm run dev -- --host 127.0.0.1 &
FRONTEND_PID=$!

sleep 3
echo "[3/3] Opening browser..."
if which xdg-open > /dev/null 2>&1; then
    xdg-open "http://127.0.0.1:5173" >/dev/null 2>&1 || true
elif which open > /dev/null 2>&1; then
    open "http://127.0.0.1:5173" >/dev/null 2>&1 || true
elif which start > /dev/null 2>&1; then
    start "http://127.0.0.1:5173" >/dev/null 2>&1 || true
fi

echo ""
echo "========================================================"
echo "  Multimodal RAG is running!"
echo "  - Frontend: http://127.0.0.1:5173"
echo "  - Backend:  http://127.0.0.1:8000"
echo "  - API Docs: http://127.0.0.1:8000/docs"
echo "========================================================"
echo "Press Ctrl+C to stop both servers."

wait $FRONTEND_PID
