#!/usr/bin/env bash
# Start the two MiniCPM-OCR recognition servers and the speed-test page (port 8000).
#   :18499  plain model, 8 parallel slots  (short regions, batched on the GPU)
#   :18500  DSpark model, 1 slot           (tables, and all regions in Default mode)
# Both use the models in ../MiniCPM-OCR/models/ollama. The system Ollama on :11434 is left alone.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../MiniCPM-OCR" && pwd)"
GGUF="$REPO/models/MiniCPM-OCR-1.0-gguf"
PORT="${PORT:-8000}"
SLOTS="${OCR_FAST_SLOTS:-8}"
export OCR_FAST_SLOTS="$SLOTS"

start_ollama() {  # port parallel-slots log-name
  if curl -fs "http://127.0.0.1:$1/api/version" >/dev/null 2>&1; then return; fi
  echo "Starting Ollama on 127.0.0.1:$1 with $2 slot(s) (log: $REPO/$3)"
  OLLAMA_HOST="127.0.0.1:$1" OLLAMA_MODELS="$REPO/models/ollama" \
  OLLAMA_NUM_PARALLEL="$2" OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m \
    nohup ollama serve >>"$REPO/$3" 2>&1 &
  for _ in $(seq 1 30); do curl -fs "http://127.0.0.1:$1/api/version" >/dev/null 2>&1 && return; sleep 1; done
  echo "Ollama on :$1 did not start" >&2; exit 1
}
start_ollama 18499 "$SLOTS" ollama-fast.log
start_ollama 18500 1 ollama-draft.log

# Register both models once from the same GGUF files (both instances share OLLAMA_MODELS).
if [ ! -f "$GGUF/Modelfile" ]; then
  cat > "$GGUF/Modelfile" <<'EOF'
FROM ./MiniCPM-OCR-1_0-Q8_0.gguf
FROM ./mmproj-model-f16-4x.gguf
DRAFT ./dspark-MiniCPM-OCR-1_0-Q8_0.gguf
REQUIRES 0.40.12
PARAMETER num_ctx 8192
PARAMETER temperature 0
PARAMETER draft_num_predict 15
EOF
fi
if ! OLLAMA_HOST=127.0.0.1:18500 ollama show minicpm-ocr >/dev/null 2>&1; then
  OLLAMA_HOST=127.0.0.1:18500 ollama create minicpm-ocr -f "$GGUF/Modelfile"
fi
if ! OLLAMA_HOST=127.0.0.1:18499 ollama show minicpm-ocr-nodraft >/dev/null 2>&1; then
  grep -v -E '^DRAFT|draft_num_predict' "$GGUF/Modelfile" > "$GGUF/Modelfile.nodraft"
  OLLAMA_HOST=127.0.0.1:18499 ollama create minicpm-ocr-nodraft -f "$GGUF/Modelfile.nodraft"
fi

source "$REPO/.minicpm-ocr/bin/activate"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
cd "$HERE"
echo "Open http://127.0.0.1:$PORT"
exec uvicorn server:app --host 127.0.0.1 --port "$PORT"
