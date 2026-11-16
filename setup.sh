#!/usr/bin/env bash
# One-time install on Apple Silicon: MiniCPM-OCR checkout, SDK environment and model files.
# Safe to rerun: every step is skipped when its result already exists.
# Needs: git, curl, uv, Ollama >= 0.40.12 (https://ollama.com/download).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$ROOT/MiniCPM-OCR"
GGUF="$REPO/models/MiniCPM-OCR-1.0-gguf"
HF=https://huggingface.co/openbmb

# 1. Upstream checkout (the SDK on main has stream_pages / load_models; PyPI 1.0.5 does not yet).
[ -d "$REPO/.git" ] || git clone --depth 1 https://github.com/OpenBMB/MiniCPM-OCR.git "$REPO"

# 2. SDK environment: SDK from the checkout, CPU PaddlePaddle for layout, page server packages.
if [ ! -x "$REPO/.minicpm-ocr/bin/python" ]; then
  uv venv "$REPO/.minicpm-ocr" --python 3.12 --seed
fi
PY="$REPO/.minicpm-ocr/bin/python"
if ! "$PY" -c 'import minicpmvocr, paddle, fastapi; from minicpmvocr import stream_pages' 2>/dev/null; then
  "$PY" -m pip install "$REPO/sdk[all]" "paddlex==3.4.2" "pypdfium2>=4" "paddlepaddle==3.2.1" \
    fastapi uvicorn python-multipart httpx
fi

# 3. Recognition model: Q8_0 main model, F16 4x vision projector, Q8_0 DSpark draft (~2 GB).
mkdir -p "$GGUF"
fetch() { [ -s "$GGUF/$2" ] || curl -fL --retry 3 -o "$GGUF/$2" "$HF/$1/resolve/main/$2"; }
fetch MiniCPM-OCR-1.0-gguf MiniCPM-OCR-1_0-Q8_0.gguf
fetch MiniCPM-OCR-1.0-gguf mmproj-model-f16-4x.gguf
fetch MiniCPM-OCR-1.0-DSpark-gguf dspark-MiniCPM-OCR-1_0-Q8_0.gguf

# 4. Layout model (PP-DocLayoutV3) and page-rotation models.
"$PY" - "$REPO/models" <<'EOF'
import sys
from pathlib import Path
from minicpmvocr.layouts.download import ensure_layout_model, ensure_upright_models
root = Path(sys.argv[1])
ensure_layout_model(root / "PP-DocLayoutV3_infer")
ensure_upright_models(root / "paddle_upright")
EOF

echo "Setup complete. Start with: ./speedtest/start.sh"
