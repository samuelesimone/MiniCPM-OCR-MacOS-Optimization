"""Shared MiniCPM-OCR configuration for the speed-test server and the benchmark."""
from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

from minicpmvocr import load_config

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "MiniCPM-OCR"
MODELS = Path(os.environ.get("OCR_ASSETS_DIR", REPO / "models"))
RUNS = HERE / "runs"
CONFIG = str(files("minicpmvocr").joinpath("configs/minicpmvocr.yaml"))

OCR_URL = os.environ.get("OCR_URL", "http://127.0.0.1:18499")
OCR_MODEL = os.environ.get("OCR_MODEL", "minicpm-ocr")
NUM_CTX = int(os.environ.get("OCR_NUM_CTX", "8192"))
# Concurrent recognition requests; match OLLAMA_NUM_PARALLEL on the server.
WORKERS = int(os.environ.get("OCR_WORKERS", "1"))
LAYOUT_WORKERS = int(os.environ.get("OCR_LAYOUT_WORKERS", "1"))
UPRIGHT = os.environ.get("OCR_UPRIGHT", "1") != "0"


def build_cfg(path: Path, tag: str, limit: int = 0, *, workers: int | None = None,
              layout_workers: int | None = None, upright: bool | None = None) -> dict:
    up = MODELS / "paddle_upright"
    return load_config(CONFIG, overrides={
        "run": {"run_tag": tag, "mode": "oneshot_file", "run_root": str(RUNS)},
        "input": {"file_path": str(path), "file_paths": [], "manifest": "",
                  "image_dir": "", "files_dir": "", "limit": limit},
        "layout": {"enabled": True, "model_dir": str(MODELS / "PP-DocLayoutV3_infer"), "device": "cpu"},
        "upright": {
            "enabled": UPRIGHT if upright is None else upright,
            "model_dir": str(up / "PP-LCNet_x1_0_doc_ori"),
            "det_model_dir": str(up / "PP-OCRv5_mobile_det"),
            "textline_model_dir": str(up / "PP-LCNet_x1_0_textline_ori"),
        },
        "engine": {"type": "ollama", "server_url": "", "server_urls": [OCR_URL],
                   "model_name": OCR_MODEL, "num_ctx": NUM_CTX, "max_model_len": NUM_CTX,
                   "startup_timeout_sec": 60, "request_ready_timeout_sec": 30},
        "pipeline": {"overlap_layout": True,
                     "max_workers": WORKERS if workers is None else workers,
                     "layout_workers": LAYOUT_WORKERS if layout_workers is None else layout_workers},
        "logging": {"progress_log_every": 10**9},
    })
