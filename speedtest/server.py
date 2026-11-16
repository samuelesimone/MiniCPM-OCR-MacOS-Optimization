"""MiniCPM-OCR speed test server.

Loads the layout models once, then parses each uploaded document through the
MiniCPM-OCR SDK and streams timestamped events (layout / block / delta / page /
done) to the browser as NDJSON so the page can measure every stage live.

Two modes:
- optimized: 8 parallel requests to the plain model (Ollama :18499, 8 slots) and
  tables to the DSpark model (Ollama :18500); 2 layout workers.
- default:   one request at a time to the DSpark model (Ollama :18500); 1 layout worker.
             This is the setup from the official guide.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from minicpmvocr import create_backend, load_models, stream_pages

from engines import RoutingEngine
from ocrcfg import HERE, NUM_CTX, RUNS, build_cfg

FAST_URL = os.environ.get("OCR_FAST_URL", "http://127.0.0.1:18499")
DRAFT_URL = os.environ.get("OCR_DRAFT_URL", "http://127.0.0.1:18500")
FAST_MODEL = os.environ.get("OCR_FAST_MODEL", "minicpm-ocr-nodraft")
DRAFT_MODEL = os.environ.get("OCR_DRAFT_MODEL", "minicpm-ocr")
FAST_SLOTS = int(os.environ.get("OCR_FAST_SLOTS", "8"))  # must match OLLAMA_NUM_PARALLEL on :18499
UPLOADS = HERE / "uploads"
SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
MODES = {
    "optimized": {"workers": FAST_SLOTS, "layout_workers": 2,
                  "describe": f"{FAST_SLOTS} parallel requests + DSpark for tables · 2 layout workers"},
    "default": {"workers": 1, "layout_workers": 1,
                "describe": "1 request at a time with DSpark · 1 layout worker (official guide setup)"},
}

UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="MiniCPM-OCR speed test")
run_lock = threading.Lock()  # one loaded model set serves one run at a time
state: dict = {"models": None, "load_sec": None, "engines": {}}


def _cfg(path: Path, tag: str, mode: str, limit: int = 0) -> dict:
    m = MODES[mode]
    cfg = build_cfg(path, tag, limit, workers=m["workers"], layout_workers=m["layout_workers"])
    cfg["engine"].update(model_name=DRAFT_MODEL, server_url="", server_urls=[DRAFT_URL])
    return cfg


@app.on_event("startup")
def _startup() -> None:
    warm = HERE / "warmup.png"
    if not warm.exists():
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (800, 200), "white")
        ImageDraw.Draw(img).text((20, 80), "Warm up the MiniCPM-OCR pipeline.", fill="black")
        img.save(warm)
    t0 = time.perf_counter()
    state["models"] = load_models(_cfg(warm, "warmup", "optimized"))  # 2 layout detectors
    state["load_sec"] = round(time.perf_counter() - t0, 2)
    base = _cfg(warm, "warmup", "default")
    state["engines"]["default"] = create_backend(base)
    state["engines"]["optimized"] = RoutingEngine(base, fast_model=FAST_MODEL, draft_model=DRAFT_MODEL,
                                                  fast_url=FAST_URL, draft_url=DRAFT_URL)
    for engine in state["engines"].values():
        engine.start()
    print(f"[speedtest] layout models loaded in {state['load_sec']}s", flush=True)


@app.on_event("shutdown")
def _shutdown() -> None:
    for engine in state["engines"].values():
        engine.stop()
    if state["models"] is not None:
        state["models"].close()


def _resources(mode: str) -> dict:
    models = state["models"]
    n = MODES[mode]["layout_workers"]
    return {"detectors": models["detectors"][:n], "upright": models.get("upright"),
            "engine": state["engines"][mode]}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


def _ollama(url: str, model: str) -> dict:
    try:
        tags = httpx.get(f"{url}/api/tags", timeout=2).json().get("models", [])
        ps = httpx.get(f"{url}/api/ps", timeout=2).json().get("models", [])
        version = httpx.get(f"{url}/api/version", timeout=2).json().get("version")
        same = lambda m: m["name"].split(":")[0] == model
        return {"ready": any(map(same, tags)), "warm": any(map(same, ps)), "version": version}
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "warm": False, "error": str(exc)}


@app.get("/api/status")
def status() -> dict:
    fast, draft = _ollama(FAST_URL, FAST_MODEL), _ollama(DRAFT_URL, DRAFT_MODEL)
    return {"num_ctx": NUM_CTX, "layout_loaded": state["models"] is not None,
            "layout_load_sec": state["load_sec"], "busy": run_lock.locked(),
            "modes": {k: v["describe"] for k, v in MODES.items()},
            "fast": {**fast, "url": FAST_URL, "model": FAST_MODEL, "slots": FAST_SLOTS},
            "draft": {**draft, "url": DRAFT_URL, "model": DRAFT_MODEL},
            "ocr_ready": fast["ready"] and draft["ready"], "ollama_version": draft.get("version")}


@app.get("/api/preview/{job}/{page}")
def preview(job: str, page: int) -> Response:
    if not job.isalnum():
        raise HTTPException(404)
    folder = UPLOADS / job
    if not folder.is_dir():
        raise HTTPException(404)
    src = next(folder.iterdir())
    if src.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(src))
        if page >= len(pdf):
            raise HTTPException(404)
        buf = io.BytesIO()
        pdf[page].render(scale=1.5).to_pil().save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png")
    return FileResponse(src)


@app.get("/api/runimg/{job}/{path:path}")
def run_image(job: str, path: str) -> FileResponse:
    """Figure crops that the parsed Markdown links to as imgs/..."""
    if not job.isalnum():
        raise HTTPException(404)
    root = (RUNS / f"speed_{job}" / "output" / "imgs").resolve()
    target = (root / path).resolve()
    if not target.is_file() or root not in target.parents:
        raise HTTPException(404)
    return FileResponse(target)


def _clean(event: dict, t0: float) -> dict:
    out = {k: v for k, v in event.items() if k != "page"}
    out["t"] = round(time.perf_counter() - t0, 4)
    return out


def _crop_timings(run_dir: Path) -> list:
    """Exact per-region request time and token counts that the SDK records."""
    path = run_dir / "runtime" / "crop_timing.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            rows.append({"image_id": r["image_id"], "order": int(r["crop_filename"].split("_", 1)[0]),
                         "infer_sec": r.get("infer_sec"), "prompt_tokens": r.get("prompt_tokens"),
                         "completion_tokens": r.get("completion_tokens")})
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return rows


@app.post("/api/parse")
def parse(file: UploadFile = File(...), limit: int = Form(0), mode: str = Form("optimized")) -> StreamingResponse:
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type {suffix!r}. Use PDF or an image.")
    if mode not in MODES:
        raise HTTPException(400, f"Unknown mode {mode!r}")
    if not run_lock.acquire(blocking=False):
        raise HTTPException(409, "Another document is being parsed. Wait for it to finish.")
    try:
        job = uuid.uuid4().hex[:12]
        folder = UPLOADS / job
        folder.mkdir()
        safe_name = Path(file.filename).name
        path = folder / safe_name
        with path.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        cfg = _cfg(path, f"speed_{job}", mode, max(0, limit))
    except Exception:
        run_lock.release()
        raise

    def gen():
        t0 = time.perf_counter()
        try:
            yield json.dumps({"event": "start", "job": job, "file": safe_name, "mode": mode,
                              "describe": MODES[mode]["describe"], "workers": MODES[mode]["workers"],
                              "bytes": path.stat().st_size, "t": 0.0}) + "\n"
            for ev in stream_pages(cfg, ordered=False, resources=_resources(mode), detail="token"):
                yield json.dumps(_clean(ev, t0), default=str) + "\n"
            yield json.dumps({"event": "done", "t": round(time.perf_counter() - t0, 4),
                              "crops": _crop_timings(Path(cfg["_meta"]["run_dir"]))}, default=str) + "\n"
        except Exception as exc:  # noqa: BLE001
            yield json.dumps({"event": "error", "message": f"{type(exc).__name__}: {exc}",
                              "t": round(time.perf_counter() - t0, 4)}) + "\n"
        finally:
            run_lock.release()

    return StreamingResponse(gen(), media_type="application/x-ndjson")
