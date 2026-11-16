"""Benchmark the MiniCPM-OCR pipeline with different settings.

Example:
  python bench.py ../samples/attention.pdf --pages 5 --workers 4 --layout-workers 2 --repeat 2
The recognition server's parallel slots (OLLAMA_NUM_PARALLEL) must be >= --workers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from minicpmvocr import load_models, stream_pages

from ocrcfg import build_cfg


def one_run(cfg, models) -> dict:
    t0 = time.perf_counter()
    layout_t, first_block, blocks, chars, md = [], None, 0, 0, {}
    for ev in stream_pages(cfg, resources=models, detail="block"):
        t = time.perf_counter() - t0
        if ev["event"] == "layout":
            layout_t.append(t)
        elif ev["event"] == "block":
            blocks += 1
            chars += len(ev.get("content") or "")
            first_block = first_block or t
        elif ev["event"] == "page":
            md[ev["index"]] = ev.get("markdown") or ""
    wall = time.perf_counter() - t0
    pages = len(layout_t)
    return {"wall": wall, "pages": pages, "s_per_page": wall / max(1, pages),
            "layout_done": layout_t[-1] if layout_t else None, "first_block": first_block,
            "blocks": blocks, "chars": chars, "chars_per_s": chars / wall,
            "markdown": "\n\n".join(md[k] for k in sorted(md))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--layout-workers", type=int, default=1)
    ap.add_argument("--layout", default="paddle", help="paddle | torch-cpu | torch-mps")
    ap.add_argument("--no-upright", action="store_true")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--save-md", default="")
    ap.add_argument("--route", default="", help="FAST,DRAFT model names: tables go to DRAFT")
    a = ap.parse_args()

    path = Path(a.input).resolve()
    mk = lambda i: build_cfg(path, f"bench_{int(time.time())}_{i}", a.pages, workers=a.workers,
                             layout_workers=a.layout_workers, upright=not a.no_upright)
    t = time.perf_counter()
    if a.layout == "paddle":
        models = load_models(mk("load"))
    else:
        from layout_mps import load_torch_models
        models = load_torch_models(mk("load"), device=a.layout.split("-", 1)[1])
    load_sec = time.perf_counter() - t
    if a.route:
        from engines import RoutingEngine
        fast, draft = a.route.split(",")
        models["engine"] = RoutingEngine(mk("engine"), fast_model=fast, draft_model=draft,
                                         fast_url="http://127.0.0.1:18499", draft_url="http://127.0.0.1:18500")
        models["engine"].start()

    runs = []
    for i in range(a.repeat):
        r = one_run(mk(i), models)
        runs.append(r)
        print(f"  run {i + 1}: {r['wall']:.2f}s  ({r['s_per_page']:.2f} s/page, layout done {r['layout_done']:.2f}s, "
              f"{r['blocks']} blocks, {r['chars_per_s']:.0f} chars/s)", flush=True)
    models.close()
    best = min(runs, key=lambda r: r["wall"])
    if a.save_md:
        Path(a.save_md).write_text(best["markdown"], encoding="utf-8")
    summary = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items() if k != "markdown"}
    summary.update(label=a.label, route=a.route, layout=a.layout, workers=a.workers, layout_workers=a.layout_workers,
                   upright=not a.no_upright, model_load_sec=round(load_sec, 2))
    print("RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
