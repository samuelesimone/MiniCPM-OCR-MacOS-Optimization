# MiniCPM-OCR speed test

A local page that parses a dropped PDF or image with MiniCPM-OCR 1.0 and shows how long each stage takes. It has two modes, so you can compare the official setup with the setup tuned for this Mac.

## Start

```bash
./start.sh          # then open http://127.0.0.1:8000
```

`start.sh` starts any of these that aren't already running:

| Port | What | Used for |
|---|---|---|
| 18499 | Ollama, plain model `minicpm-ocr-nodraft`, 8 parallel slots | Text, titles, formulas (Optimized mode) |
| 18500 | Ollama, DSpark model `minicpm-ocr`, 1 slot | Tables (Optimized mode), all regions (Default mode) |
| 8000 | The page (FastAPI) | |

The system Ollama on port 11434 is left alone. Both instances read the models in `../MiniCPM-OCR/models/ollama` and keep them loaded for 30 minutes.

## Modes

- **Optimized:** 8 regions are recognized in parallel by the plain model, tables go to the DSpark model, and 2 layout workers run.
- **Default:** one region at a time with the DSpark model and 1 layout worker. This is the setup from the official guide.

Both modes produce byte-identical Markdown (temperature 0).

## Why this layout (measured on M3 Pro, 36 GB)

"Attention Is All You Need" (arXiv 1706.03762), models already loaded:

| Setup | 5 pages | 15 pages |
|---|---|---|
| DSpark, 1 request at a time (official) | 23.0 s | 131.8 s |
| DSpark, 2 / 4 parallel requests | 31.4 / 52.1 s | – |
| Plain model, 4 parallel | 15.8 s | – |
| Plain model, 8 parallel | 12.4 s | 104.1 s |
| Plain model, 16 parallel | 13.0 s | – |
| **Plain ×8 + DSpark for tables, two Ollama instances** | **13.3 s** | **39.0 s** |

What the measurements showed:
- **DSpark doesn't get faster with parallel requests in Ollama.** Speculative decoding already keeps the GPU busy, and extra requests slow every stream down.
- **Without DSpark, parallel requests batch well.** Throughput rises up to 8 slots and is flat after that.
- **The SDK's worker count must match `OLLAMA_NUM_PARALLEL`.** For example, 8 workers on 16 slots took 25 s instead of 12 s.
- **Tables are the exception.** Their HTML output is very predictable, so DSpark decodes them about 13× faster. In the plain parallel run, one 681-token table took 48.6 s, compared with 3.7 s with DSpark. That's why tables are routed to DSpark.
- **The two models need separate Ollama instances.** A single Ollama keeps only one of them loaded, even with `OLLAMA_MAX_LOADED_MODELS=2`, and swaps on every switch. Tables waited 60–90 s each that way.
- **Layout isn't the bottleneck.** PP-DocLayoutV3 runs on the CPU at about 1 s/page with 2 workers and overlaps with recognition.

Rerun the benchmark with `bench.py`, for example:

```bash
source ../MiniCPM-OCR/.minicpm-ocr/bin/activate
python bench.py ../samples/attention.pdf --pages 15 --workers 8 --layout-workers 2 \
  --route minicpm-ocr-nodraft,minicpm-ocr
```

## What the numbers on the page mean

- **Total time** runs from when the server has the file until the last region is recognized. Upload time is shown separately.
- **Request time / tokens per region** come from the SDK's own records (`runtime/crop_timing.jsonl`), so they stay correct when regions run in parallel. "Image tokens" is the size of the region image as seen by the model.
- **The timeline** puts regions that overlap in time on separate rows. This includes time a request waits in Ollama's queue.

## Files

- `server.py`: page server with both modes
- `engines.py`: routes tables to DSpark and everything else to the parallel model
- `ocrcfg.py`: shared SDK configuration
- `bench.py`: command-line benchmark
- `static/index.html`: the page
