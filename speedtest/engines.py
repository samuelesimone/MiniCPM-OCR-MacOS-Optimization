"""Route each region to the recognition model that is fastest for it on Apple Silicon.

Measured on an M3 Pro with Ollama 0.40.16:
- Tables (long, very predictable HTML) decode ~13x faster with DSpark speculative
  decoding than without it, but DSpark slows down when several requests share the GPU.
- Short text regions are fastest as many parallel requests to the plain model, which
  Ollama batches together.
So tables go to the DSpark model and everything else to the plain model.
Both models produce the same text at temperature 0.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict

from minicpmvocr import create_backend


def _is_table(prompt: str) -> bool:
    return prompt.strip().lower().startswith("table")


class RoutingEngine:
    name = "routing"

    def __init__(self, cfg: Dict, fast_model: str, draft_model: str, fast_url: str, draft_url: str,
                 use_draft: Callable[[str], bool] = _is_table):
        # Ollama keeps only one of the two models loaded per server and swaps them on
        # every switch, so each model needs its own Ollama instance (own URL).
        def backend(model: str, url: str):
            c = deepcopy(cfg)
            c["engine"].update(model_name=model, server_url="", server_urls=[url])
            return create_backend(c)
        self.fast = backend(fast_model, fast_url)
        self.draft = backend(draft_model, draft_url)
        self.use_draft = use_draft

    def start(self) -> None:
        self.fast.start()
        self.draft.start()

    def stop(self) -> None:
        self.fast.stop()
        self.draft.stop()

    def recognize(self, image, prompt: str, *args, **kwargs) -> Dict:
        engine = self.draft if self.use_draft(prompt) else self.fast
        return engine.recognize(image, prompt, *args, **kwargs)
