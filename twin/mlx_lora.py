#!/usr/bin/env python3
"""Run MLX LoRA with Twin chat-template kwargs pinned onto the tokenizer."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _patch():
    raw = os.environ.get("TWIN_CHAT_TEMPLATE_ARGS") or ""
    extra = json.loads(raw) if raw.strip() else {}
    if not extra:
        return
    import mlx_lm.lora as lora

    original_load = lora.load

    def load(model, tokenizer_config=None, **kwargs):
        loaded = original_load(model, tokenizer_config=tokenizer_config, **kwargs)
        model_obj, tokenizer = loaded
        original = tokenizer.apply_chat_template

        def apply_chat_template(*args, **kw):
            return original(*args, **{**extra, **kw})

        tokenizer.apply_chat_template = apply_chat_template
        return model_obj, tokenizer

    lora.load = load


if __name__ == "__main__":
    _patch()
    from mlx_lm.lora import main

    main()
