#!/usr/bin/env python3
"""Run MLX LoRA with Twin chat-template kwargs pinned onto the tokenizer."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pin_chat_template_args(tokenizer, extra):
    """Force Twin template kwargs without recursing through MLX TokenizerWrapper.

    TokenizerWrapper.__setattr__ forwards public attributes onto the inner
    Hugging Face tokenizer. Replacing apply_chat_template on the wrapper
    therefore patches the inner method, while the captured original still
    calls that same inner method. Extra kwargs win over MLX defaults.
    """
    extra = dict(extra or {})
    if not extra:
        return tokenizer
    original = tokenizer.apply_chat_template

    def apply_chat_template(*args, **kw):
        return original(*args, **{**kw, **extra})

    object.__setattr__(tokenizer, "apply_chat_template", apply_chat_template)
    return tokenizer


def clamp_prompt_mask(tokens, offset):
    """If the generation prompt is longer than the example, train on the full sequence."""
    if offset >= len(tokens):
        return tokens, 0
    return tokens, offset


def _patch():
    raw = os.environ.get("TWIN_CHAT_TEMPLATE_ARGS") or ""
    extra = json.loads(raw) if raw.strip() else {}
    from mlx_lm.tuner.datasets import ChatDataset

    original_process = ChatDataset.process

    def process(self, d):
        tokens, offset = original_process(self, d)
        return clamp_prompt_mask(tokens, offset)

    ChatDataset.process = process
    if not extra:
        return
    import mlx_lm.lora as lora

    original_load = lora.load

    def load(model, tokenizer_config=None, **kwargs):
        loaded = original_load(model, tokenizer_config=tokenizer_config, **kwargs)
        model_obj, tokenizer = loaded
        pin_chat_template_args(tokenizer, extra)
        return model_obj, tokenizer

    lora.load = load


if __name__ == "__main__":
    _patch()
    from mlx_lm.lora import main

    main()
