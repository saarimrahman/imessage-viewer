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


ASSISTANT_HEADER = (151644, 77091, 198)
IM_END = 151645


def assistant_spans(ids, header=ASSISTANT_HEADER, end_id=IM_END):
    """Mark every token the twin itself must produce.

    ChatML writes each of my turns as `<|im_start|>assistant\n ... <|im_end|>`.
    `--mask-prompt` trains on the last such turn only, so a 4-turn window taught
    the model one reply and used the other three as context. This marks all of
    them, including the closing `<|im_end|>` so the model still learns to stop.
    """
    mask = [0] * len(ids)
    i = 0
    while i + 2 < len(ids):
        if (ids[i], ids[i + 1], ids[i + 2]) == header:
            j = i + 3
            while j < len(ids) and ids[j] != end_id:
                mask[j] = 1
                j += 1
            if j < len(ids):
                mask[j] = 1
            i = j + 1
        else:
            i += 1
    return mask


def assistant_mask_mx(ids):
    """Same spans as `assistant_spans`, built with MLX ops only.

    The loss runs inside `nn.value_and_grad`, so the token ids cannot be moved
    to numpy: MLX refuses to eval an array during a function transformation.
    A running count of opened minus closed turns gives the same mask.
    """
    import mlx.core as mx

    rows, length = ids.shape
    header = (
        (ids[:, :-2] == ASSISTANT_HEADER[0])
        & (ids[:, 1:-1] == ASSISTANT_HEADER[1])
        & (ids[:, 2:] == ASSISTANT_HEADER[2])
    )
    # Content begins three tokens after the header starts.
    starts = mx.concatenate(
        [mx.zeros((rows, 3), dtype=mx.int32), header.astype(mx.int32)], axis=1
    )[:, :length]
    ends = ids == IM_END
    # A running count does not work: system and user turns also close with
    # `<|im_end|>`, so unmatched ends drive it negative. Compare the most recent
    # assistant start against the most recent end instead.
    position = mx.arange(length, dtype=mx.int32)
    last_start = mx.cummax(mx.where(starts > 0, position, -1), axis=1)
    last_end = mx.cummax(mx.where(ends, position, -1), axis=1)
    # Ends strictly before this token, so a closing `<|im_end|>` stays inside
    # its own turn and the model still learns to stop.
    before = mx.concatenate(
        [mx.full((rows, 1), -1, dtype=mx.int32), last_end[:, :-1]], axis=1
    )
    return last_start > before


def multiturn_loss(model, batch, lengths):
    """Cross entropy over every assistant turn, not only the last one."""
    import mlx.core as mx
    import mlx.nn as nn

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)

    # `targets[:, t]` is `batch[:, t + 1]`, so the token mask shifts by one.
    span = assistant_mask_mx(batch)[:, 1:]
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(span, steps <= lengths[:, 1:])

    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / mx.maximum(ntoks, 1)
    return ce, ntoks


def _patch():
    raw = os.environ.get("TWIN_CHAT_TEMPLATE_ARGS") or ""
    extra = json.loads(raw) if raw.strip() else {}
    from mlx_lm.tuner.datasets import ChatDataset
    from twin.export import sanitize_chat_template

    original_process = ChatDataset.process

    def process(self, d):
        tokens, offset = original_process(self, d)
        return clamp_prompt_mask(tokens, offset)

    ChatDataset.process = process

    import mlx_lm.lora as lora

    original_load = lora.load

    def load(model, tokenizer_config=None, **kwargs):
        loaded = original_load(model, tokenizer_config=tokenizer_config, **kwargs)
        model_obj, tokenizer = loaded
        if sanitize_chat_template(tokenizer):
            print("[twin] removed the empty think block from the training target")
        if extra:
            pin_chat_template_args(tokenizer, extra)
        return model_obj, tokenizer

    lora.load = load

    if os.environ.get("TWIN_MULTITURN_LOSS") == "1":
        # `train` binds its loss default when it is defined, so the call has to
        # carry the replacement.
        original_train = lora.train

        def train(*args, **kwargs):
            kwargs.setdefault("loss", multiturn_loss)
            return original_train(*args, **kwargs)

        lora.train = train
        print("[twin] training on every assistant turn, not only the last")


if __name__ == "__main__":
    _patch()
    from mlx_lm.lora import main

    main()
