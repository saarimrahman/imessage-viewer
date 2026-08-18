#!/usr/bin/env python3
"""Chat with one model-specific Twin adapter."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.export import (
    BUBBLE,
    CHAT_TEMP,
    CHAT_TOP_P,
    MAX_BUBBLES,
    REPETITION_PENALTY,
    TWIN_DIR,
    clip_bubbles,
    coerce_chat,
    parse_person_arg,
    resolve_subject,
    strip_scaffold,
    system_for,
)
from twin.retrieve import RETRIEVE_K, load_index, with_retrieved_shots
from twin.train import DEFAULT_MODEL, MODELS, model_config, resolved_adapter_dir


def load_model(model, adapter):
    try:
        from mlx_lm import load
    except ImportError:
        sys.exit(
            "mlx_lm is not installed. Run: "
            "./.venv/bin/python -m pip install -r twin/requirements.txt"
        )
    if not os.path.isfile(os.path.join(adapter, "adapters.safetensors")):
        sys.exit("No adapter exists for this model. Train it on the Twin page.")
    from twin.export import sanitize_chat_template

    model_obj, tokenizer = load(model, adapter_path=adapter)
    sanitize_chat_template(tokenizer)
    return model_obj, tokenizer


def generate_kwargs(
    max_tokens,
    temp=CHAT_TEMP,
    top_p=CHAT_TOP_P,
    seed=0,
    repetition_penalty=None,
):
    """Sampler for Twin decoding. The bubble cap stops delimiter loops."""
    kwargs = {"max_tokens": max_tokens}
    if temp and temp > 0:
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx

        mx.random.seed(seed)
        kwargs["sampler"] = make_sampler(temp, top_p=top_p)
    penalty = REPETITION_PENALTY if repetition_penalty is None else repetition_penalty
    if penalty and abs(penalty - 1.0) > 1e-6:
        from mlx_lm.sample_utils import make_logits_processors

        processors = make_logits_processors(repetition_penalty=penalty)
        if processors:
            kwargs["logits_processors"] = processors
    return kwargs


def complete(
    model,
    tokenizer,
    messages,
    max_tokens=64,
    chat_template_args=None,
    temp=CHAT_TEMP,
    top_p=CHAT_TOP_P,
    seed=0,
):
    from mlx_lm import generate

    prompt = tokenizer.apply_chat_template(
        coerce_chat(messages),
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_args or {}),
    )
    kwargs = generate_kwargs(max_tokens, temp=temp, top_p=top_p, seed=seed)
    text = generate(model, tokenizer, prompt=prompt, **kwargs)
    return clip_bubbles(strip_scaffold(text), MAX_BUBBLES)


def _with_retrieval(messages, text, enabled, peer=None):
    index = load_index(TWIN_DIR) if enabled else []
    return with_retrieved_shots(
        messages,
        index,
        k=RETRIEVE_K if enabled else 0,
        exclude=[text],
        peer=peer,
    )


def main():
    parser = argparse.ArgumentParser(description="Chat with a local Twin adapter")
    parser.add_argument("--model-key", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--model", help="Custom MLX model repository or local path")
    parser.add_argument("--adapter", help="Custom adapter directory")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint id from the Twin page, like qwen3-capable/20260816-160000-ab12cd34ef56-12400/latest",
    )
    parser.add_argument("--once", metavar="TEXT", help="One prompt, then exit")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=CHAT_TEMP)
    parser.add_argument("--top-p", type=float, default=CHAT_TOP_P)
    parser.add_argument("--retrieve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--to",
        default="",
        help="Contact name you are texting, so the twin can switch register",
    )
    parser.add_argument(
        "--person",
        default="me",
        help="me, or a phone number / email / person id from the Twin page",
    )
    args = parser.parse_args()

    try:
        person_id = parse_person_arg(args.person)
        subject = resolve_subject(person_id)
    except ValueError as e:
        sys.exit(str(e))

    config = model_config(args.model_key)
    adapter = args.adapter
    if not adapter and args.checkpoint:
        from twin.train import load_dir_for, resolve_checkpoint

        try:
            ckpt = resolve_checkpoint(args.checkpoint, person_id)
        except ValueError as e:
            sys.exit(str(e))
        adapter = load_dir_for(ckpt["path"], ckpt["step"])
        config = model_config(ckpt["model"])
    model, tokenizer = load_model(
        args.model or config["repo"],
        adapter or resolved_adapter_dir(args.model_key, person_id),
    )
    messages = [{"role": "system", "content": system_for(subject["name"], peer=args.to or None)}]

    if args.once is not None:
        turn = _with_retrieval(
            messages + [{"role": "user", "content": args.once}],
            args.once,
            args.retrieve,
            peer=args.to or None,
        )
        print(
            complete(
                model,
                tokenizer,
                turn,
                args.max_tokens,
                config.get("chat_template_args"),
                temp=args.temp,
                top_p=args.top_p,
            )
        )
        return

    print(f"Twin ready ({subject['name']}). Ctrl-C to quit.")
    try:
        while True:
            text = input("you: ").strip()
            if not text:
                continue
            messages.append({"role": "user", "content": text})
            prompt = _with_retrieval(messages, text, args.retrieve, peer=args.to or None)
            reply = complete(
                model,
                tokenizer,
                prompt,
                args.max_tokens,
                config.get("chat_template_args"),
                temp=args.temp,
                top_p=args.top_p,
            )
            messages.append({"role": "assistant", "content": reply})
            print("twin:", reply.replace(BUBBLE, " / ").replace("<|bubble|>", " / "))
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
