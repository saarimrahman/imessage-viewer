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
    CONTEXT_TURNS,
    TWIN_DIR,
    coerce_chat,
    parse_person_arg,
    resolve_subject,
    system_for,
)
from twin.retrieve import RETRIEVE_K, few_shot_messages, load_index, retrieve
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
    return load(model, adapter_path=adapter)


def complete(
    model,
    tokenizer,
    messages,
    max_tokens=64,
    chat_template_args=None,
    temp=0.0,
    top_p=0.9,
    seed=0,
):
    from mlx_lm import generate

    prompt = tokenizer.apply_chat_template(
        coerce_chat(messages),
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_args or {}),
    )
    kwargs = {"prompt": prompt, "max_tokens": max_tokens}
    if temp and temp > 0:
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx

        mx.random.seed(seed)
        kwargs["sampler"] = make_sampler(temp, top_p=top_p)
    return generate(model, tokenizer, **kwargs).strip()


def _with_retrieval(messages, text, enabled):
    if not enabled:
        return messages
    shots = few_shot_messages(retrieve(text, load_index(TWIN_DIR), k=RETRIEVE_K, exclude=[text]))
    if not shots:
        return messages
    system = messages[:1] if messages and messages[0]["role"] == "system" else []
    rest = messages[len(system) :]
    budget = max(2, CONTEXT_TURNS - len(shots))
    rest = rest[-budget:]
    return system + shots + rest


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
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--retrieve", action=argparse.BooleanOptionalAction, default=True)
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
    messages = [{"role": "system", "content": system_for(subject["name"])}]

    if args.once is not None:
        turn = _with_retrieval(messages + [{"role": "user", "content": args.once}], args.once, args.retrieve)
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
            prompt = _with_retrieval(messages, text, args.retrieve)
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
            print("twin:", reply.replace("<|bubble|>", " / "))
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
