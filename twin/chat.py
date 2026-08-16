#!/usr/bin/env python3
"""Chat with one model-specific Twin adapter."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.export import SYSTEM
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


def complete(model, tokenizer, messages, max_tokens=64, chat_template_args=None):
    from mlx_lm import generate

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_args or {}),
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens).strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with a local Twin adapter")
    parser.add_argument("--model-key", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--model", help="Custom MLX model repository or local path")
    parser.add_argument("--adapter", help="Custom adapter directory")
    parser.add_argument("--once", metavar="TEXT", help="One prompt, then exit")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    config = model_config(args.model_key)
    model, tokenizer = load_model(
        args.model or config["repo"],
        args.adapter or resolved_adapter_dir(args.model_key),
    )
    messages = [{"role": "system", "content": SYSTEM}]

    if args.once is not None:
        messages.append({"role": "user", "content": args.once})
        print(
            complete(
                model,
                tokenizer,
                messages,
                args.max_tokens,
                config.get("chat_template_args"),
            )
        )
        return

    print("Twin ready. Ctrl-C to quit.")
    try:
        while True:
            text = input("you: ").strip()
            if not text:
                continue
            messages.append({"role": "user", "content": text})
            reply = complete(
                model,
                tokenizer,
                messages,
                args.max_tokens,
                config.get("chat_template_args"),
            )
            messages.append({"role": "assistant", "content": reply})
            print("twin:", reply)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
