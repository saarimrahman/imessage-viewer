"""NVIDIA backend, using transformers and peft. Used on Modal.

CAUTION: unverified. `mlx-community` repos are Apple conversions, so this maps
each model key to the original Qwen repo and quantizes with bitsandbytes.
"""

from twin.export import sanitize_chat_template
from twin.train import model_config

# The `mlx-community` 4-bit conversions do not load on NVIDIA.
TORCH_REPOS = {
    "qwen3-capable": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-compact": "Qwen/Qwen3-0.6B",
}


def load_model(model_key, adapter_path=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = dict(model_config(model_key))
    repo = TORCH_REPOS.get(model_key)
    if not repo:
        raise ValueError(f"No torch repo mapped for {model_key}")
    config["repo"] = repo

    tokenizer = AutoTokenizer.from_pretrained(repo)
    sanitize_chat_template(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return (model.eval(), tokenizer), config
