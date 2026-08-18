"""Apple silicon backend, using mlx_lm."""

from twin.export import sanitize_chat_template
from twin.train import model_config


def load_model(model_key, adapter_path=None):
    from mlx_lm import load

    config = model_config(model_key)
    kwargs = {"adapter_path": adapter_path} if adapter_path else {}
    model, tokenizer = load(config["repo"], **kwargs)
    # Without this the training target opens with an empty think block and the
    # adapter answers with `<tool_call>`. The fault is not Apple-specific, so
    # the torch backend applies it too.
    sanitize_chat_template(tokenizer)
    return (model, tokenizer), config
