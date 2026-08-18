"""Where a model is loaded and text is generated. Nothing else lives here.

The boundary is deliberately narrow. Two parallel trees, one for MLX and one for
Modal, would end with two copies of the metrics, and a metric that exists twice
drifts. Three separate conclusions in `EXPERIMENTS.md` had to be corrected
because two versions of `style_distance` coexisted inside a single codebase.

So the split is not Apple against NVIDIA. It is "produces predictions" against
everything else:

- `export.py`, `style.py`, `retrieve.py` never touch a backend.
- `bench.py`, `multiturn.py`, `metriceval.py` take one.
- Scoring always runs where the results log is, on the stored predictions, so
  every run in `results.jsonl` shares one metric version whatever hardware
  generated it.

A remote backend therefore only has to return text. It never scores.

CAUTION: an MLX adapter and a PEFT adapter are different formats and cannot be
loaded by each other. Two backends compare recipes on a shared holdout, never
the same weights.
"""

import os

BACKENDS = ("mlx", "torch")


def default_backend():
    """MLX on Apple silicon, torch anywhere else. `TWIN_BACKEND` overrides."""
    chosen = (os.environ.get("TWIN_BACKEND") or "").strip().lower()
    if chosen in BACKENDS:
        return chosen
    try:
        import mlx.core  # noqa: F401

        return "mlx"
    except ImportError:
        return "torch"


def load(model_key, adapter_path=None, backend=None):
    """Return ``((model, tokenizer), config)`` for the chosen backend."""
    name = backend or default_backend()
    if name == "mlx":
        from twin.backends.mlx import load_model
    elif name == "torch":
        from twin.backends.torch import load_model
    else:
        raise ValueError(f"Unknown backend: {name}. Pick one of {BACKENDS}.")
    return load_model(model_key, adapter_path)
