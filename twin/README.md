# Twin

Twin is a private local model that learns how a person writes iMessage texts. It uses MLX LM on Apple silicon.

Your messages and adapters stay on this Mac. The app stores the generated dataset and adapters in `.cache/twin/`. Each contact adapter lives under `.cache/twin/people/`.

## Data coverage

The Complete run uses every non-empty text from the chosen person. By default that person is you. You can train as any contact. Each person keeps a separate adapter.

The exporter groups nearby consecutive bubbles into one turn. A gap of more than 30 minutes starts a new turn.

Each target has up to six real conversation turns for context. The exporter also adds shorter versions of long conversation pairs.

Media-only messages do not contain text for the model. The Audit section counts these messages separately.

## Models

| Group | Models |
| --- | --- |
| Featured ≤4B | Qwen 3 4B Instruct 2507, G9v3 3B, MiniCPM5 1B, Nanbeige 4.1 3B, Nemotron 3 Nano 4B, Ministral 3 3B, Phi-4 Mini, Granite 4.1 3B, Gemma 3 4B |
| Maximum on 16 GB | Qwen 3 8B, Llama 3.1 8B, Granite 4.1 8B |
| Efficient | Qwen 3 0.6B and 1.7B, Gemma 3 1B, SmolLM3 3B |
| Existing adapters | Qwen 2.5 0.5B, 1.5B, and 3B |

The selector is curated from the [Artificial Analysis Tiny](https://artificialanalysis.ai/models/open-source/tiny) and [Small](https://artificialanalysis.ai/models/open-source/small) lists, then filtered for text chat, a compatible MLX LM architecture, a system-message chat template, 4-bit weights, and a maximum of 8B total parameters.

Each choice has a separate adapter. MLX LM downloads the selected weights on the first training or chat load, not when the page opens or the selection changes. The 7B–8B choices use batch size 1 and adapt four layers to stay near the practical limit of a 16 GB Mac. Close memory-heavy apps before using them.

Model keys remain tied to their original bases so an older adapter is never loaded onto incompatible weights. The broader MLX tag also contains image, audio, embedding, base, and inference-only models that Twin cannot train.

## Install

Create the app virtual environment first. Then install the MLX training dependencies.

```bash
./.venv/bin/python -m pip install -r twin/requirements.txt
./.venv/bin/python app.py
```

Open <http://127.0.0.1:8765/twin>.

Quick uses 160 recent examples and 30 steps. Use this run to make sure that local training works.

Complete builds the full dataset. Then it makes one pass through all generated examples unless you set a step count.

Each train writes a new adapter folder named with the start time, a hash of `train.jsonl`, and the step count. Earlier adapters stay on disk. Long runs save checkpoints about every 5% of the step count. Stop mid-run and you can chat with the last save, or continue from it.

The chat dropdown lists those runs and their checkpoints. It defaults to the latest save of the newest run. The Model tab can start from fresh weights or from a saved checkpoint.

The charts show training loss, reference loss, throughput, and peak memory. The reference sample also remains in the training file.

## Command line

Export the complete dataset. Then train the recommended Qwen 3 4B model for one complete pass.

```bash
./.venv/bin/python twin/export.py
./.venv/bin/python twin/train.py --model-key qwen3-capable --complete
./.venv/bin/python twin/train.py --model-key qwen3-capable --complete --iters 2000
./.venv/bin/python twin/train.py --model-key qwen3-capable --iters 500 --resume qwen3-capable/20260816-160000-ab12cd34ef56-2000/latest
```

To train as a contact, pass a phone number or email:

```bash
./.venv/bin/python twin/export.py --person +15555550100
./.venv/bin/python twin/train.py --model-key qwen3-capable --complete --person +15555550100
```

Start an interactive chat with the Qwen 3 4B adapter.

```bash
./.venv/bin/python twin/chat.py --model-key qwen3-capable
./.venv/bin/python twin/chat.py --model-key qwen3-capable --person +15555550100
./.venv/bin/python twin/chat.py --checkpoint qwen3-capable/20260816-160000-ab12cd34ef56-2000/latest
```

Use `--once` for one reply.

```bash
./.venv/bin/python twin/chat.py --model-key qwen3-capable --once "you free tonight?"
```
