"""Train the sequence model on a GPU for three seeds and compare with the baselines.

The local machine can prepare the data but a CPU training loop over a 9,700-item
vocabulary is slow enough to discourage iteration, so the runs happen on Modal and the
results come back as JSON.

    python -m modal run modal_sequences.py
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
APP = modal.App("posthog-sequence-model")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_file(str(ROOT / "event_sequences.py"), "/root/event_sequences.py")
)


@APP.function(image=image, gpu="A10G", timeout=3600)
def run_seed(prepared: dict, seed: int, epochs: int = 20) -> dict:
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    import event_sequences as sequences

    started = time.time()
    model = sequences.train_model(prepared, seed=seed, epochs=epochs, batch=128,
                                  lr=3e-3, device="cuda")
    result = {"seed": seed, "seconds": round(time.time() - started, 1)}
    for split in ("valid", "future", "cold"):
        scores, total = sequences.evaluate_model(model, prepared, split, device="cuda")
        result[split] = {"values": scores, "predictions": total}
    return result


@APP.local_entrypoint()
def main(seeds: str = "0,1,2"):
    prepared = json.loads((ROOT / "data" / "sequences" / "prepared.json").read_text(encoding="utf-8"))
    baselines = json.loads((ROOT / "results-sequences-baselines.json").read_text(encoding="utf-8")) \
        if (ROOT / "results-sequences-baselines.json").is_file() else {}

    runs = []
    for seed in [int(value) for value in seeds.split(",") if value.strip()]:
        print(f"--- seed {seed}")
        result = run_seed.remote(prepared=prepared, seed=seed)
        print({split: result[split]["values"] for split in ("valid", "future", "cold")})
        runs.append(result)

    summary = {"prepared_counts": prepared["counts"], "source": prepared["source"],
               "baselines": baselines, "runs": runs}
    for split in ("valid", "future", "cold"):
        for k in ("top1", "top5", "top10"):
            values = [run[split]["values"][k] for run in runs]
            summary.setdefault("summary", {}).setdefault(split, {})[k] = {
                "mean": round(sum(values) / len(values), 4),
                "values": values,
            }
    (ROOT / "results-sequences.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary.get("summary", {}), indent=1))
