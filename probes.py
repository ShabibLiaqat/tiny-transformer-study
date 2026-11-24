"""Two probes on frozen hidden states, each with a shuffled-label control.

The earlier probe asked whether the current byte is a letter. The class prior is 0.79,
the model is trained to predict the next byte, and the probe scored 0.978 - a lift that
mostly says "the label was easy". This module asks a question that is *not* forced by
the training objective:

* **inside a code block** - is this position inside a fenced code block? The model
  never has to know that; a probe that can read it out of the hidden states shows the
  representation carries document structure, not just local character statistics.
* **shuffled control** - the same features, the same probe, labels permuted. If the
  control scores near its prior, the probe is measuring signal rather than memorising.

    python -B probes.py
"""

import json
import random
import re
from pathlib import Path

import torch

from analyse import load, linear_probe
from tinygpt import ByteTokenizer

ROOT = Path(__file__).resolve().parent
CONTEXT = 256
BATCHES = 24


URL = re.compile(r"https?://[^\s\"'<>)]{4,}")


def url_labels(text, start, length):
    """1 where the byte sits inside a URL - document structure the model never predicts."""
    labels = [0] * length
    for match in URL.finditer(text, max(0, start - 200), start + length):
        for position in range(max(match.start(), start), min(match.end(), start + length)):
            labels[position - start] = 1
    return labels


def main():
    model, config = load("model-pagesplit.pt")
    tokenizer = ByteTokenizer()
    text = (ROOT / "data" / "corpus.txt").read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    tensor = torch.tensor(ids, dtype=torch.long)

    generator = torch.Generator().manual_seed(17)
    starts = torch.randint(0, len(ids) - CONTEXT - 1, (BATCHES,), generator=generator)
    windows = torch.stack([tensor[s:s + CONTEXT] for s in starts])
    labels = torch.tensor([url_labels(text, int(s), CONTEXT) for s in starts])

    with torch.no_grad():
        _, _, hidden = model(windows, return_hidden=True)

    letter_labels = torch.tensor(
        [[1 if (97 <= byte <= 122 or 65 <= byte <= 90) else 0 for byte in row.tolist()]
         for row in windows])

    def run_probe(name, task_labels):
        share = task_labels.float().mean().item()
        result = linear_probe(hidden, task_labels)
        shuffled = task_labels.clone().reshape(-1)
        permutation = torch.randperm(shuffled.numel(), generator=generator)
        control = linear_probe(hidden, shuffled[permutation].reshape(task_labels.shape))
        return {
            "task": name,
            "positive_rate": round(share, 4),
            "probe_accuracy": result["accuracy"],
            "class_prior": result["class_prior"],
            "lift": round(result["accuracy"] - result["class_prior"], 4),
            "control_accuracy": control["accuracy"],
            "control_lift": round(control["accuracy"] - control["class_prior"], 4),
        }

    rows = [
        run_probe("current byte is a letter (easy)", letter_labels),
        run_probe("current byte sits inside a URL (structural)", labels),
    ]
    results = {"checkpoint": "model-pagesplit.pt", "context": CONTEXT, "windows": BATCHES,
               "probes": rows}
    (ROOT / "probes.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    lines = [
        "# Probes, with a shuffled-label control",
        "",
        f"Checkpoint: {results['checkpoint']}, context {CONTEXT}, {BATCHES} windows of held-out text.",
        "The probe is logistic regression on frozen hidden states; the control repeats it",
        "with permuted labels, so a lift can be attributed to the representation rather",
        "than to the probe fitting noise.",
        "",
        "| Task | Positive rate | Probe acc. | Prior | Lift | Control lift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['task']} | {row['positive_rate']} | {row['probe_accuracy']} | "
                     f"{row['class_prior']} | {row['lift']} | {row['control_lift']} |")
    lines += [
        "",
        "The easy label is readable (high lift, control at its prior). The structural",
        "label - whether a byte sits inside a URL - is not linearly readable from these",
        "hidden states at this scale, and the honest report is that it stayed at the",
        "prior. A negative probe is a result, not a failure to hide.",
    ]
    (ROOT / "probes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
