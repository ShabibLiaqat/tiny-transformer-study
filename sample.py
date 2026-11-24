"""Generate text from the trained tiny model, to show what it actually learned.

A perplexity number says the model predicts bytes better than a bigram. Samples say
whether that is a real ability to model structure or a lucky average. Both are shown
here, unedited, at two temperatures.

    python -B sample.py
"""

import json
from pathlib import Path

import torch

from tinygpt import ByteTokenizer, TinyGPT

ROOT = Path(__file__).resolve().parent

PROMPTS = [
    "PostHog is",
    "def ",
    "The best way to track",
    "Session replay",
]


def main():
    tokenizer = ByteTokenizer()
    checkpoint = torch.load(ROOT / "model.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = TinyGPT(vocab=256, d_model=config["d_model"], heads=config["heads"],
                    layers=config["layers"], context=config["context"],
                    swiglu=config["swiglu"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    torch.manual_seed(5)
    rows = []
    for prompt in PROMPTS:
        for temperature in (0.7, 0.9):
            ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            with torch.no_grad():
                for _ in range(120):
                    window = ids[:, -config["context"]:]
                    logits, _ = model(window)
                    logits = logits[:, -1, :] / temperature
                    probabilities = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probabilities, 1)
                    ids = torch.cat([ids, next_id], dim=1)
            text = tokenizer.decode(ids[0].tolist())
            rows.append({"prompt": prompt, "temperature": temperature, "text": text})

    (ROOT / "samples.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    lines = ["# Samples from the trained model", "",
             "Byte-level generation on a 836k-parameter model trained on 348 KB of public",
             "PostHog pages. Unedited output at two temperatures.", ""]
    for row in rows:
        lines += [f"## {row['prompt']!r} at temperature {row['temperature']}", "", "```text",
                  row["text"].strip(), "```", ""]
    (ROOT / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(rows)} samples written to samples.md")
    for row in rows[:2]:
        print(f"\n--- {row['prompt']!r} @ {row['temperature']}\n{row['text'].strip()[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
