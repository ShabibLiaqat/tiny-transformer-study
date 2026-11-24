"""Train the tiny transformer on PostHog's public text and measure it honestly.

One CPU run, a fixed seed, and two baselines that are hard to beat by accident: a
unigram model and a bigram model. If a 4-layer transformer cannot beat a bigram on
perplexity, the training run is broken, and that is worth knowing before quoting a
number in an application.

    python -B train.py --steps 1200 --minutes 8
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tinygpt import ByteTokenizer, TinyGPT

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def batches(ids, context, batch, generator=None):
    span = context + 1
    usable = (len(ids) - span)
    while True:
        starts = torch.randint(0, usable, (batch,), generator=generator)
        x = torch.stack([ids[s:s + context] for s in starts])
        y = torch.stack([ids[s + 1:s + context + 1] for s in starts])
        yield x, y


def loss_of(model, ids, context, batch, rounds=8):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for step, (x, y) in enumerate(batches(ids, context, batch)):
            _, loss = model(x, y)
            total += float(loss)
            if step + 1 >= rounds:
                break
    return total / rounds


def ngram_baselines(train, test, order=2):
    """Unigram and bigram cross-entropy on the same held-out bytes."""
    train_bytes = train.tolist()
    counts = {}
    totals = {}
    for value in set(train_bytes):
        counts[(value,)] = train_bytes.count(value)
    totals[0] = len(train_bytes)
    unigram = -sum(math.log(max(counts.get((b,), 0), 1e-9) / totals[0]) for b in test.tolist()) / len(test)

    pairs = {}
    previous = {}
    for left, right in zip(train_bytes, train_bytes[1:]):
        pairs[(left, right)] = pairs.get((left, right), 0) + 1
        previous[left] = previous.get(left, 0) + 1
    test_bytes = test.tolist()
    total = 0.0
    for left, right in zip(test_bytes, test_bytes[1:]):
        p = pairs.get((left, right), 0) / max(previous.get(left, 0), 1)
        total -= math.log(max(p, 1e-9))
    bigram = total / max(len(test_bytes) - 1, 1)
    return {"unigram_bits_per_byte": unigram / math.log(2),
            "bigram_bits_per_byte": bigram / math.log(2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--context", type=int, default=96)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--swiglu", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="", help="suffix for the saved results and weights")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = ByteTokenizer()
    corpus = (DATA / "corpus.txt").read_text(encoding="utf-8")
    ids = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    split = int(len(ids) * 0.9)
    train, test = ids[:split], ids[split:]
    print(f"corpus {len(ids):,} bytes | train {len(train):,} | test {len(test):,}")

    baselines = ngram_baselines(train, test)
    print(f"baselines: unigram {baselines['unigram_bits_per_byte']:.2f} bpb, "
          f"bigram {baselines['bigram_bits_per_byte']:.2f} bpb")

    model = TinyGPT(vocab=256, d_model=args.d_model, heads=args.heads, layers=args.layers,
                    context=args.context, dropout=0.05, swiglu=args.swiglu)
    print(f"model: {model.parameter_count:,} parameters "
          f"({args.layers} layers, {args.heads} heads, d_model {args.d_model}, "
          f"{'SwiGLU' if args.swiglu else 'GELU'})")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01)
    warmup = max(20, args.steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min(1.0, (step + 1) / warmup) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))))

    history = []
    started = time.time()
    step = 0
    running = None
    for x, y in batches(train, args.context, args.batch):
        _, loss = model(x, y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        step += 1
        value = float(loss.detach())
        running = value if running is None else 0.98 * running + 0.02 * value
        if step % 100 == 0 or step == 1:
            elapsed = time.time() - started
            print(f"step {step:>5} | loss {running:.3f} | {elapsed:.0f}s")
            history.append({"step": step, "loss": round(running, 4)})
        if step >= args.steps:
            break

    train_loss = loss_of(model, train, args.context, args.batch)
    test_loss = loss_of(model, test, args.context, args.batch)
    results = {
        "corpus_bytes": int(len(ids)),
        "train_bytes": int(len(train)),
        "test_bytes": int(len(test)),
        "parameters": model.parameter_count,
        "config": {"d_model": args.d_model, "heads": args.heads, "layers": args.layers,
                   "context": args.context, "batch": args.batch, "steps": args.steps,
                   "swiglu": args.swiglu, "seed": args.seed, "lr": args.lr},
        "baselines": baselines,
        "train_bits_per_byte": train_loss / math.log(2),
        "test_bits_per_byte": test_loss / math.log(2),
        "train_perplexity": math.exp(train_loss),
        "test_perplexity": math.exp(test_loss),
        "history": history,
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    suffix = f"-{args.tag}" if args.tag else ""
    results["tag"] = args.tag or "base"
    (ROOT / f"results{suffix}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": results["config"]}, ROOT / f"model{suffix}.pt")
    print(json.dumps({k: results[k] for k in ("test_bits_per_byte", "test_perplexity",
                                             "wall_clock_seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
