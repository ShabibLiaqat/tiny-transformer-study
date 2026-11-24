"""Next-event prediction on a real public interaction-sequence dataset.

What it does
------------
Product analytics is mostly a sequence problem, not a text problem: it is event sequences - one user, a series of things they did, in order, and the
question of what happens next. This module runs the same loop on sequences of real
interactions with the baselines that make the result mean something:

* **popularity** - recommend the globally most frequent items, no personalisation,
* **Markov** - first-order item-to-item transitions counted on training data,
* **transformer** - a small causal model over the item sequence.

Dataset: MovieLens `ml-latest-small` from GroupLens - public, timed, per-user
interactions, licensed for research use. It is not PostHog data and not product
clickstream; the loop, the splits and the baselines are what transfer, and the README
says so once.

Three splits, because they answer different questions:

* **valid**  - the last 10% of each training user's in-window sequence,
* **future** - every event after a timestamp cutoff, for known users,
* **cold**   - every event from users the model never saw.

    python -B event_sequences.py prepare     # download, split, print baselines
"""

import json
import math
import random
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sequences"
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
LICENCE = "https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html"
MIN_SEQUENCE = 5
COLD_SHARE = 5
FUTURE_CUTOFF = 0.8
VALID_SHARE = 0.1
CONTEXT = 32


# ---------------------------------------------------------------- data preparation

def download():
    DATA.mkdir(parents=True, exist_ok=True)
    csv = DATA / "ratings.csv"
    if csv.is_file():
        return csv
    archive = DATA / "ml-latest-small.zip"
    if not archive.is_file():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(DATA)
    next(DATA.glob("ml-latest-small/ratings.csv")).replace(csv)
    return csv


def read_sequences():
    """user -> [(timestamp, item)] ordered, de-duplicated, long enough to model."""
    events = defaultdict(list)
    with download().open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip().split(",")
            if len(parts) < 4:
                continue
            user, item, _, timestamp = parts[:4]
            events[int(user)].append((int(timestamp), int(item)))
    sequences = {}
    for user, rows in events.items():
        rows.sort()
        seen, deduped = set(), []
        for timestamp, item in rows:
            if item in seen:
                continue
            seen.add(item)
            deduped.append((timestamp, item))
        if len(deduped) >= MIN_SEQUENCE:
            sequences[user] = deduped
    return sequences


def prepare():
    sequences = read_sequences()
    users = sorted(sequences)
    cold_users = set(users[-max(1, len(users) // COLD_SHARE):])
    known_users = [user for user in users if user not in cold_users]

    times = sorted(timestamp for user in known_users for timestamp, _ in sequences[user])
    cutoff = times[int(len(times) * FUTURE_CUTOFF)]

    counts = Counter()
    for user in known_users:
        counts.update(item for timestamp, item in sequences[user] if timestamp <= cutoff)
    ordered = [item for item, _ in counts.most_common()]
    index = {item: position + 1 for position, item in enumerate(ordered)}  # 0 = unknown

    fit, valid, future = [], [], []
    for user in known_users:
        history = [index.get(item, 0) for timestamp, item in sequences[user]
                   if timestamp <= cutoff]
        later = [index.get(item, 0) for timestamp, item in sequences[user]
                 if timestamp > cutoff]
        if len(history) >= 3:
            split = max(2, int(len(history) * (1 - VALID_SHARE)))
            fit.append(history[:split])
            valid.append(history[split - 1:])
        if len(later) >= 2:
            future.append(later)
    cold = [[index.get(item, 0) for _, item in sequences[user]] for user in cold_users]

    prepared = {
        "vocab_size": len(ordered) + 1,
        "fit": fit,
        "valid": [seq for seq in valid if len(seq) >= 2],
        "future": future,
        "cold": [seq for seq in cold if len(seq) >= 2],
        "cutoff": cutoff,
        "counts": {
            "users": len(users),
            "known_users": len(known_users),
            "cold_users": len(cold_users),
            "fit_sequences": len(fit),
            "future_sequences": len(future),
            "cold_sequences": len([seq for seq in cold if len(seq) >= 2]),
            "items_in_vocab": len(ordered),
        },
        "source": {"url": URL, "licence": LICENCE,
                   "note": "MovieLens ml-latest-small (GroupLens), public research dataset; "
                           "not PostHog data and not product clickstream"},
    }
    (DATA / "prepared.json").write_text(json.dumps(prepared), encoding="utf-8")
    print(json.dumps(prepared["counts"], indent=1))
    return prepared


def load():
    path = DATA / "prepared.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else prepare()


# ---------------------------------------------------------------- baselines

def popularity(prepared):
    counts = Counter()
    for ids in prepared["fit"]:
        counts.update(ids)
    return [item for item, _ in counts.most_common()]


def markov(prepared):
    transitions = defaultdict(Counter)
    for ids in prepared["fit"]:
        for left, right in zip(ids, ids[1:]):
            transitions[left][right] += 1
    return transitions


def topk_accuracy(rankings, k_values=(1, 5, 10)):
    """rankings: list of (ranked_predictions, target). Returns accuracy per k."""
    hits = {k: 0 for k in k_values}
    for ranked, target in rankings:
        for k in k_values:
            hits[k] += 1 if target in ranked[:k] else 0
    total = len(rankings)
    return {f"top{k}": round(hits[k] / max(total, 1), 4) for k in k_values}, total


def baseline_results(prepared, k_values=(1, 5, 10), context=1):
    popular = popularity(prepared)
    transitions = markov(prepared)
    top_k = max(k_values)
    results = {}
    for split in ("valid", "future", "cold"):
        popularity_pairs, markov_pairs = [], []
        for ids in prepared[split]:
            for position in range(1, len(ids)):
                target = ids[position]
                popularity_pairs.append((popular[:top_k], target))
                distribution = transitions.get(ids[position - context])
                ranked = ([item for item, _ in distribution.most_common(top_k)]
                          if distribution else popular[:top_k])
                markov_pairs.append((ranked, target))
        popularity_scores, total = topk_accuracy(popularity_pairs, k_values)
        markov_scores, _ = topk_accuracy(markov_pairs, k_values)
        results[split] = {"predictions": total, "popularity": popularity_scores,
                          "markov": markov_scores}
    return results


# ---------------------------------------------------------------- model

def build_model(vocab_size, d_model=128, layers=2, heads=4, context=CONTEXT, prior=None):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                    nn.Linear(4 * d_model, d_model))

        def forward(self, x, mask):
            hidden = self.ln1(x)
            attended, _ = self.attn(hidden, hidden, hidden, attn_mask=mask,
                                    need_weights=False)
            return x + attended + self.ff(self.ln2(x + attended))

    class SequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos = nn.Embedding(context, d_model)
            self.blocks = nn.ModuleList(Block() for _ in range(layers))
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=True)
            self.head.weight = self.tok.weight
            if prior is not None:
                # Start from the popularity prior: the model then only has to learn the
                # personal and sequential part, which is what the baselines already fix.
                self.head.bias.data.copy_(prior)

        def logits(self, idx):
            positions = torch.arange(idx.size(1), device=idx.device)
            x = self.tok(idx) + self.pos(positions)
            mask = torch.triu(torch.full((idx.size(1), idx.size(1)), float("-inf"),
                                         device=idx.device), diagonal=1)
            for block in self.blocks:
                x = block(x, mask)
            return self.head(self.ln(x))

        def forward(self, idx, targets):
            import torch.nn.functional as F
            return F.cross_entropy(self.logits(idx).reshape(-1, vocab_size),
                                   targets.reshape(-1), ignore_index=-100)

    model = SequenceModel()
    # Default nn.Embedding init is N(0, 1); tied with the output head that produces
    # logits of magnitude ~1e2 at step 1, and cross-entropy starts near 88 instead of
    # the ~9.2 chance value. Scale the big matrices down and zero the biases.
    for name, parameter in model.named_parameters():
        if parameter.dim() > 1:
            torch.nn.init.normal_(parameter, std=0.02)
        else:
            torch.nn.init.zeros_(parameter)
    return model


def train_model(prepared, seed=0, epochs=8, batch=64, lr=1e-3, device="cpu"):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    counts = [1.0] * prepared["vocab_size"]
    for seq in prepared["fit"]:
        for item in seq:
            if 0 <= item < len(counts):
                counts[item] += 1.0
    total = sum(counts)
    prior = torch.tensor([math.log(count / total) for count in counts], dtype=torch.float32)
    model = build_model(prepared["vocab_size"], prior=prior).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = 50
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min(1.0, (step + 1) / warmup))
    rows = [seq for seq in prepared["fit"] if len(seq) >= 2]
    rng = random.Random(seed)

    def make_batch():
        inputs, targets = [], []
        for _ in range(batch):
            ids = rows[rng.randrange(len(rows))]
            start = rng.randrange(max(1, len(ids) - CONTEXT))
            chunk = ids[start:start + CONTEXT + 1]
            inputs.append(chunk[:-1] + [0] * (CONTEXT - len(chunk[:-1])))
            targets.append(chunk[1:] + [-100] * (CONTEXT - len(chunk[1:])))
        return (torch.tensor(inputs, device=device),
                torch.tensor(targets, device=device))

    model.train()
    first_steps = 0
    steps = max(1, len(rows) // batch)
    for epoch in range(epochs):
        total = 0.0
        for _ in range(steps):
            x, y = make_batch()
            loss = model(x, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
            if first_steps < 3:
                print(f"  seed {seed} step {first_steps + 1} loss {float(loss.detach()):.3f}",
                      flush=True)
                first_steps += 1
        print(f"  seed {seed} epoch {epoch + 1}/{epochs} loss {total / steps:.4f}", flush=True)
    return model


def evaluate_model(model, prepared, split, k_values=(1, 5, 10), device="cpu"):
    import torch
    rankings = []
    model.eval()
    with torch.no_grad():
        for ids in prepared[split]:
            for position in range(1, len(ids)):
                window = ids[max(0, position - CONTEXT):position]
                padded = [0] * (CONTEXT - len(window)) + window
                tensor = torch.tensor([padded], device=device)
                ranked = torch.topk(model.logits(tensor)[0, -1], max(k_values)).indices.tolist()
                rankings.append((ranked, ids[position]))
    return topk_accuracy(rankings, k_values)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        data = prepare()
        print(json.dumps(baseline_results(data), indent=1))
    else:
        print(__doc__)
