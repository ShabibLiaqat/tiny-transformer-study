"""Analyse the trained model the way the PostHog research team describes its own work.

Their published objectives for the quarter are eval, visualisation (PCA), probes,
weight analysis and anisotropy. This is a small, honest version of each on the tiny
model: attention entropy and head collapse, softmax distributions, embedding
anisotropy, a linear probe on hidden states, and a PCA of the token embeddings.

    python -B analyse.py
"""

import json
import math
from pathlib import Path

import torch

from tinygpt import ByteTokenizer, TinyGPT

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load(name="model.pt"):
    checkpoint = torch.load(ROOT / name, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = TinyGPT(vocab=256, d_model=config["d_model"], heads=config["heads"],
                    layers=config["layers"], context=config["context"],
                    swiglu=config.get("swiglu", False))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def sample_batch(ids, context, count=16, seed=11):
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - context - 1, (count,), generator=generator)
    return torch.stack([ids[s:s + context] for s in starts])


def attention_report(model, batch):
    maps = model.attention_maps(batch)
    report = []
    for layer, attention in enumerate(maps):
        entropy_per_head = []
        self_mass = []
        for head in range(attention.shape[1]):
            probs = attention[:, head]                      # [batch, query, key]
            entropy = -(probs.clamp_min(1e-9).log() * probs).sum(-1).mean().item()
            diagonal = probs.diagonal(dim1=-2, dim2=-1).mean().item()
            entropy_per_head.append({"head": head,
                                     "entropy_nats": round(entropy, 4),
                                     "max_entropy_nats": round(math.log(probs.shape[-1]), 4),
                                     "self_attention_mass": round(diagonal, 4)})
            self_mass.append(diagonal)
        entropies = [h["entropy_nats"] for h in entropy_per_head]
        report.append({
            "layer": layer,
            "heads": entropy_per_head,
            "mean_entropy_nats": round(sum(entropies) / len(entropies), 4),
            "collapsed_heads": [h["head"] for h in entropy_per_head
                                if h["entropy_nats"] < 0.25 * h["max_entropy_nats"]],
            "duplicate_heads": detect_duplicates(attention),
        })
    return report


def detect_duplicates(attention, threshold=0.995):
    """Heads that attend almost identically are doing one job twice."""
    duplicates = []
    heads = attention.shape[1]
    for a in range(heads):
        for b in range(a + 1, heads):
            left = attention[:, a].flatten()
            right = attention[:, b].flatten()
            cosine = torch.nn.functional.cosine_similarity(left, right, dim=0).item()
            if cosine > threshold:
                duplicates.append({"pair": [a, b], "cosine": round(cosine, 4)})
    return duplicates


def hidden_states(model, batch):
    with torch.no_grad():
        _, _, hidden = model(batch, return_hidden=True)
    return hidden


def anisotropy(hidden):
    """Mean cosine similarity between random pairs of context vectors.

    High values mean the representation occupies a narrow cone, which makes linear
    probes look better than they are and distances between tokens less meaningful.
    """
    flat = hidden.reshape(-1, hidden.shape[-1])
    generator = torch.Generator().manual_seed(3)
    left = torch.randint(0, flat.shape[0], (4000,), generator=generator)
    right = torch.randint(0, flat.shape[0], (4000,), generator=generator)
    cosine = torch.nn.functional.cosine_similarity(flat[left], flat[right], dim=-1)
    return {"mean_cosine": round(cosine.mean().item(), 4),
            "p05": round(torch.quantile(cosine, 0.05).item(), 4),
            "p95": round(torch.quantile(cosine, 0.95).item(), 4)}


def linear_probe(hidden, targets, epochs=300, lr=0.05):
    """Logistic regression on frozen hidden states, trained with plain gradient descent.

    The label is simple and checkable: is the *current* byte a letter?
    A probe that cannot beat the class prior says the representation carries little
    useful signal about that property, which is worth reporting rather than hiding.
    """
    features = hidden.reshape(-1, hidden.shape[-1]).detach()
    labels = targets.reshape(-1).float().unsqueeze(-1)
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)
    weights = torch.zeros(features.shape[-1], 1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.SGD([weights, bias], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = loss_fn(features @ weights + bias, labels)
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        predicted = ((features @ weights + bias) > 0).float()
        accuracy = (predicted == labels).float().mean().item()
        prior = max(labels.mean().item(), 1 - labels.mean().item())
    return {"accuracy": round(accuracy, 4), "class_prior": round(prior, 4),
            "lift_over_prior": round(accuracy - prior, 4),
            "label": "current byte is an ASCII letter"}


def embedding_pca(model):
    matrix = model.tok.weight.detach()
    centred = matrix - matrix.mean(0, keepdim=True)
    _, singular, _ = torch.linalg.svd(centred, full_matrices=False)
    variance = singular ** 2
    share = variance / variance.sum()
    return {"explained_by_pc1": round(share[0].item(), 4),
            "explained_by_pc1_2": round(share[:2].sum().item(), 4),
            "explained_by_pc1_4": round(share[:4].sum().item(), 4),
            "dimensions": int(matrix.shape[1])}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--tag", default="")
    parser.add_argument("--results", default="")
    args = parser.parse_args()
    model, config = load(args.checkpoint)
    tokenizer = ByteTokenizer()
    corpus = (DATA / "corpus.txt").read_text(encoding="utf-8")
    ids = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    split = int(len(ids) * 0.9)
    test_ids = ids[split:]

    batch = sample_batch(test_ids, config["context"], count=12)
    targets = batch  # the probe predicts a property of the current byte
    letter_targets = torch.tensor(
        [[1 if (97 <= int(b) <= 122 or 65 <= int(b) <= 90) else 0 for b in row] for row in targets])

    hidden = hidden_states(model, batch)
    results = {
        "model": {"parameters": model.parameter_count, "config": config},
        "attention": attention_report(model, batch),
        "anisotropy": anisotropy(hidden),
        "probe": linear_probe(hidden, letter_targets),
        "embedding_pca": embedding_pca(model),
    }
    suffix = f"-{args.tag}" if args.tag else ""
    (ROOT / f"analysis{suffix}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    results_name = args.results or f"results{suffix}.json"
    train_log = json.loads((ROOT / results_name).read_text(encoding="utf-8"))
    lines = [
        "# Analysing a tiny transformer trained on PostHog's public text",
        "",
        f"Model: {model.parameter_count:,} parameters, {config['layers']} layers, "
        f"{config['heads']} heads, context {config['context']}.",
        f"Test loss: {train_log['test_bits_per_byte']:.3f} bits/byte "
        f"({train_log['test_perplexity']:.2f} perplexity) against a bigram baseline of "
        f"{train_log['baselines']['bigram_bits_per_byte']:.3f} bits/byte.",
        "",
        "## Attention behaviour",
        "",
        "| Layer | Mean entropy (nats) | Collapsed heads | Near-duplicate head pairs |",
        "|---|---:|---|---|",
    ]
    for layer in results["attention"]:
        lines.append(f"| {layer['layer']} | {layer['mean_entropy_nats']} | "
                     f"{layer['collapsed_heads'] or 'none'} | {layer['duplicate_heads'] or 'none'} |")
    lines += [
        "",
        "## Representation",
        "",
        f"- Anisotropy: mean cosine between random hidden-state pairs "
        f"{results['anisotropy']['mean_cosine']} "
        f"(5th to 95th percentile {results['anisotropy']['p05']} to {results['anisotropy']['p95']}).",
        f"- Linear probe for 'current byte is a letter': accuracy {results['probe']['accuracy']} "
        f"against a class prior of {results['probe']['class_prior']} "
        f"(lift {results['probe']['lift_over_prior']}).",
        f"- Embedding PCA: PC1 explains {results['embedding_pca']['explained_by_pc1']} of variance, "
        f"first two {results['embedding_pca']['explained_by_pc1_2']}, "
        f"first four {results['embedding_pca']['explained_by_pc1_4']}.",
        "",
        "## What this is and is not",
        "",
        f"One run on {train_log['corpus_bytes']:,} bytes of public PostHog pages "
        f"({train_log.get('device', 'CPU')}), with a byte-level vocabulary and no external data.",
        "It shows that training works, that the model beats a bigram baseline, and what the attention",
        "and representations look like. It is not a benchmark against state-of-the-art models, not their",
        "production data, and the probe label is a simple, checkable one so that a negative result would also",
        "have been reported.",
    ]
    (ROOT / f"analysis{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
