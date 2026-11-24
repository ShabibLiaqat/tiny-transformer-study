"""Draw the figures that make the run checkable at a glance.

Three charts: the training loss curve against the two baselines, per-layer attention
entropy, and a PCA of the token embeddings coloured by byte class. They are small,
plain and reproducible from the committed JSON.

    python -B plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tinygpt import ByteTokenizer, TinyGPT

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"


def loss_curve(tag="", results_name="results.json"):
    results = json.loads((ROOT / results_name).read_text(encoding="utf-8"))
    steps = [row["step"] for row in results["history"]]
    bits = [row["loss"] / 0.693147 for row in results["history"]]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(steps, bits, color="#1d4ed8", linewidth=2, label="training loss (bits/byte)")
    axis.axhline(results["baselines"]["bigram_bits_per_byte"], color="#b45309",
                 linestyle="--", linewidth=1.2, label="bigram baseline")
    axis.axhline(results["baselines"]["unigram_bits_per_byte"], color="#6b7280",
                 linestyle=":", linewidth=1.2, label="unigram baseline")
    axis.axhline(results["test_bits_per_byte"], color="#047857",
                 linestyle="-", linewidth=1.2, label="held-out loss")
    axis.set_xlabel("training step")
    axis.set_ylabel("bits per byte")
    axis.set_title(f"{results['parameters']:,}-parameter transformer on PostHog public text "
                   f"({results.get('device', 'CPU')})")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = FIG / (f"loss-curve-{tag}.png" if tag else "loss-curve.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def entropy_chart(tag="", analysis_name="analysis.json"):
    analysis = json.loads((ROOT / analysis_name).read_text(encoding="utf-8"))
    layers = analysis["attention"]
    figure, axis = plt.subplots(figsize=(6, 3.4))
    width = 0.2
    for head in range(len(layers[0]["heads"])):
        values = [layer["heads"][head]["entropy_nats"] for layer in layers]
        positions = [layer["layer"] + (head - 1.5) * width for layer in layers]
        axis.bar(positions, values, width=width, label=f"head {head}")
    axis.set_xlabel("layer")
    axis.set_ylabel("mean attention entropy (nats)")
    axis.set_title("Where attention sharpens: entropy per layer and head")
    axis.legend(frameon=False, fontsize=8, ncol=4)
    axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    path = FIG / (f"attention-entropy-{tag}.png" if tag else "attention-entropy.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def embedding_pca(tag="", checkpoint_name="model.pt"):
    checkpoint = torch.load(ROOT / checkpoint_name, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = TinyGPT(vocab=256, d_model=config["d_model"], heads=config["heads"],
                    layers=config["layers"], context=config["context"],
                    swiglu=config.get("swiglu", False))
    model.load_state_dict(checkpoint["model"])
    matrix = model.tok.weight.detach()
    centred = matrix - matrix.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(centred, full_matrices=False)
    projected = (centred @ vh[:2].T).numpy()

    def kind(value):
        if 97 <= value <= 122:
            return "lowercase"
        if 65 <= value <= 90:
            return "uppercase"
        if 48 <= value <= 57:
            return "digit"
        if value in (32, 10, 9):
            return "whitespace"
        return "punctuation"

    colours = {"lowercase": "#1d4ed8", "uppercase": "#b45309", "digit": "#047857",
               "whitespace": "#6b7280", "punctuation": "#7c3aed"}
    figure, axis = plt.subplots(figsize=(6.4, 5))
    for label in colours:
        points = [projected[i] for i in range(256) if kind(i) == label]
        if not points:
            continue
        axis.scatter([p[0] for p in points], [p[1] for p in points], s=16,
                     alpha=0.75, label=label, color=colours[label])
    centred_matrix = matrix - matrix.mean(0, keepdim=True)
    singular = torch.linalg.svd(centred_matrix, full_matrices=False)[1]
    share = (singular ** 2) / (singular ** 2).sum()
    axis.set_title(f"Token embeddings: PC1 explains {share[0].item():.0%} of variance")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = FIG / (f"embedding-pca-{tag}.png" if tag else "embedding-pca.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main():
    FIG.mkdir(exist_ok=True)
    for path in (loss_curve(), entropy_chart(), embedding_pca()):
        print("wrote", path.relative_to(ROOT))
    for path in (loss_curve("pagesplit", "results-pagesplit.json"),
                 entropy_chart("final", "analysis-final.json"),
                 embedding_pca("final", "model-pagesplit.pt")):
        print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
