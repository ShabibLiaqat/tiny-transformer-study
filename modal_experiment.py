"""The corrected experiment: deduplicated, page-level split, three seeds, fair kernel comparison.

Two design points worth knowing:

* The corpus is split by whole page (no page appears on both sides) and every
  configuration runs three seeds, so the headline is a mean with a range rather than a
  single draw over a slice of concatenated text.
* The kernel's inputs are float16, so the fair comparison is against fp16 SDPA; the fp32
  number is reported separately rather than used as a headline speedup.

    python -m modal run modal_experiment.py
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
APP = modal.App("posthog-attention-v2")
VOLUME = modal.Volume.from_name("posthog-attention-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_file(str(ROOT / "tinygpt.py"), "/root/tinygpt.py")
)


@APP.function(image=image, gpu="A10G", timeout=3600, volumes={"/runs": VOLUME})
def train_seed(train_bytes: bytes, test_bytes: bytes, config: dict, seed: int) -> dict:
    import math
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from tinygpt import ByteTokenizer, TinyGPT

    torch.manual_seed(seed)
    tokenizer = ByteTokenizer()
    device = "cuda"
    train_ids = torch.tensor(tokenizer.encode(train_bytes.decode("utf-8", "replace")),
                             dtype=torch.long, device=device)
    test_ids = torch.tensor(tokenizer.encode(test_bytes.decode("utf-8", "replace")),
                            dtype=torch.long, device=device)

    model = TinyGPT(vocab=256, d_model=config["d_model"], heads=config["heads"],
                    layers=config["layers"], context=config["context"],
                    dropout=config["dropout"], swiglu=False).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95),
                                  weight_decay=0.1)
    steps, context, batch = config["steps"], config["context"], config["batch"]
    warmup = max(50, steps // 40)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min(1.0, (step + 1) / warmup) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))))

    started = time.time()
    history, running = [], None
    for step in range(1, steps + 1):
        starts = torch.randint(0, len(train_ids) - context - 1, (batch,), device=device)
        x = torch.stack([train_ids[s:s + context] for s in starts])
        y = torch.stack([train_ids[s + 1:s + context + 1] for s in starts])
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        schedule.step()
        value = float(loss.detach())
        running = value if running is None else 0.98 * running + 0.02 * value
        if step % 250 == 0:
            history.append({"step": step, "loss": round(running, 4)})

    model.eval()

    def evaluate(source, rounds=32):
        total = 0.0
        with torch.no_grad():
            for _ in range(rounds):
                starts = torch.randint(0, len(source) - context - 1, (batch,), device=device)
                x = torch.stack([source[s:s + context] for s in starts])
                y = torch.stack([source[s + 1:s + context + 1] for s in starts])
                _, loss = model(x, y)
                total += float(loss)
        return total / rounds

    train_bits = evaluate(train_ids) / math.log(2)
    test_bits = evaluate(test_ids) / math.log(2)
    return {"seed": seed, "train_bits_per_byte": train_bits, "test_bits_per_byte": test_bits,
            "test_perplexity": math.exp(test_bits * math.log(2)),
            "parameters": model.parameter_count, "seconds": round(time.time() - started, 1),
            "history": history}


@APP.function(image=image, gpu="A10G", timeout=1800)
def kernel_fair() -> dict:
    """Triton attention vs PyTorch SDPA, with the comparison made like for like."""
    import time

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def attention_fwd(Q, K, V, O, stride_b, stride_h, stride_m,
                      N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PRECISION: tl.constexpr):
        block_m = tl.program_id(0)
        head = tl.program_id(1)
        batch = tl.program_id(2)
        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        base = batch * stride_b + head * stride_h
        q = tl.load(Q + base + offs_m[:, None] * stride_m + offs_d[None, :],
                    mask=offs_m[:, None] < N_CTX, other=0.0).to(tl.float32)
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        denominator = tl.zeros((BLOCK_M,), dtype=tl.float32)
        running_max = tl.full((BLOCK_M,), -1.0e6, dtype=tl.float32)
        scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        for start_n in range(0, N_CTX, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(K + base + offs_n[None, :] * stride_m + offs_d[:, None],
                        mask=offs_n[None, :] < N_CTX, other=0.0).to(tl.float32)
            scores = tl.dot(q, k, input_precision=PRECISION) * scale
            scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, -1.0e6)
            new_max = tl.maximum(running_max, tl.max(scores, axis=1))
            correction = tl.exp(running_max - new_max)
            weights = tl.exp(scores - new_max[:, None])
            denominator = denominator * correction + tl.sum(weights, axis=1)
            v = tl.load(V + base + offs_n[:, None] * stride_m + offs_d[None, :],
                        mask=offs_n[:, None] < N_CTX, other=0.0).to(tl.float32)
            accumulator = accumulator * correction[:, None] + tl.dot(weights, v,
                                                                     input_precision=PRECISION)
            running_max = new_max
        accumulator = accumulator / denominator[:, None]
        tl.store(O + base + offs_m[:, None] * stride_m + offs_d[None, :],
                 accumulator.to(tl.float16), mask=offs_m[:, None] < N_CTX)

    def timed(fn, rounds=50):
        fn()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(rounds):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / rounds * 1000

    torch.manual_seed(0)
    batch, heads, context, head_dim = 8, 8, 512, 64
    q = torch.randn(batch, heads, context, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)

    def runner(block, precision):
        def run():
            attention_fwd[(triton.cdiv(context, block), heads, batch)](
                q, k, v, out, q.stride(0), q.stride(1), q.stride(2),
                N_CTX=context, HEAD_DIM=head_dim, BLOCK_M=block, BLOCK_N=block,
                PRECISION=precision)
        return run

    reference_fp16 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    reference_fp32 = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True)

    results = {}
    for label, block, precision in (("ieee_64", 64, "ieee"), ("tf32_128", 128, "tf32")):
        run = runner(block, precision)
        run()
        torch.cuda.synchronize()
        difference = (out.float() - reference_fp32).abs().max().item()
        results[label] = {"max_abs_diff_vs_fp32_reference": difference,
                          "millis": round(timed(run), 4)}

    return {
        "device": torch.cuda.get_device_name(0),
        "shape": {"batch": batch, "heads": heads, "context": context, "head_dim": head_dim},
        "storage_dtype": "float16 inputs, float32 accumulation inside the kernel",
        "torch_sdpa_fp16_ms": round(timed(lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True)), 4),
        "torch_sdpa_fp32_ms": round(timed(lambda: torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=True)), 4),
        "triton": results,
    }


@APP.local_entrypoint()
def main(seeds: str = "0,1,2"):
    config = {"steps": 2500, "layers": 8, "d_model": 384, "heads": 6, "context": 256,
              "batch": 48, "lr": 6e-4, "dropout": 0.1}
    train_bytes = (ROOT / "data" / "train.txt").read_bytes()
    test_bytes = (ROOT / "data" / "test.txt").read_bytes()
    print(f"train {len(train_bytes):,} bytes | test {len(test_bytes):,} bytes | config {config}")

    runs = []
    for seed in [int(value) for value in seeds.split(",") if value.strip()]:
        print(f"--- seed {seed}")
        result = train_seed.remote(train_bytes=train_bytes, test_bytes=test_bytes,
                                   config=config, seed=seed)
        print({k: round(result[k], 4) for k in ("train_bits_per_byte", "test_bits_per_byte",
                                                 "test_perplexity")})
        runs.append(result)

    values = sorted(run["test_bits_per_byte"] for run in runs)
    summary = {
        "config": config,
        "train_bytes": len(train_bytes),
        "test_bytes": len(test_bytes),
        "parameters": runs[0]["parameters"],
        "seeds": [run["seed"] for run in runs],
        "test_bits_per_byte": {
            "values": [round(value, 4) for value in values],
            "mean": round(sum(values) / len(values), 4),
            "min": round(values[0], 4),
            "max": round(values[-1], 4),
        },
        "train_bits_per_byte": {
            "values": [round(run["train_bits_per_byte"], 4) for run in runs],
            "mean": round(sum(run["train_bits_per_byte"] for run in runs) / len(runs), 4),
        },
        "runs": runs,
    }
    suffix = "-" + seeds.replace(",", "-") if seeds != "0,1,2" else ""
    (ROOT / f"results-multiseed{suffix}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["test_bits_per_byte"], indent=1))

    kernels = kernel_fair.remote()
    (ROOT / "results-kernel-fair.json").write_text(json.dumps(kernels, indent=1), encoding="utf-8")
    print(json.dumps(kernels, indent=1))

@APP.function(image=image, gpu="A10G", timeout=3600, volumes={"/runs": VOLUME})
def train_seed_save(train_bytes: bytes, test_bytes: bytes, config: dict, seed: int) -> dict:
    """Same training run as train_seed, but keep the checkpoint for local analysis."""
    import math
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from tinygpt import ByteTokenizer, TinyGPT

    torch.manual_seed(seed)
    tokenizer = ByteTokenizer()
    train_ids = torch.tensor(tokenizer.encode(train_bytes.decode("utf-8", "replace")),
                             dtype=torch.long, device="cuda")
    test_ids = torch.tensor(tokenizer.encode(test_bytes.decode("utf-8", "replace")),
                            dtype=torch.long, device="cuda")
    model = TinyGPT(vocab=256, d_model=config["d_model"], heads=config["heads"],
                    layers=config["layers"], context=config["context"],
                    dropout=config["dropout"], swiglu=False).cuda()
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95),
                                  weight_decay=0.1)
    steps, context, batch = config["steps"], config["context"], config["batch"]
    warmup = max(50, steps // 40)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min(1.0, (step + 1) / warmup) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))))
    for step in range(1, steps + 1):
        starts = torch.randint(0, len(train_ids) - context - 1, (batch,), device="cuda")
        x = torch.stack([train_ids[s:s + context] for s in starts])
        y = torch.stack([train_ids[s + 1:s + context + 1] for s in starts])
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        schedule.step()
    torch.save({"model": model.state_dict(), "config": config}, "/runs/model-pagesplit.pt")
    VOLUME.commit()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(32):
            starts = torch.randint(0, len(test_ids) - context - 1, (batch,), device="cuda")
            x = torch.stack([test_ids[s:s + context] for s in starts])
            y = torch.stack([test_ids[s + 1:s + context + 1] for s in starts])
            _, loss = model(x, y)
            total += float(loss)
    return {"seed": seed, "parameters": model.parameter_count,
            "test_bits_per_byte": (total / 32) / math.log(2), "saved": True}


@APP.local_entrypoint()
def save_checkpoint():
    config = {"steps": 2500, "layers": 8, "d_model": 384, "heads": 6, "context": 256,
              "batch": 48, "lr": 6e-4, "dropout": 0.1}
    result = train_seed_save.remote(
        train_bytes=(ROOT / "data" / "train.txt").read_bytes(),
        test_bytes=(ROOT / "data" / "test.txt").read_bytes(),
        config=config, seed=0)
    print(json.dumps(result, indent=1))
