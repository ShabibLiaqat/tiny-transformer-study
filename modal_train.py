"""Run the real training job on Modal GPUs, and benchmark a Triton attention kernel.

The CPU run in this repo proved the pipeline at 836k parameters. This is the same
model, bigger, trained on the extended corpus on an A10G, which is where the numbers
worth quoting come from. The second function answers the kernel question with code:
a Triton attention forward pass, checked against PyTorch's own implementation and
timed on the same GPU.

    python -m modal run modal_train.py::train --limit-pages 0
    python -m modal run modal_train.py::kernel

Requires the local Modal profile in ~/.modal.toml (already configured here).
"""

import json
import math
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
APP = modal.App("posthog-attention")
VOLUME = modal.Volume.from_name("posthog-attention-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_file(str(ROOT / "tinygpt.py"), "/root/tinygpt.py")
)


def _baselines(train_bytes, test_bytes):
    counts, totals = {}, len(train_bytes)
    for value in set(train_bytes):
        counts[value] = train_bytes.count(value)
    unigram = -sum(math.log(max(counts.get(b, 0), 1e-9) / totals) for b in test_bytes) / len(test_bytes)

    pairs, previous = {}, {}
    for left, right in zip(train_bytes, train_bytes[1:]):
        pairs[(left, right)] = pairs.get((left, right), 0) + 1
        previous[left] = previous.get(left, 0) + 1
    total = 0.0
    for left, right in zip(test_bytes, test_bytes[1:]):
        p = pairs.get((left, right), 0) / max(previous.get(left, 0), 1)
        total -= math.log(max(p, 1e-9))
    return {"unigram_bits_per_byte": unigram / math.log(2),
            "bigram_bits_per_byte": (total / (len(test_bytes) - 1)) / math.log(2)}


@APP.function(image=image, gpu="A10G", timeout=3600, volumes={"/runs": VOLUME})
def train(corpus: bytes, config: dict) -> dict:
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from tinygpt import ByteTokenizer, TinyGPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.get("seed", 7))
    tokenizer = ByteTokenizer()
    text = corpus.decode("utf-8", "replace")
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(len(ids) * 0.95)
    train_ids, test_ids = ids[:split], ids[split:]

    baselines = _baselines(train_ids.tolist(), test_ids.tolist())
    model = TinyGPT(vocab=256, d_model=config["d_model"], heads=config["heads"],
                    layers=config["layers"], context=config["context"],
                    dropout=config.get("dropout", 0.1),
                    swiglu=config.get("swiglu", False)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95),
                                  weight_decay=0.1)
    steps = config["steps"]
    warmup = max(50, steps // 40)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min(1.0, (step + 1) / warmup) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))))

    context, batch, accum = config["context"], config["batch"], config.get("accum", 1)
    train_ids, test_ids = train_ids.to(device), test_ids.to(device)
    history = []
    running = None
    started = time.time()
    for step in range(1, steps + 1):
        starts = torch.randint(0, len(train_ids) - context - 1, (batch,), device=device)
        x = torch.stack([train_ids[s:s + context] for s in starts])
        y = torch.stack([train_ids[s + 1:s + context + 1] for s in starts])
        _, loss = model(x, y)
        (loss / accum).backward()
        if step % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            schedule.step()
        value = float(loss.detach())
        running = value if running is None else 0.98 * running + 0.02 * value
        if step % 100 == 0 or step == 1:
            elapsed = time.time() - started
            print(f"step {step} | loss {running:.3f} | {elapsed:.0f}s", flush=True)
            history.append({"step": step, "loss": round(running, 4)})

    model.eval()

    def evaluate(source, rounds=24):
        total = 0.0
        with torch.no_grad():
            for _ in range(rounds):
                starts = torch.randint(0, len(source) - context - 1, (batch,), device=device)
                x = torch.stack([source[s:s + context] for s in starts])
                y = torch.stack([source[s + 1:s + context + 1] for s in starts])
                _, loss = model(x, y)
                total += float(loss)
        return total / rounds

    train_loss = evaluate(train_ids)
    test_loss = evaluate(test_ids)

    samples = []
    for prompt, temperature in (("PostHog is", 0.7), ("Session replay", 0.8), ("def ", 0.7)):
        ids_gen = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(160):
                window = ids_gen[:, -context:]
                logits, _ = model(window)
                probabilities = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, 1)
                ids_gen = torch.cat([ids_gen, next_id], dim=1)
        samples.append({"prompt": prompt, "temperature": temperature,
                        "text": tokenizer.decode(ids_gen[0].tolist())})

    results = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "parameters": model.parameter_count,
        "config": config,
        "corpus_bytes": int(len(ids)),
        "train_bytes": int(len(train_ids)),
        "test_bytes": int(len(test_ids)),
        "baselines": baselines,
        "train_bits_per_byte": train_loss / math.log(2),
        "test_bits_per_byte": test_loss / math.log(2),
        "test_perplexity": math.exp(test_loss),
        "history": history,
        "samples": samples,
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    torch.save({"model": model.state_dict(), "config": config}, "/runs/model-gpu.pt")
    Path("/runs/results-gpu.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    VOLUME.commit()
    return results


@APP.function(image=image, gpu="A10G", timeout=1800)
def kernel() -> dict:
    """A Triton attention forward kernel, checked and timed against PyTorch."""
    import time

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def attention_fwd(Q, K, V, O, stride_b, stride_h, stride_m,
                      N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      PRECISION: tl.constexpr):
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

        # Online softmax: every block rescales what came before it, so a block whose
        # entries are all causally masked contributes nothing instead of leaking a
        # weight of one. The first version of this kernel got that wrong and the
        # correctness check caught it.
        for start_n in range(0, N_CTX, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(K + base + offs_n[None, :] * stride_m + offs_d[:, None],
                        mask=offs_n[None, :] < N_CTX, other=0.0).to(tl.float32)
            scores = tl.dot(q, k, input_precision=PRECISION) * scale
            scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, -1.0e6)
            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            correction = tl.exp(running_max - new_max)
            weights = tl.exp(scores - new_max[:, None])
            denominator = denominator * correction + tl.sum(weights, axis=1)
            v = tl.load(V + base + offs_n[:, None] * stride_m + offs_d[None, :],
                        mask=offs_n[:, None] < N_CTX, other=0.0).to(tl.float32)
            accumulator = accumulator * correction[:, None] + tl.dot(weights, v, input_precision=PRECISION)
            running_max = new_max

        accumulator = accumulator / denominator[:, None]
        tl.store(O + base + offs_m[:, None] * stride_m + offs_d[None, :],
                 accumulator.to(tl.float16), mask=offs_m[:, None] < N_CTX)

    torch.manual_seed(0)
    batch, heads, context, head_dim, block = 8, 8, 512, 64, 64
    q = torch.randn(batch, heads, context, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)

    def make_runner(block_m, precision):
        def runner():
            attention_fwd[(triton.cdiv(context, block_m), heads, batch)](
                q, k, v, out, q.stride(0), q.stride(1), q.stride(2),
                N_CTX=context, HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_m,
                PRECISION=precision)
        return runner

    run_triton = make_runner(block, "ieee")

    run_triton()
    torch.cuda.synchronize()
    # Compare in float32: the kernel accumulates in fp32, and comparing an fp16
    # reference against it measures rounding, not correctness.
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True)
    difference = (out.float() - reference).abs().max().item()

    def timed(fn, rounds=50):
        fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(rounds):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / rounds * 1000

    reference_ms = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True))
    reference_fp32_ms = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True))
    fast = make_runner(128, "tf32")
    fast()
    torch.cuda.synchronize()
    fast_out = out.clone()
    fast_difference = (fast_out.float() - reference).abs().max().item()
    fast_ms = timed(fast)
    triton_ms = timed(run_triton)
    return {
        "device": torch.cuda.get_device_name(0),
        "shape": {"batch": batch, "heads": heads, "context": context, "head_dim": head_dim},
        "max_abs_difference_vs_torch_sdpa": difference,
        "triton_ms": round(triton_ms, 4),
        "torch_sdpa_ms": round(reference_ms, 4),
        "speedup": round(reference_ms / triton_ms, 3),
        "torch_sdpa_fp32_ms": round(reference_fp32_ms, 4),
        "speedup_vs_fp32": round(reference_fp32_ms / triton_ms, 3),
        "tf32_128_ms": round(fast_ms, 4),
        "tf32_128_max_abs_difference": fast_difference,
        "tf32_128_speedup_vs_fp32": round(reference_fp32_ms / fast_ms, 3),
    }

@APP.local_entrypoint()
def train_entry(steps: int = 2500, layers: int = 8, d_model: int = 384, heads: int = 6,
                context: int = 256, batch: int = 48, lr: float = 6e-4):
    """Local entrypoint: upload the corpus once, train on an A10G, keep the results."""
    config = {"steps": steps, "layers": layers, "d_model": d_model, "heads": heads,
              "context": context, "batch": batch, "lr": lr, "dropout": 0.1, "seed": 7}
    corpus = (ROOT / "data" / "corpus.txt").read_bytes()
    print(f"corpus {len(corpus):,} bytes | config {config}")
    results = train.remote(corpus=corpus, config=config)
    (ROOT / "results-gpu.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({key: results[key] for key in
                      ("device", "parameters", "test_bits_per_byte", "test_perplexity",
                       "wall_clock_seconds")}, indent=1))


@APP.local_entrypoint()
def kernel_entry():
    results = kernel.remote()
    (ROOT / "results-triton.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
