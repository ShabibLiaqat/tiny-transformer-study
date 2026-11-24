"""Build and benchmark the Rust attention kernel on Modal, next to PyTorch CPU.

The local machine has no MSVC linker, and installing Visual Studio build tools to
compile one file is a poor trade. A Modal container has a working toolchain, so the
Rust source lives in this repo and the benchmark runs where it compiles.

    python -m modal run modal_rust.py
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
APP = modal.App("attention-kernel-bench")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "curl")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("torch", "numpy")
    .add_local_dir(str(ROOT / "rust-kernel"), "/root/rust-kernel")
)


@APP.function(image=image, cpu=8, timeout=1800)
def benchmark(batch: int = 4, heads: int = 8, seq: int = 256, dim: int = 64, rounds: int = 20) -> dict:
    import subprocess
    import time

    import numpy as np
    import torch

    build = subprocess.run(["cargo", "build", "--release"], cwd="/root/rust-kernel",
                           capture_output=True, text=True, timeout=900)
    if build.returncode != 0:
        return {"error": (build.stderr or build.stdout)[-800:]}

    rng = np.random.default_rng(0)
    shape = (batch, heads, seq, dim)
    q, k, v = (rng.standard_normal(shape).astype(np.float32) for _ in range(3))

    def write(path, array):
        header = np.array(shape, dtype=np.uint32).tobytes()
        Path(path).write_bytes(header + array.tobytes())

    write("/tmp/q.bin", q)
    write("/tmp/k.bin", k)
    write("/tmp/v.bin", v)
    binary = "/root/rust-kernel/target/release/attention-kernel"

    outputs = []
    for _ in range(rounds):
        result = subprocess.run([binary, "/tmp/q.bin", "/tmp/k.bin", "/tmp/v.bin", "/tmp/out.bin"],
                                capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {"error": result.stderr[-800:]}
        outputs.append(json.loads(result.stdout))
    rust_ms = sorted(item["millis"] for item in outputs)[len(outputs) // 2]

    produced = np.frombuffer(Path("/tmp/out.bin").read_bytes(), dtype=np.float32).reshape(shape)
    qt, kt, vt = (torch.from_numpy(array) for array in (q, k, v))
    reference = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
    difference = float(np.abs(produced - reference.numpy()).max())

    torch.rounds = rounds
    torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
    started = time.perf_counter()
    for _ in range(rounds):
        torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
    torch_ms = (time.perf_counter() - started) / rounds * 1000

    return {
        "shape": {"batch": batch, "heads": heads, "seq": seq, "head_dim": dim},
        "rust_ms_median": round(rust_ms, 3),
        "torch_cpu_ms": round(torch_ms, 3),
        "torch_threads": torch.get_num_threads(),
        "max_abs_difference_vs_torch": difference,
        "note": "Rust kernel: rayon over (batch, head) with blocked value accumulation; "
                "PyTorch uses its optimised CPU kernel, so a slowdown is expected. Parallelising "
                "barely moved it because the score loop is memory-bound and unvectorised.",
    }


@APP.local_entrypoint()
def main():
    results = benchmark.remote()
    (ROOT / "results-rust.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
