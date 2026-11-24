# A byte-level transformer, written from scratch

A decoder-only transformer implemented in plain PyTorch, trained on public pages from
PostHog's site, evaluated against two baselines on a page-level split, analysed for
attention and representation behaviour, and paired with a Triton and a Rust version of
the attention kernel.

Everything here is reproducible from the scripts. The checkpoints are not committed
because they are large and rebuildable.

## Results

| | |
|---|---|
| Model | Decoder-only transformer, written from scratch: multi-head causal attention, pre-LN blocks, GELU feed-forward, weight tying |
| Parameters | 14,380,800 (8 layers, 6 heads, d_model 384, context 256) |
| Corpus | 221 public pages, 2,176,949 bytes, split by page |
| Train / test | 1,585,903 / 591,046 bytes (185 / 36 pages) |
| Device | NVIDIA A10 (Modal), 2,500 steps per seed |
| Held-out loss (3 seeds) | 1.937 bits/byte mean, range 1.920 to 1.956 |
| Train loss | 1.156 bits/byte |
| Baselines, same split | unigram 4.869, bigram 3.742 bits/byte |
| Improvement | 48.2% below the bigram baseline |

The model overfits: 1.156 training against 1.937 held out is what a 14M-parameter model
does to 1.6 MB of text, and the first lever is more data rather than a wider model.

On split hygiene: pages are assigned to train or test by URL hash, so no page appears on
both sides, and a near-duplicate check found no pair above 0.6 shingle overlap. Repeated
site furniture is not stripped, and 1.06% of held-out 8-word n-grams also appear in
training (`results-boilerplate.json`). The number is fair, not pristine; stripping
repeated lines is the obvious way to tighten it.

An earlier run held out the tail of a concatenated corpus and reported 1.753. Pages share
sentences, so that number was optimistic. The page split above is the one quoted.

## Kernels

Inputs are stored in float16 and accumulated in float32, so float16 SDPA is the fair
comparison.

| Kernel | Correctness (vs fp32 ref) | Time | Comparison |
|---|---|---:|---|
| Triton, TF32 dots, 128-wide blocks | 1.2e-3 | 0.181 ms | 2.63x slower than SDPA fp16 (0.069 ms) |
| Triton, IEEE fp32 dots, 64-wide blocks | 9.6e-4 | 17.33 ms | 251x slower than SDPA fp16 |
| PyTorch SDPA, fp16 | reference | 0.069 ms | |
| PyTorch SDPA, fp32 | reference | 0.454 ms | Triton is 2.5x faster than this path, which is not the one to beat |
| Rust, f32, rayon over (batch, head) | 7.7e-7 | 10.58 ms | about 9x slower than PyTorch CPU (1.18 ms) |

No kernel here beats PyTorch's tuned one. The Rust pass is exact to float32 and about
nine times slower; PyTorch measured 1.18 ms in the same container and 1.70 ms in an
earlier smaller one, so treat the gap as roughly 9x with container noise. Adding rayon
over (batch, head) moved it only from 12.4 to 10.6 ms: the score loop is memory-bound and
unvectorised, so threads were never the bottleneck. The next change is a blocked,
SIMD-friendly score kernel.

The Triton kernel took three attempts to become correct: the first returned NaNs, the
second was wrong by 3.7, the third is correct to 1e-3 and about two and a half times
slower than the fused fp16 path. The correctness check runs before any timing is quoted.

## Analysis

`analysis-final.md` and `analysis-final.json` hold the checks on the page-split checkpoint:

- Attention entropy per layer and head: no collapsed heads (never below a quarter of
  maximum) and no near-duplicate head pairs (cosine never above 0.995). Layer 0 is the
  sharpest at 2.21 nats, the last layers diffuse at 2.92.
- Anisotropy: mean cosine between random hidden-state pairs 0.143 (5th to 95th percentile
  -0.084 to 0.458).
- Embedding PCA: PC1 explains 73.4% of token-embedding variance.

`probes.md` reports two linear probes on frozen hidden states, each repeated with shuffled
labels:

| Task | Probe accuracy | Prior | Lift | Control lift |
|---|---:|---:|---:|---:|
| current byte is a letter | 0.9645 | 0.7912 | 0.1733 | 0.0005 |
| current byte inside a URL | 0.9790 | 0.9741 | 0.0049 | 0.0000 |

The first label is readable, the second is not linearly readable at this scale, and the
shuffled control is what makes either statement mean something.

## Next-event prediction on real sequences

PostHog's data problem is event sequences rather than text, so `event_sequences.py` runs
the same loop on MovieLens `ml-latest-small` (public, timed, per-user interactions; URL
and licence in the results file). Same discipline: popularity baseline, first-order
Markov baseline, a small causal transformer, three seeds, three splits.

| Split | Predictions | Popularity top10 | Markov top10 | Transformer top10 |
|---|---:|---:|---:|---:|
| valid (last 10% of each user) | 6,454 | 1.18% | 3.63% | 1.34% |
| future (events after a cutoff) | 15,587 | 2.34% | 4.30% | 2.14% |
| cold (users never seen) | 22,243 | 2.15% | 6.02% | 2.08% |

The transformer loses to Markov on every split and to popularity on two of three. With 396
training sequences, a 7,094-item vocabulary and a two-layer model it is an under-tuned
baseline, not a verdict on the architecture. The standard setting for this problem is
ML-1M with the vocabulary filtered to items seen at least five times and leave-one-out
evaluation, which is the next run.

## Files

```
fetch_corpus.py     collects the first 348 KB of public pages
fetch_more.py       extends it from the sitemap to 2.18 MB across 221 pages
build_pages.py      near-duplicate check and page-level train/test split
tinygpt.py          the model: causal self-attention, pre-LN, GELU, weight tying
train.py            CPU training loop (the first, smaller run)
modal_experiment.py GPU experiment: three seeds, and the kernel benchmark
modal_train.py      the earlier single-seed GPU run and first Triton attempt
modal_rust.py       builds and benchmarks the Rust kernel in a container
event_sequences.py  next-event prediction on public interaction sequences
modal_sequences.py  the sequence model on a GPU
fetch_sequences.py  downloads the sequence dataset inside a container
analyse.py          attention, anisotropy, probe and PCA analysis
probes.py           two probes with shuffled-label controls
sample.py           generations at two temperatures
plot.py             loss curve, entropy chart, embedding PCA
rust-kernel/        the attention forward pass in Rust
RESEARCH-PLAN.md    a plan mapped to the team's published quarterly objectives
results-*.json      every number above, baselines and kernel timings included
analysis-*.md/json  the analysis for each checkpoint
figures/            the charts
```

Reproduce the headline number:

```powershell
python -B fetch_corpus.py
python -B fetch_more.py --limit 250
python -B build_pages.py
python -m modal run modal_experiment.py
```

## Scope

The corpus is public documentation and marketing pages, not PostHog's product data. This
is not a benchmark against state-of-the-art models, not production model training and not
published research. It is a first transformer trained from scratch, built so the method and
the numbers can be checked.

One Rust kernel and one Triton kernel exist here. There is no CUDA work, and the Rust
kernel is correct rather than fast.
