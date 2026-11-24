# A research plan, mapped to a published agenda

PostHog publishes its quarterly objectives for its AI research team. This note works
through each one: what I would do on it, what I have tried at small scale, and where I
would expect to be wrong. It is a plan written while doing the small versions, not a
pitch written after them.

## 1. Finish training the encoder, then evaluate honestly

The most useful thing I can add on day one is evaluation discipline. A small model
trained on 2.18 MB of PostHog's public pages goes from 3.74 to 1.94 bits/byte
against a bigram baseline on a page-level split, in three seeds, on a GPU (run committed with the code).
That number means something only because the baseline is in the same file. On your
cluster the same principle applies at a different scale: hold out by customer and
by time, report bits/byte next to the model you are trying to beat, and keep the
sampling script in the repo so the sample is not cherry-picked.

## 2. Benchmarks against SOTA

Two traps I would guard against: comparing against a weaker baseline than the
literature uses, and comparing on a different preprocessing. I would pin the
tokeniser and the held-out split in a config file, and report the variance across
at least three seeds. For a domain model over product data, the honest comparison
is often "the previous PostHog model", not a public leaderboard, and that is worth
saying out loud.

## 3. Visualisation (PCA and beyond)

Small runs make this cheap to try. On the trained model, PC1 explains 73% of the token
embedding variance, which is the anisotropy problem in one number: distances
between tokens are dominated by one direction. That is why the PCA plot is in the
repo next to the loss curve, and why I would add a cosine-similarity distribution
before trusting any nearest-neighbour claim.

## 4. Probes

Two probes on frozen hidden states, each repeated with shuffled labels as a control: a
byte-is-a-letter probe scores 0.9645 against a prior of 0.7912 (lift 0.173, control lift
0.0005), and a byte-is-inside-a-URL probe scores 0.9790 against a prior of 0.9741 (lift
0.0049, control 0.0000). The second is a negative result: this model does not linearly
encode that piece of document structure, and reporting it as a win would be dishonest.
The next probe should ask something the objective does not already force - whether a
token is inside a code block, or whether a session ends within N events - and keep the
shuffled control next to every number.

## 5. Weights analysis

Attention entropy per layer and head is the cheapest diagnostic that catches real
problems. On the trained model, layer 0 is the sharpest at 2.21 nats while the last layers sit near 2.9, and no head collapsed or duplicated - which is what you want to see before
adding capacity. I would automate the checks: collapsed head, duplicate head pair by
cosine, dead SwiGLU path by activation frequency, and store them per checkpoint so
regressions show up as a diff rather than a hunch.

## 6. Kernels

This is where I have the most to learn and the least to show. I wrote the attention
forward pass twice: in Triton (correct to 1.2e-3, 0.181 ms against SDPA fp16's 0.069 ms) and in Rust
(correct to 7.7e-7, 10.6 ms against PyTorch CPU's 1.18 ms, which is about 9x slower). Rayon over (batch, head) moved the Rust pass only from 12.4 to 10.6 ms: the score loop is memory-bound and unvectorised, so the next step is a blocked SIMD kernel rather than more threads.
Neither is faster than PyTorch's tuned kernel, and saying so is the point. Next: a blocked score
kernel, the Triton backward pass, then a benchmark against the fused fp16 kernel with the correctness
test kept as the gate.

## 7. Anisotropy

Mean cosine between random hidden-state pairs is 0.143, with a 5th-95th percentile range of -0.084 to 0.458. That is not extreme, but it is high
enough to distort cosine-based retrieval. I would track it per checkpoint next to
the loss, and try the usual remedies (whitening, contrastive objective) with the
same seed and split so the comparison is real.

## 8. The problem PostHog actually has: event sequences

Text was a stand-in. `event_sequences.py` runs the same loop on public interaction
sequences (MovieLens) with a popularity baseline, a first-order Markov baseline and a
causal transformer, on three splits: last 10% of known users, events after a time cutoff,
and cold users the model never saw. Top-10 accuracy: Markov 3.63% / 4.30% / 6.02% against the transformer's 1.34% / 2.14% / 2.08%, and
popularity beats the transformer on two of the three splits (2.34% vs 2.14%, 2.15% vs 2.08%). That is an
under-tuned baseline at this scale, not a verdict on transformers: the next run is ML-1M with a filtered
vocabulary and leave-one-out evaluation, and until then the sequence model is reported and not claimed. On PostHog's data the same harness
would tell you within a day whether a learned model earns its cost, and the answer would
be believable because the baselines are in the same file.

## What this note is

A working plan written from the outside by someone who read the objectives and
tried the small versions. The value is in the loop, not the claims: train, measure
against a baseline, look at the weights, write down what broke, repeat.
