# Analysing a tiny transformer trained on PostHog's public text

Model: 836,352 parameters, 4 layers, 4 heads, context 96.
Test loss: 3.341 bits/byte (10.13 perplexity) against a bigram baseline of 4.320 bits/byte.

## Attention behaviour

| Layer | Mean entropy (nats) | Collapsed heads | Near-duplicate head pairs |
|---|---:|---|---|
| 0 | 2.768 | none | none |
| 1 | 1.939 | none | none |
| 2 | 2.8941 | none | none |
| 3 | 2.9628 | none | none |

## Representation

- Anisotropy: mean cosine between random hidden-state pairs 0.2355 (5th to 95th percentile -0.1051 to 0.6255).
- Linear probe for 'current byte is a letter': accuracy 0.9878 against a class prior of 0.7899 (lift 0.1979).
- Embedding PCA: PC1 explains 0.8033 of variance, first two 0.8267, first four 0.8542.

## What this is and is not

One CPU-scale run on 348 KB of public PostHog pages, with a byte-level vocabulary and no
external data. It shows that training works, that the model beats a bigram baseline, and what
the attention and representations look like. It is not a benchmark against state-of-the-art
models, not their production data, and the probe label is deliberately trivial so that a
negative result would also have been reported.
