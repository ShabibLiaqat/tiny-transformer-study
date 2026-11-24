# Analysing a tiny transformer trained on PostHog's public text

Model: 14,380,800 parameters, 8 layers, 6 heads, context 256.
Test loss: 1.937 bits/byte (3.83 perplexity) against a bigram baseline of 3.742 bits/byte.

## Attention behaviour

| Layer | Mean entropy (nats) | Collapsed heads | Near-duplicate head pairs |
|---|---:|---|---|
| 0 | 2.2107 | none | none |
| 1 | 2.62 | none | none |
| 2 | 2.5398 | none | none |
| 3 | 2.7712 | none | none |
| 4 | 2.9152 | none | none |
| 5 | 3.1544 | none | none |
| 6 | 3.3016 | none | none |
| 7 | 3.6094 | none | none |

## Representation

- Anisotropy: mean cosine between random hidden-state pairs 0.143 (5th to 95th percentile -0.0836 to 0.4579).
- Linear probe for 'current byte is a letter': accuracy 0.9782 against a class prior of 0.7897 (lift 0.1885).
- Embedding PCA: PC1 explains 0.7335 of variance, first two 0.7587, first four 0.7782.

## What this is and is not

One run on 2,176,949 bytes of public PostHog pages (NVIDIA A10 (Modal)), with a byte-level vocabulary and no external data.
It shows that training works, that the model beats a bigram baseline, and what the attention
and representations look like. It is not a benchmark against state-of-the-art models, not their
production data, and the probe label is a simple, checkable one so that a negative result would also
have been reported.
