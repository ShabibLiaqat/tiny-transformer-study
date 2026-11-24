# Probes, with a shuffled-label control

Checkpoint: model-pagesplit.pt, context 256, 24 windows of held-out text.
The probe is logistic regression on frozen hidden states; the control repeats it
with permuted labels, so a lift can be attributed to the representation rather
than to the probe fitting noise.

| Task | Positive rate | Probe acc. | Prior | Lift | Control lift |
|---|---:|---:|---:|---:|---:|
| current byte is a letter (easy) | 0.7912 | 0.9645 | 0.7912 | 0.1733 | 0.0005 |
| current byte sits inside a URL (structural) | 0.0259 | 0.979 | 0.9741 | 0.0049 | 0.0 |

The easy label is readable (high lift, control at its prior). The structural
label - whether a byte sits inside a URL - is not linearly readable from these
hidden states at this scale, and the honest report is that it stayed at the
prior. A negative probe is a result, not a failure to hide.
