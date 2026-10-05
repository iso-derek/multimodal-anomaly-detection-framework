# Synthetic fusion benchmark — 2026-10-05

Seed 42, 1,800 aligned synthetic events; 600 final-test events. Thresholds fixed using validation negatives at a target 5% FPR. All listed scenarios retained 100% coverage because at least three modalities remained available.

| Test scenario | Equal AUROC | Fixed-prior AUROC | Quality-adaptive AUROC |
|---|---:|---:|---:|
| Clean | .9729 | .9751 | .9752 |
| Missing tabular | .9554 | .9572 | .9566 |
| Missing time-series | .9526 | .9546 | .9544 |
| Missing image | .9672 | .9703 | .9696 |
| Missing video | .9716 | .9726 | .9732 |
| Video noise 50% | .9675 | .9699 | .9726 |
| Video noise 100%, known quality | .9340 | .9442 | .9732 |
| Video noise 100%, quality unknown | .9340 | .9442 | .9428 |

Quality adaptation helps substantially when it is told that the video channel is fully corrupted. It slightly underperforms fixed priors on several missing-input scenarios and underperforms fixed priors when corruption is not reflected in quality. This is a conditional mechanism result, not a universal superiority claim.

For full video corruption, adaptive F1 is .8720 and FPR .0482 with known quality, versus F1 .7862 and FPR .0665 with unknown quality. Validation's target FPR does not constrain FPR under shift.

Input hash: `7bc58cea7ae77cbef76814d95592b199fc4883b26e03e5e7d57cc00e0d02c8d9`.

Reproduce with `python scripts/run_research.py`. No real aligned multimodal dataset or trained-detector performance claim is included.
