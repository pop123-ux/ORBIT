# Experiments

The main comparison used a RoPE GPT-style decoder on FineWeb-Edu. Context length was 512. The 124M model used 12 layers, 12 heads, width 768, and batch size 8; the 355M transfer model used 24 layers, 16 heads, width 1024, batch size 4, and gradient checkpointing.

All comparisons used the same warmup/cosine schedule, Muon momentum 0.95, and five Newton-Schulz iterations.

## Matched Muon comparison

Muon and ORBIT were given the same ten candidate configurations. Both selected the same configuration before the held-out evaluation. The confirmation used seeds 500-509 at 124M for 900 steps.

```text
Muon mean loss      5.495082
ORBIT mean loss     5.488686
ORBIT - Muon       -0.006396
95% CI             [-0.009838, -0.002955]
ORBIT wins          9 / 10
```

Mean runtime was 802.9 s for Muon and 859.3 s for ORBIT in this setup. Peak allocated CUDA memory was 4.710 GB and 4.700 GB respectively.

## Mechanism ablations

The extended ablation used the matched ORBIT configuration and seeds 600-609.

| comparison | mean ORBIT delta | 95% CI | ORBIT wins |
| --- | ---: | --- | ---: |
| full vs identity | -0.008398 | [-0.011634, -0.005163] | 9/10 |
| full vs no-RoPE | -0.003963 | [-0.006060, -0.001867] | 10/10 |
| full vs diagonal | -0.001952 | [-0.003850, -0.000054] | 8/10 |

The identity comparison measures the effect of functional Q/K conditioning. The no-RoPE comparison removes relative-position transport. The diagonal variant removes off-diagonal coupling inside each 2x2 frequency pair.

## Transfer runs

At 124M/2700 steps, two seeds gave mean ORBIT deltas of -0.3759 vs Muon, -0.2543 vs NorMuon, and -0.0587 vs ASTRO-v2.

At 355M/900 steps, two seeds gave mean deltas of -0.2126 vs Muon and -0.1896 vs NorMuon. The ASTRO-v2 comparison was mixed: one seed favored each method, with a mean ORBIT-minus-ASTRO delta of +0.0297.

The transfer cells use independently frozen optimizer recipes and only two seeds. They are useful checks that the method still trains outside the primary setting, not high-powered matched estimates. The 355M run also uses a smaller batch, so it should not be read as a controlled scaling law.
