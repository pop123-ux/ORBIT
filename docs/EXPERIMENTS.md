# Experiments

This document summarizes the evidence reported for ORBIT. The primary claim comes from the matched Muon comparison; the broader optimizer table and transfer runs are secondary context.

## Experimental setup

The main experiments used RoPE GPT-style decoders trained on FineWeb-Edu with a context length of 512.

| Model | Layers | Heads | Width | Batch | Gradient checkpointing |
| --- | ---: | ---: | ---: | ---: | --- |
| 124M | 12 | 12 | 768 | 8 | No |
| 355M | 24 | 16 | 1024 | 4 | Yes |

Across the reported comparisons, the optimizer schedule used a 10% linear warmup followed by cosine decay to \(0.1\times\) the peak learning rate, gradient clipping at 1.0, Muon momentum 0.95, and five Newton-Schulz iterations.

## Primary result: matched Muon comparison

The main question is whether ORBIT still improves on Muon when the surrounding hyperparameter recipe is held fixed.

Muon and ORBIT received the same deterministic ten-candidate search grid. Both independently selected the same configuration before any held-out confirmation run was evaluated.

The confirmation used ten paired seeds, 500-509, on the 124M model for 900 steps.

| Statistic | Muon | ORBIT |
| --- | ---: | ---: |
| Mean validation loss | 5.495082 | **5.488686** |
| Mean runtime | 802.9 s | 859.3 s |
| Peak allocated CUDA memory | 4.710 GB | 4.700 GB |

For each seed \(s\), the paired effect is

$$
d_s
=
L_{\mathrm{ORBIT},s}
-
L_{\mathrm{Muon},s}.
$$

The mean paired effect was

$$
\bar d=-0.006396\ \text{nats},
$$

with a two-sided 95% Student-\(t\) confidence interval

$$
[-0.009838,\,-0.002955].
$$

ORBIT had the lower validation loss on **9 of 10 paired runs**.

This is the primary mechanism-isolated estimate in the study because optimizer and selected hyperparameters are matched.

## Mechanism ablations

The extended ablation used the matched ORBIT configuration and ten new seeds, 600-609.

| Comparison | Mean ORBIT delta | 95% CI | ORBIT wins |
| --- | ---: | --- | ---: |
| Full vs identity | -0.008398 | [-0.011634, -0.005163] | 9/10 |
| Full vs no-RoPE | -0.003963 | [-0.006060, -0.001867] | 10/10 |
| Full vs diagonal | -0.001952 | [-0.003850, -0.000054] | 8/10 |

The controls isolate different parts of the method:

- **Identity** keeps the ORBIT statistics path but removes functional preconditioning.
- **No-RoPE** keeps Q/K covariance conditioning but removes relative-position transport.
- **Diagonal** retains RoPE-aware per-frequency scaling but removes off-diagonal coupling inside each \(2\times2\) pair.

The largest contrast is against the identity control. RoPE-aware transport adds a further consistent effect, while the full off-diagonal coupling contributes a smaller increment in this setting.

## Broader optimizer context

A separate 124M/900-step benchmark compared complete independently selected optimizer recipes over five seeds each.

| Optimizer | Mean validation loss |
| --- | ---: |
| AdamW | 5.9280 |
| Muon | 5.6171 |
| AdaMuon | 5.6022 |
| NorMuon | 5.5152 |
| **ORBIT** | **5.4769** |
| ASTRO | 5.4741 |

This table should not be interpreted as a pure optimizer-mechanism ranking. Each method carries its independently selected recipe, so learning-rate and regularization choices are part of the observed differences.

The matched Muon experiment above is the appropriate result for isolating the ORBIT update itself.

## Longer-horizon transfer

At 124M and 2700 steps, two paired seeds gave the following mean ORBIT-minus-baseline differences:

| Baseline | Mean delta |
| --- | ---: |
| Muon | -0.3759 |
| NorMuon | -0.2543 |
| ASTRO | -0.0587 |

Both tested seeds favored ORBIT in each of these comparisons.

Because \(n=2\), this is transfer evidence rather than a high-powered statistical estimate.

## 355M transfer

At 355M and 900 steps, again with two seeds:

| Baseline | Mean ORBIT-minus-baseline delta |
| --- | ---: |
| Muon | -0.2126 |
| NorMuon | -0.1896 |
| ASTRO | +0.0297 |

Both tested seeds favored ORBIT over Muon and NorMuon. The ASTRO comparison was mixed, with one seed favoring each method.

The 355M run uses batch size 4 rather than 8 and therefore a different sampled-token budget per optimizer step. It demonstrates transfer to the larger tested model; it is **not** a controlled scaling law.

## Cost

In the primary 124M matched setting, mean wall-clock time increased from 802.9 s to 859.3 s, approximately a 7% training-time overhead.

Peak allocated CUDA memory was effectively unchanged in the recorded runs.

ORBIT is an optimizer-only modification, so the covariance statistics and metric transforms are discarded after training. The inference graph is unchanged.

## Statistical interpretation

The primary estimand is the paired validation-loss difference

$$
L_{\mathrm{ORBIT}}-L_{\mathrm{baseline}}.
$$

Negative values favor ORBIT.

Confidence intervals reported for the matched and ablation experiments are two-sided Student-\(t\) intervals over the paired per-seed differences.

The evidence is intentionally separated into:

1. **matched confirmation** — primary mechanism estimate;
2. **ablation evidence** — mechanism attribution;
3. **long-horizon / 355M transfer** — secondary transfer checks;
4. **independently tuned benchmark** — practical context, not a matched causal comparison.

This separation avoids treating hyperparameter-recipe effects as if they were entirely caused by the optimizer update rule.
