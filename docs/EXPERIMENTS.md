# Experiment status

The manuscript is the source of truth for the experimental protocol. A pre-release correctness audit changed the ORBIT implementation used by the earlier campaign, so the historical numerical results are retained only as development provenance and are **not release evidence**.

## Why the campaign must be rerun

Two findings affect paper claims directly.

First, the earlier full ORBIT path transported the local metric with $R(+\Delta)$ even though the reference model’s causal RoPE score is

```math
q_{i,f}^{\top}R_f(-\Delta)k_{j,f},
\qquad
\Delta=i-j\ge0.
```

The corrected implementation and paper now use the same $R(-\Delta)$ convention.

Second, gradient-checkpoint recomputation previously updated the Q/K second-moment EMA a second time. The corrected implementation suppresses statistics collection during recomputation, so the 355M cells must also be rerun under the intended EMA cadence.

## Affected evidence

Every result containing the old **full ORBIT** implementation must be regenerated, including:

- matched Muon–ORBIT tuning and held-out confirmation;
- full ORBIT vs identity, no-RoPE, and diagonal controls;
- cross-configuration cells containing ORBIT;
- longer-horizon ORBIT comparisons;
- 355M ORBIT comparisons;
- the broad independently selected ORBIT cell.

Although `orbit_identity` and `orbit_norope` do not use signed RoPE transport, their previously reported paired differences against full ORBIT are not reusable.

The corrected matched search must also be rerun rather than reusing the old ORBIT-selected hyperparameter configuration, because the corrected update rule can change the selected recipe.

## Evidence hierarchy for release

The final paper uses the following hierarchy:

1. **Matched Muon–ORBIT confirmation** — primary mechanism estimate. Both methods receive the same deterministic candidate set; the selected configuration is frozen before held-out paired seeds.
2. **Mechanism ablations** — full, identity, no-RoPE, and diagonal under the corrected matched ORBIT recipe.
3. **Cross-configuration experiment** — separates update-rule behavior from recipe sensitivity.
4. **Long-horizon and 355M cells** — secondary transfer checks only.
5. **Broad independently selected benchmark** — exploratory context, not an isolated optimizer ranking.

The unusually large historical transfer gaps and the difference between the historical broad Muon and matched Muon cells are therefore not used as evidence for a larger ORBIT mechanism effect.

## Release gate

Before release, the corrected campaign must provide machine-readable per-seed records sufficient to reconstruct every reported:

- mean validation loss;
- paired difference;
- confidence interval;
- win count;
- runtime and memory summary used in the manuscript.

The paper build is bound to the audited implementation digest and rejects the pre-audit evidence digest.

The standalone reproduction path is [`experiments/run.py`](../experiments/run.py). It reproduces the primary Muon–ORBIT and ORBIT-ablation protocol without manuscript-generation code or JSONL campaign machinery. Corrected per-seed records will be committed under `results/corrected/` only after the audited campaign completes.
