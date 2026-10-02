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

The corrected primary search uses Muon only. Ten deterministic 124M/900-step Muon candidates are evaluated at seed 0; the lowest-loss Muon candidate is frozen before any held-out Muon–ORBIT comparison and is then applied unchanged to both methods. ORBIT tuning outcomes do not influence the primary recipe.

## Evidence hierarchy for release

The final paper uses the following hierarchy:

1. **Matched Muon–ORBIT confirmation** — the sole primary confirmatory contrast. Muon is tuned over ten deterministic candidates; its winner is frozen and applied unchanged to Muon and ORBIT on held-out paired seeds.
2. **Mechanism ablations** — secondary full/identity/no-RoPE/diagonal analyses under that same Muon-selected frozen recipe. Their confidence intervals are reported without multiplicity adjustment.
3. **Cross-configuration experiment** — separates update-rule behavior from recipe sensitivity.
4. **Long-horizon and 355M cells** — descriptive transfer checks only.
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

The standalone reproduction path is split deliberately: [`experiments/matched.py`](../experiments/matched.py) generates and freezes the Muon-selected primary recipe, while [`experiments/run.py`](../experiments/run.py) executes individual training/evaluation runs. Neither script contains manuscript-generation code or JSONL campaign machinery. Corrected per-seed records will be committed under `results/corrected/` only after the audited campaign completes.
