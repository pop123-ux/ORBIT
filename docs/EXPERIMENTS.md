# Experiment status

The paper's primary experiments were run with ORBIT's optimizer-side $R(+\Delta)$ rotary transport. That architecture is now the canonical method documented by this repository.

## What the audit clarified

For causal separation

```math
\Delta=i-j\ge0,
```

the **forward** RoPE attention score contains

```math
q_{i,f}^{\top}R_f(-\Delta)k_{j,f}.
```

The historical ORBIT implementation used

```math
R_f(+\Delta)
```

when transporting opposite-side Q/K second moments to construct the optimizer metric. The release now documents this explicitly as an optimizer-side design choice rather than describing it as the exact score-side quadratic pullback.

That distinction does not invalidate the non-checkpointed 124M experiments: those experiments genuinely evaluated the $R(+\Delta)$ optimizer that is now specified in the paper and package.

## Release evidence

The manuscript uses the following evidence hierarchy:

1. **Matched Muon–ORBIT confirmation** — the primary paired comparison at 124M/900 steps.
2. **Mechanism ablations** — identity, no-RoPE, and diagonal controls under the matched recipe.
3. **Crossed discovery recipes** — separates update-rule behavior from recipe sensitivity.
4. **124M long-horizon transfer** — descriptive two-seed transfer outside the primary horizon.
5. **Broad 124M benchmark** — exploratory context across the wider optimizer set.

The historical 355M checkpointed cell is not used as release evidence for the reusable package. In that exploratory run, checkpoint recomputation could update ORBIT's running Q/K statistics more than once per logical training step. The current package suppresses statistic collection during recomputation, which is the safer behavior for downstream users.

## Reproduction boundary

The primary 124M experiments did not use gradient checkpointing, so the current package preserves their optimizer geometry while adding engineering hardening around persistent statistics, distributed aggregation, and checkpoint-safe EMA updates.

For strict reproduction of the paper's central result, keep the following fixed:

- the $R(+\Delta)$ optimizer-side transport;
- the RoPE distance grid $\{1,2,4,8,16,32,64,128\}$;
- uncentered Q/K second moments with $\beta=0.95$;
- inverse-square-root preconditioning with $10^{-5}I$ regularization and condition cap 100;
- one joint Q/K Frobenius restoration factor;
- the exact frozen matched recipe and paired seeds in [`configs/paper_124m_matched.json`](../configs/paper_124m_matched.json).

The primary held-out seed set is `500` through `509`. The matched tuning seed is `0`, and both optimizers selected `shared-04` from the same deterministic ten-candidate grid. The config file records the full-precision learning rate, weight decay, auxiliary multiplier, data specification, and secondary study seed sets.

For direct reproduction, [`experiments/paper_run.py`](../experiments/paper_run.py) reads the frozen config so users do not need to rediscover the primary recipe. [`experiments/matched.py`](../experiments/matched.py) remains available to audit or rerun the selection procedure itself.

The full historical paper campaign remains in the paper-development repository used to generate the manuscript artifacts; the standalone package focuses on the primary Muon–ORBIT result and ORBIT-specific mechanism variants.
