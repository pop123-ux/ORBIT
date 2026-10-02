# Experiment status

The implementation was audited before release. Two correctness issues were found in the code used by the earlier ORBIT experiment campaign:

1. the RoPE metric transported covariance with (R(+Delta)) while the causal score produced by the model is (q_i^	op R(-Delta)k_j) for (Delta=i-jge0);
2. gradient-checkpoint recomputation updated the Q/K covariance EMA a second time.

Both issues are corrected in the current repository and covered by regression tests.

## Consequence for the archived results

The earlier numerical results are retained in the development evidence store, but they are **not release evidence for the corrected implementation**.

The sign correction affects every result that used the full transported ORBIT metric, including:

- the matched Muon–ORBIT confirmation;
- full ORBIT vs no-RoPE;
- full ORBIT vs diagonal;
- longer-horizon ORBIT comparisons;
- the 355M ORBIT comparisons;
- the independently tuned broad ORBIT cell.

The identity and no-RoPE control implementations do not use the signed RoPE transport, but their previously reported *paired differences against full ORBIT* still depend on the pre-fix full ORBIT runs and must be recomputed.

The 355M ORBIT runs also used gradient checkpointing and therefore require rerunning for the independent EMA-cadence fix.

## Release gate

The manuscript should not restore numerical claims until the corrected campaign has produced:

- a new matched Muon–ORBIT confirmation on held-out paired seeds;
- a new mechanism ablation with full, identity, no-RoPE and diagonal variants;
- corrected longer-horizon and 355M transfer cells;
- per-seed result files sufficient to recompute every reported mean, paired difference and confidence interval.

The broad independently tuned comparison is secondary context only. It must not be used to isolate the ORBIT mechanism because optimizer-specific recipe selection is a confound.

## Reproducibility requirement

For release, the repository should include a compact experiment harness and machine-readable per-seed results for the final corrected campaign. Paper-generation files do not need to live in this repository, but the reported numbers must be reconstructible from the released experiment records.
