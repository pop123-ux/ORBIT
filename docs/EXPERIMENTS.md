# Experimental design and evidence hierarchy

ORBIT was evaluated as a sequence of increasingly strict questions rather than as one undifferentiated benchmark.

## 1. Broad optimizer context

**Question:** Where does ORBIT sit among commonly used and recent matrix optimizers when each method carries its own selected recipe?

124M, 900 steps, five seeds per optimizer.

Methods: AdamW, AdaMuon, Muon, NorMuon, ASTRO, ORBIT.

This experiment is useful context, but optimizer and hyperparameter-recipe effects are mixed. It is not the main causal claim.

## 2. Crossed recipes

**Question:** Does ORBIT still improve when the surrounding recipe is frozen?

Muon and ORBIT are each evaluated under both the Muon-selected and ORBIT-selected configurations.

Result: ORBIT has lower validation loss in all five paired seeds under both frozen recipes.

The experiment also shows that recipe choice itself can be larger than the update-rule effect, which is why the broad tuned gap is not interpreted as purely algorithmic.

## 3. Matched confirmation — primary result

**Question:** If Muon and ORBIT receive the same search space, same candidate configurations, and ultimately the same selected hyperparameters, is there still a reproducible ORBIT effect?

- shared deterministic grid: 10 candidates
- both optimizers independently selected candidate `shared-04`
- held-out seeds: 500–509
- 124M / 900 steps

Primary paired result:

```text
Muon mean loss     5.495082
ORBIT mean loss    5.488686
ORBIT - Muon      -0.006396 nats
95% CI            [-0.009838, -0.002955]
ORBIT wins         9 / 10
```

This is the paper's central mechanism-isolated estimate.

## 4. Mechanism ablation

All controls use the same matched ORBIT configuration and seeds 600–609.

- **Identity:** ORBIT training/statistics path, but no functional preconditioning.
- **No-RoPE:** covariance conditioning remains, but relative-position rotation is removed.
- **Diagonal:** RoPE-aware per-frequency scaling remains, but off-diagonal coupling inside each 2x2 pair is removed.
- **Full ORBIT:** complete metric.

Paired effects of full ORBIT:

```text
vs identity   -0.008398   CI [-0.011634, -0.005163]   wins 9/10
vs no-RoPE    -0.003963   CI [-0.006060, -0.001867]   wins 10/10
vs diagonal   -0.001952   CI [-0.003850, -0.000054]   wins 8/10
```

Interpretation: generic functional Q/K conditioning contributes most; RoPE-aware transport adds a further consistent gain; full off-diagonal coupling adds a smaller effect.

## 5. Longer horizon

124M, 2700 steps, two paired seeds.

ORBIT remains below Muon, NorMuon, and ASTRO in both tested seeds.

Because n=2 is small, this is transfer evidence rather than a high-powered significance test.

## 6. Larger model

355M, 900 steps, two paired seeds, gradient checkpointing.

ORBIT is lower than Muon and NorMuon in both tested seeds. The ASTRO comparison is mixed: one seed favors each method.

The 355M model uses batch size 4 instead of 8, so this experiment is **not** a controlled scaling law.

## Reproducibility rule

All final records carry:
- seed
- optimizer
- frozen configuration
- task ID
- implementation digest
- environment versions
- runtime
- peak CUDA allocation
- validation loss
- training trajectory
- ORBIT diagnostics when applicable

The merge scripts reject stale implementation digests, foreign task IDs, missing expected tasks, and incomplete phase structure.
