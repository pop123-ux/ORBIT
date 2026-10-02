# Architecture

This document describes only the interfaces and state that are specific to ORBIT.

## Code map

| File | Responsibility |
| --- | --- |
| `src/orbit/model.py` | RoPE attention, Q/K covariance state, causal metric construction |
| `src/orbit/optimizer.py` | Muon candidate, $2\times2$ inverse metric, Q/K preconditioning |
| `src/orbit/baselines.py` | matched Muon control |
| `tests/test_correctness.py` | causal-sign, checkpoint, DDP, scheduler and state-resume regressions |

## Model contract

`Orbit` expects the model to expose

```python
model.orbit_qk_pairs()
model.set_orbit_stat_collection(enabled)
```

Each item returned by `orbit_qk_pairs()` contains

```python
{
    "q": q_parameter,
    "k": k_parameter,
    "module": attention_module,
}
```

and the attention module provides

```python
attention_module.orbit_metrics(deltas, rotate=True, eps=...)
```

The reference `OrbitGPT` also exposes `wte` and `lm_head` so the auxiliary path can identify the embedding/head parameters explicitly rather than by tensor shape.

## Relative-position convention

`RotaryAttention._apply_rope` rotates the query at $i$ by $R(i)$ and the key at $j$ by $R(j)$. Their score therefore contains

```math
R(i)^\top R(j)=R(j-i).
```

ORBIT defines `delta = i - j >= 0` and constructs `R(-delta)` through `RotaryAttention._causal_relative_rotation`. The same helper is used by the metric path, which prevents the documentation and implementation from adopting opposite sign conventions.

## Statistics state

Each attention layer stores

```text
orbit_q_cov      [n_head, n_freq, 2, 2]
orbit_k_cov      [n_head, n_freq, 2, 2]
orbit_stats_seen scalar
```

All three are persistent buffers and therefore part of `model.state_dict()`.

### Gradient checkpointing

The reference model uses non-reentrant checkpointing. The original forward is allowed to update the statistics; the recomputation context calls `suspend_orbit_stat_collection()`. This keeps the EMA update count independent of whether checkpointing is enabled.

### Distributed data parallel

`_update_covariances` forms unnormalized Q/K second-moment sums and a sample count. When `torch.distributed` is initialized, these sufficient statistics are reduced across ranks before normalization and before the EMA. Consequently each rank holds the same covariance state before the optimizer step.

## Preconditioning path

`Orbit._spectral_candidate` constructs the matrix candidate. For Q/K pairs, `Orbit._precondition_pair` then:

1. requests $M_Q,M_K$ from the corresponding attention module;
2. optionally diagonalizes the metrics for `orbit_diag`;
3. computes $M^{-p/2}$ with `inverse_metric_power`;
4. reshapes each candidate to `[head, frequency, 2, input_dimension]`;
5. left-multiplies each 2-row RoPE pair by its local transform;
6. applies one shared Q/K Frobenius restore factor.

Other hidden 2-D matrices remain on the matched spectral path.

## Learning-rate coupling

The constructor accepts

```python
Orbit(model, lr=matrix_lr, adamw_lr=aux_lr)
```

and stores the ratio

```math
r_{\mathrm{aux}}=\frac{\mathrm{aux\_lr}}{\mathrm{matrix\_lr}}.
```

At each step the auxiliary learning rate is computed as

```math
\eta_{\mathrm{aux}}=\eta_{\mathrm{matrix}}r_{\mathrm{aux}}.
```

A standard PyTorch scheduler therefore scales both paths through the single scheduled `lr` field.

## Diagnostics

The optimizer accumulates diagnostic scalars as device tensors during `step()`. Device-to-host conversion occurs only when

```python
optimizer.diagnostics()
```

is called.
