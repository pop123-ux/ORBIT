# Architecture

This document covers the ORBIT-specific model/optimizer interface and state.

## Code map

| File | Responsibility |
| --- | --- |
| `src/orbit/model.py` | RoPE attention, Q/K second-moment state, causal metric construction |
| `src/orbit/optimizer.py` | Muon candidate, analytic $2\times2$ inverse metric, Q/K preconditioning |
| `src/orbit/baselines.py` | matched Muon control used by repository tests |
| `tests/test_correctness.py` | paper-aligned correctness regressions |

The canonical paper implementation and this repository use the same `model.py` and `optimizer.py` source.

## Model contract

`Orbit` expects

```python
model.orbit_qk_pairs()
model.set_orbit_stat_collection(enabled)
```

Each Q/K pair provides

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

The reference `OrbitGPT` exposes `wte` and `lm_head` so the auxiliary path can identify the embedding/head parameters explicitly.

## Relative-position convention

`RotaryAttention._apply_rope` rotates the query at $i$ by $R(i)$ and the key at $j$ by $R(j)$:

```math
R(i)^\top R(j)=R(j-i).
```

ORBIT defines `delta = i - j >= 0`, so `RotaryAttention._causal_relative_rotation(delta)` constructs $R(-\delta)$. The score-gradient and metric paths share this convention, and the regression suite checks it independently with autograd.

## Second-moment state

Each attention layer stores

```text
orbit_q_second_moment  [n_head, n_freq, 2, 2]
orbit_k_second_moment  [n_head, n_freq, 2, 2]
orbit_stats_seen       scalar
```

These are persistent buffers.

### Gradient checkpointing

The reference model uses non-reentrant checkpointing. The original forward updates the second moments; the recomputation context calls `suspend_orbit_stat_collection()`. This keeps one second-moment update per logical training step.

### Distributed training

`_update_second_moments` forms unnormalized Q/K second-moment sums and a sample count. When `torch.distributed` is initialized, those sufficient statistics are reduced across ranks before normalization and before the EMA. Every replica therefore constructs the same ORBIT metric.

## Q/K update path

`Orbit._spectral_candidate` first constructs the Muon matrix candidate. For each Q/K pair, `Orbit._precondition_pair` then:

1. requests $M_Q,M_K$ from the attention module;
2. optionally diagonalizes them for `orbit_diag`;
3. computes $M^{-p/2}$ with `inverse_metric_power`;
4. reshapes each candidate to `[head, frequency, 2, input_dimension]`;
5. left-multiplies each two-row RoPE pair by the local transform;
6. applies one shared Q/K Frobenius restoration factor.

Other hidden two-dimensional matrices stay on the matched spectral path.

## Learning-rate coupling

The constructor accepts

```python
Orbit(model, lr=matrix_lr, adamw_lr=aux_lr)
```

and stores

```math
r_{\mathrm{aux}}
=
\frac{\eta_{\mathrm{aux},0}}
     {\eta_{\mathrm{matrix},0}}.
```

The effective auxiliary rate is

```math
\eta_{\mathrm{aux},t}
=
r_{\mathrm{aux}}\eta_{\mathrm{matrix},t}.
```

A standard scheduler therefore needs to update only `lr`.

## Diagnostics

Metric-condition and restoration diagnostics remain device tensors during `step()`. Device-to-host conversion occurs when

```python
optimizer.diagnostics()
```

is called.
