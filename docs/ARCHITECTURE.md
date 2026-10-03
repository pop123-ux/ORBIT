# Architecture

This document covers the ORBIT-specific model/optimizer interface and state.

## Code map

| File | Responsibility |
| --- | --- |
| `src/orbit/model.py` | RoPE attention, Q/K second-moment state, optimizer-side rotary metric construction |
| `src/orbit/optimizer.py` | Muon candidate, analytic $2\times2$ inverse metric, Q/K preconditioning |
| `src/orbit/baselines.py` | matched Muon control used by repository tests |
| `tests/test_correctness.py` | paper-aligned correctness regressions |

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

## Forward vs optimizer rotary convention

`RotaryAttention._apply_rope` rotates the query at $i$ by $R(i)$ and the key at $j$ by $R(j)$:

```math
R(i)^\top R(j)=R(j-i).
```

With causal separation $\Delta=i-j\ge0$, the forward score therefore contains $R(-\Delta)$.

ORBIT deliberately uses the inverse orientation for its **optimizer-side transport**. `RotaryAttention._optimizer_transport_rotation(delta)` constructs $R(+\Delta)$, and `orbit_metrics` uses

```math
M_Q=\mathbb E[R(+\Delta)S_KR(+\Delta)^\top],
```

```math
M_K=\mathbb E[R(+\Delta)^\top S_QR(+\Delta)].
```

The regression suite checks the two conventions separately: autograd verifies the forward $R(-\Delta)$ score, while the metric test verifies ORBIT's $R(+\Delta)$ optimizer transport.

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

This is engineering hardening for downstream use. The primary 124M paper experiments did not use gradient checkpointing, so it does not change the optimizer geometry evaluated there.

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

This is an output-side left preconditioner. ORBIT does not construct the full parameter-space pullback through the Q/K input activation.

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
