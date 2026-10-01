# Architecture and code map

This document maps the main ORBIT operations to the public implementation.

## Package layout

| File | Responsibility |
| --- | --- |
| `src/orbit/optimizer.py` | Muon candidate, \(2\times2\) inverse metric, ORBIT preconditioning |
| `src/orbit/baselines.py` | matched Muon control |
| `src/orbit/model.py` | RoPE GPT reference model and Q/K statistics |
| `examples/quickstart.py` | minimal executable training example |
| `tests/test_orbit.py` | metric, statistics, optimizer-step and ablation tests |
| `tests/test_muon_control.py` | baseline-control tests |

## Muon base update

`optimizer.py::newton_schulz` implements the five-step quintic polar iteration.

`Orbit._spectral_candidate` applies momentum, look-ahead, Newton-Schulz orthogonalization, and the same aspect-ratio scaling used by the matched Muon control.

ORBIT therefore modifies the Q/K candidate **after** the generic spectral matrix step has been constructed.

## Q/K statistics

`model.py::RotaryAttention._update_covariances` maintains one query covariance and one key covariance for every

~~~text
[attention head, RoPE frequency pair]
~~~

location.

Each covariance is \(2\times2\). Statistics are updated only during training and only when ORBIT enables collection.

`RotaryAttention.orbit_metrics` applies the configured relative-position rotations and returns the query-side and key-side metrics.

## Inverse metric

`optimizer.py::inverse_metric_power` computes

$$
M^{-p/2}
$$

for a batch of symmetric \(2\times2\) matrices.

The implementation uses the analytic two-dimensional spectrum rather than `torch.linalg.eigh`. It also performs scale normalization, condition-number clipping, a repeated-eigenvalue branch, and a local identity fallback for invalid auxiliary statistics.

## Q/K preconditioning

`Orbit._precondition_pair` reshapes each matrix candidate to

~~~text
[head, frequency, 2, input_dimension]
~~~

so the local metric acts directly on the two RoPE coordinates.

After applying the transforms, one scalar restores the combined Q/K Frobenius norm.

## Parameter routing

During initialization, `Orbit` assigns each parameter to one of five internal roles:

~~~text
q
k
spectral
aux_decay
aux_nodecay
~~~

The `q`, `k`, and `spectral` paths use the Muon-style matrix candidate. Only `q` and `k` then enter `_precondition_pair`.

The auxiliary paths implement AdamW-style updates for parameters that are not routed through the spectral matrix update.

## Model interface

The reference `OrbitGPT` exposes:

~~~python
model.orbit_qk_pairs()
model.set_orbit_stat_collection(enabled)
~~~

Each item returned by `orbit_qk_pairs()` provides:

~~~python
{
    "q": q_parameter,
    "k": k_parameter,
    "module": attention_module,
}
~~~

and the attention module must provide:

~~~python
attention_module.orbit_metrics(deltas, rotate=True, eps=...)
~~~

The current optimizer also identifies `model.wte.weight` and `model.lm_head.weight` as the embedding/head parameters assigned to the auxiliary decay path.

## Optimizer step

At a high level, one ORBIT step is:

~~~text
gradient
  │
  ├─ hidden 2-D matrix ──> Muon candidate
  │                          │
  │                          ├─ Q/K ──> ORBIT metric transform ──> joint norm restore
  │                          └─ other matrix ────────────────────> update
  │
  └─ vector / embedding / head ──> auxiliary AdamW
~~~

The Q/K metric path is local to each attention layer. No full-model curvature matrix is formed.

## Inference boundary

ORBIT changes only the optimizer and training-time statistics.

The following are not required after training:

- covariance buffers;
- metric construction;
- inverse metric transforms;
- joint restore factors;
- optimizer diagnostics.

The deployed model retains the standard RoPE attention forward pass.
