# Implementation map

This document maps the paper equations to the public implementation.

## 1. Base Muon candidate

Paper: the matrix gradient is momentum-smoothed and passed through a five-step quintic Newton–Schulz polar map.

Code:
- `src/orbit/optimizer.py::newton_schulz`
- `src/orbit/optimizer.py::Orbit._spectral_candidate`
- `src/orbit/baselines.py::Muon._spectral_candidate`

The same candidate rule is used by the matched Muon control and by ORBIT before Q/K conditioning.

## 2. Optimizer-only Q/K statistics

Paper: for every attention head and RoPE frequency pair, maintain 2x2 query and key covariances.

Code:
- `src/orbit/model.py::RotaryAttention._update_covariances`

The statistics are exponential moving averages with beta = 0.95. They are buffers, not model parameters, and are disabled for ordinary Muon.

## 3. RoPE transport

Paper:

```text
M_Q,f = E_delta [ R_f(delta) C_K,f R_f(delta)^T ]
M_K,f = E_delta [ R_f(delta)^T C_Q,f R_f(delta) ]
```

Code:
- `src/orbit/model.py::RotaryAttention.orbit_metrics`

The expectation is approximated over the fixed displacement grid:

```text
{1, 2, 4, 8, 16, 32, 64, 128}
```

## 4. Analytic 2x2 inverse metric

Paper: apply M^{-1/2} to each local update pair.

Code:
- `src/orbit/optimizer.py::inverse_metric_power`

The implementation uses the closed-form spectrum of a symmetric 2x2 matrix rather than a batched eigensolver. The metric is symmetrized, regularized, condition-capped, and locally falls back to identity for non-finite auxiliary statistics.

## 5. Q/K preconditioning

Code:
- `src/orbit/optimizer.py::Orbit._precondition_pair`

The Q and K Muon candidates are reshaped into:

```text
[head, frequency, 2, input_dimension]
```

and the local 2x2 transform acts on the two RoPE coordinates.

## 6. Joint Frobenius restoration

After preconditioning, ORBIT computes one scalar restore factor from the combined Q/K Frobenius norm and multiplies both transformed updates by it.

This is the crucial control that keeps the total Q/K step budget equal to the original Muon candidate. ORBIT changes update direction/geometry rather than simply increasing update magnitude.

## 7. Other parameters

- Q/K: Muon candidate + ORBIT preconditioner
- other hidden 2D matrices: standard Muon
- token embeddings / tied LM head: auxiliary AdamW with decay
- vectors, biases, norms: auxiliary AdamW without decay

## 8. Inference

Nothing from the optimizer is required at inference:
- covariance buffers are training-only;
- no new model parameters are introduced;
- the attention forward pass remains standard RoPE attention.

Therefore ORBIT has zero inference-time graph overhead.
