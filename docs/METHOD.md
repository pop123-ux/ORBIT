# Method

## Causal RoPE geometry

For a query at position $i$ attending to a causal key at position $j\le i$, define the non-negative distance

```math
\Delta=i-j.
```

The implementation rotates each unrotated pair by its absolute token position. Therefore

```math
(R(i)q_i)^\top(R(j)k_j)
=
q_i^\top R(i)^\top R(j)k_j
=
q_i^\top R(j-i)k_j
=
q_i^\top R(-\Delta)k_j.
```

For one RoPE frequency pair $f$,

```math
s_{ij,f}
=
q_{i,f}^{\top}R_f(-\Delta)k_{j,f}.
```

A perturbation $\delta q_f$ gives

```math
\delta s_f
=
\delta q_f^\top R_f(-\Delta)k_f,
```

so

```math
\mathbb E[(\delta s_f)^2]
=
\delta q_f^\top
\mathbb E\!\left[
R_f(-\Delta)C_{K,f}R_f(-\Delta)^\top
\right]
\delta q_f.
```

This yields

```math
M_{Q,f}
=
\mathbb E_{\Delta}
\left[
R_f(-\Delta)C_{K,f}R_f(-\Delta)^\top
\right],
```

and symmetrically

```math
M_{K,f}
=
\mathbb E_{\Delta}
\left[
R_f(-\Delta)^\top C_{Q,f}R_f(-\Delta)
\right].
```

The implementation approximates the relative-position expectation over

```math
\mathcal D=\{1,2,4,8,16,32,64,128\}.
```

Negative values are rejected by the API because `deltas` are distances $i-j$, not signed offsets.

## Q/K statistics

For each head and RoPE frequency pair, ORBIT maintains unrotated $2\times2$ covariances

```math
C_{Q,f}=\mathbb E[q_fq_f^\top],
\qquad
C_{K,f}=\mathbb E[k_fk_f^\top].
```

The running update is

```math
C_t
=
\beta C_{t-1}
+
(1-\beta)C_{\mathrm{batch}},
\qquad
\beta=0.95,
```

with the first observed batch used directly.

The sufficient statistics are globally reduced before the EMA when `torch.distributed` is initialized, so every DDP replica uses the same metric. Gradient-checkpoint recomputation explicitly suspends statistics collection; one logical forward/backward contributes one covariance update, not two.

The covariance buffers and observation counter are persistent entries in the model state dict, so checkpoint/resume preserves the EMA exactly.

## Preconditioning

Muon first provides candidate Q/K matrix updates $U_Q,U_K$. ORBIT reshapes each candidate to

```text
[head, frequency, 2, input_dimension]
```

and applies

```math
\widehat U_Q=M_Q^{-p/2}U_Q,
\qquad
\widehat U_K=M_K^{-p/2}U_K.
```

The default $p=1$ gives $M^{-1/2}$. For a symmetric positive-definite metric

```math
M=V\Lambda V^\top,
```

the transform is

```math
M^{-1/2}=V\Lambda^{-1/2}V^\top.
```

It whitens the local quadratic metric:

```math
(M^{-1/2}u)^\top M(M^{-1/2}u)
=
u^\top u.
```

Thus directions with larger functional sensitivity are attenuated more strongly while preserving the geometry of the candidate in the whitened coordinates.

## Joint Q/K norm restoration

Preconditioning changes both direction and magnitude. ORBIT isolates the geometric change by using one shared restoration factor

```math
\rho
=
\sqrt{
\frac{
\lVert U_Q\rVert_F^2+\lVert U_K\rVert_F^2
}{
\lVert\widehat U_Q\rVert_F^2+\lVert\widehat U_K\rVert_F^2
}
}.
```

The final updates are

```math
\widetilde U_Q=\rho\widehat U_Q,
\qquad
\widetilde U_K=\rho\widehat U_K,
```

which guarantees

```math
\lVert\widetilde U_Q\rVert_F^2+
\lVert\widetilde U_K\rVert_F^2
=
\lVert U_Q\rVert_F^2+
\lVert U_K\rVert_F^2.
```

A shared factor preserves ORBIT's ability to redistribute the Q/K budget between the two coupled projections.

## Numerical $2\times2$ inverse power

`inverse_metric_power` uses the analytic spectrum of a symmetric $2\times2$ matrix rather than a general batched eigensolver. It symmetrizes the input, scale-normalizes it, floors the spectrum, caps the effective condition number, handles repeated eigenvalues isotropically, and falls back to identity for a non-finite local metric.

Default numerical settings:

```text
functional power      1.0
metric epsilon        1e-5
condition cap         100
Q/K covariance EMA    0.95
RoPE distances        1,2,4,8,16,32,64,128
```

## Ablation variants

```python
Orbit(model, variant="orbit")
Orbit(model, variant="orbit_norope")
Orbit(model, variant="orbit_diag")
Orbit(model, variant="orbit_identity")
```

- `orbit`: full causal RoPE-transported metric.
- `orbit_norope`: opposite-side covariance without RoPE transport.
- `orbit_diag`: diagonalized transported metric.
- `orbit_identity`: statistics path retained, functional preconditioner disabled.
