# Method

This document follows the paper’s mathematical convention exactly.

## Causal RoPE score

For a query at position $i$ and a causal key at $j\le i$, define

```math
\Delta=i-j\ge0.
```

The implementation applies $R_f(i)$ to the query pair and $R_f(j)$ to the key pair. Since RoPE rotations are orthogonal,

```math
(R_f(i)q_{i,f})^\top(R_f(j)k_{j,f})
=
q_{i,f}^\top R_f(i)^\top R_f(j)k_{j,f}
=
q_{i,f}^\top R_f(-\Delta)k_{j,f}.
```

For a perturbation $\delta q_f$,

```math
\delta s_f
=
\delta q_f^\top R_f(-\Delta)k_f.
```

Therefore

```math
\mathbb E[(\delta s_f)^2]
=
\delta q_f^\top
\mathbb E\!\left[
R_f(-\Delta)S_{K,f}R_f(-\Delta)^\top
\right]
\delta q_f,
```

where

```math
S_{K,f}=\mathbb E[k_fk_f^\top].
```

The symmetric key-side expression follows from perturbing $k_f$.

## Local Q/K metric

ORBIT maintains the uncentered second moments

```math
S_{Q,f}=\mathbb E[q_fq_f^\top],
\qquad
S_{K,f}=\mathbb E[k_fk_f^\top],
```

and defines

```math
M_{Q,f}
=
\mathbb E_\Delta
\left[
R_f(-\Delta)S_{K,f}R_f(-\Delta)^\top
\right],
```

```math
M_{K,f}
=
\mathbb E_\Delta
\left[
R_f(-\Delta)^\top S_{Q,f}R_f(-\Delta)
\right].
```

The relative-position expectation is approximated by a uniform average over the fixed log-spaced causal-distance grid

```math
\mathcal D=\{1,2,4,8,16,32,64,128\}.
```

This grid is a design approximation. It is not an estimate of the empirical token-pair distance distribution.

The second moments use an EMA,

```math
S_t
=
\beta S_{t-1}
+
(1-\beta)S_{\mathrm{batch}},
\qquad
\beta=0.95,
```

with the first observation used directly.

## Inverse-square-root preconditioning

Let $U_Q,U_K$ be the Muon candidate updates. ORBIT acts on each two-row RoPE pair:

```math
\widehat U_Q=M_Q^{-p/2}U_Q,
\qquad
\widehat U_K=M_K^{-p/2}U_K.
```

The default $p=1$ gives $M^{-1/2}$.

For the regularized effective metric

```math
M=V\Lambda V^\top,
```

the transform is

```math
T=M^{-1/2}=V\Lambda^{-1/2}V^\top.
```

It satisfies

```math
T^\top M T=I.
```

Thus the local quadratic metric is whitened: directions with larger functional sensitivity are attenuated more strongly. This metric is local to the pre-softmax rotary Q/K interaction; it is not a full Fisher matrix, Hessian, or exact natural-gradient metric.

The implementation evaluates the symmetric $2\times2$ inverse power analytically, adds $10^{-5}I$, and caps the effective condition number at 100.

## Joint Frobenius restoration

Preconditioning changes both geometry and magnitude. ORBIT uses one shared factor

```math
\rho
=
\sqrt{
\frac{
\lVert U_Q\rVert_F^2+\lVert U_K\rVert_F^2
}{
\lVert\widehat U_Q\rVert_F^2+\lVert\widehat U_K\rVert_F^2
}
},
```

then

```math
\widetilde U_Q=\rho\widehat U_Q,
\qquad
\widetilde U_K=\rho\widehat U_K.
```

By construction,

```math
\lVert\widetilde U_Q\rVert_F^2+
\lVert\widetilde U_K\rVert_F^2
=
\lVert U_Q\rVert_F^2+
\lVert U_K\rVert_F^2.
```

Since

```math
\lVert U\rVert_F=\lVert\operatorname{vec}(U)\rVert_2,
```

the restoration fixes the aggregate Euclidean parameter-space magnitude of the Q/K step while leaving the metric free to reshape the coupled update.

## Variants

```python
Orbit(model, variant="orbit")
Orbit(model, variant="orbit_norope")
Orbit(model, variant="orbit_diag")
Orbit(model, variant="orbit_identity")
```

- `orbit`: full causal RoPE-transported metric.
- `orbit_norope`: opposite-side second moment without RoPE transport.
- `orbit_diag`: transported metric with off-diagonal coupling removed.
- `orbit_identity`: second-moment path retained, functional preconditioner disabled.
