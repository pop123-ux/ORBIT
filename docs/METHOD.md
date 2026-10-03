# Method

This document defines the canonical ORBIT optimizer used by the paper's primary 124M experiments.

## Forward RoPE score

For a query at position $i$ and a causal key at $j\le i$, define

```math
\Delta=i-j\ge0.
```

The model applies $R_f(i)$ to the query pair and $R_f(j)$ to the key pair. Since RoPE rotations are orthogonal,

```math
(R_f(i)q_{i,f})^\top(R_f(j)k_{j,f})
=
q_{i,f}^\top R_f(i)^\top R_f(j)k_{j,f}
=
q_{i,f}^\top R_f(-\Delta)k_{j,f}.
```

That identity describes the **forward attention computation**.

ORBIT does not claim to use the exact quadratic pullback of this score. Instead, it uses the same RoPE frequency structure to define an **optimizer-side transport geometry** in the inverse orientation, $R_f(+\Delta)$. This distinction is intentional and is part of the optimizer specification.

## ORBIT optimizer-side Q/K metric

ORBIT maintains the uncentered second moments

```math
S_{Q,f}=\mathbb E[q_fq_f^\top],
\qquad
S_{K,f}=\mathbb E[k_fk_f^\top].
```

For non-negative causal separations $\Delta=i-j$, the paper-trained optimizer defines

```math
M_{Q,f}
=
\mathbb E_\Delta
\left[
R_f(+\Delta)S_{K,f}R_f(+\Delta)^\top
\right],
```

```math
M_{K,f}
=
\mathbb E_\Delta
\left[
R_f(+\Delta)^\top S_{Q,f}R_f(+\Delta)
\right].
```

The relative-position expectation is approximated by a uniform average over the fixed log-spaced grid

```math
\mathcal D=\{1,2,4,8,16,32,64,128\}.
```

This grid is a design approximation. It is not an estimate of the empirical token-pair distance distribution.

The use of $R(+\Delta)$ should be read as an optimizer design choice: ORBIT transports the opposite-side second moment in the orientation inverse to the causal score rotation. The resulting matrices are RoPE-informed local preconditioners, not exact Fisher blocks, Hessians, natural-gradient metrics, or complete parameter-space pullbacks.

The matrices $M_{Q,f}$ and $M_{K,f}$ are output-coordinate factors. For example,

```math
\delta q_f=\delta W_{Q,f}x,
```

so an exact quadratic metric on $\delta W_{Q,f}$ would also contain statistics of the input activation $x$. ORBIT deliberately keeps only the frequency-local $2\times2$ output-side factor and applies it as a left preconditioner.

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

Thus the local quadratic geometry encoded by ORBIT's metric is whitened before the update is restored to the original joint Q/K step budget. The implementation evaluates the symmetric $2\times2$ inverse power analytically, adds $10^{-5}I$, and caps the effective condition number at 100.

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

the restoration fixes the aggregate Euclidean parameter-space magnitude of the Q/K step while leaving the ORBIT metric free to reshape the coupled direction.

## Compact update rule

For one paired Q/K projection, define

```math
\mathcal T_Q=\operatorname{blkdiag}_f(M_{Q,f}^{-1/2}),
\qquad
\mathcal T_K=\operatorname{blkdiag}_f(M_{K,f}^{-1/2}).
```

Then

```math
(\widetilde U_Q,\widetilde U_K)
=
\rho(\mathcal T_Q U_Q,\mathcal T_K U_K),
```

with $\rho$ given above, followed by

```math
W_{Q,t+1}=(1-\eta_t\lambda)W_{Q,t}-\eta_t\widetilde U_Q,
```

```math
W_{K,t+1}=(1-\eta_t\lambda)W_{K,t}-\eta_t\widetilde U_K.
```

Other hidden matrix parameters retain the ordinary Muon update, while embeddings, biases, normalization parameters, and the tied output head use the auxiliary AdamW path.

## Variants

```python
Orbit(model, variant="orbit")
Orbit(model, variant="orbit_norope")
Orbit(model, variant="orbit_diag")
Orbit(model, variant="orbit_identity")
```

- `orbit`: full $R(+\Delta)$ optimizer-side transported metric.
- `orbit_norope`: opposite-side second moment without rotary transport.
- `orbit_diag`: transported metric with off-diagonal coupling removed.
- `orbit_identity`: second-moment path retained, functional preconditioner disabled.

## Engineering semantics

The reusable package keeps the paper-trained optimizer geometry but hardens the surrounding state handling: second moments are persistent, DDP aggregates sufficient statistics before the EMA, and checkpoint recomputation does not update the EMA twice. These engineering changes do not alter the non-checkpointed 124M optimizer architecture evaluated in the paper.
