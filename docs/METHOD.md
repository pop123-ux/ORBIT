# Method

ORBIT modifies only the query/key part of a Muon step while leaving the model architecture unchanged.

## Rotary query-key interaction

For one attention head and one two-dimensional RoPE frequency pair \(f\), the pre-softmax score contribution is

$$
s_{ij,f}
=
q_{i,f}^{\top}
R_f(i-j)
k_{j,f}.
$$

Writing the relative displacement as

$$
\Delta=i-j,
$$

the score becomes

$$
s_f=q_f^\top R_f(\Delta)k_f.
$$

A small perturbation \(\delta q_f\) changes the score by

$$
\delta s_f
=
\delta q_f^\top R_f(\Delta)k_f.
$$

The squared functional movement induced by that query perturbation is

$$
(\delta s_f)^2
=
\delta q_f^\top
R_f(\Delta)
k_fk_f^\top
R_f(\Delta)^\top
\delta q_f.
$$

Taking an expectation over keys motivates the local query metric

$$
M_{Q,f}
=
\mathbb E_{\Delta}
\left[
R_f(\Delta)
C_{K,f}
R_f(\Delta)^\top
\right],
$$

where

$$
C_{K,f}=\mathbb E[k_fk_f^\top].
$$

The key-side expression follows symmetrically:

$$
M_{K,f}
=
\mathbb E_{\Delta}
\left[
R_f(\Delta)^\top
C_{Q,f}
R_f(\Delta)
\right],
$$

with

$$
C_{Q,f}=\mathbb E[q_fq_f^\top].
$$

This is why ORBIT conditions Query using Key statistics and Key using Query statistics.

## Running statistics

The implementation maintains exponential-moving-average covariance matrices for the unrotated Q/K pairs:

$$
C_t
=
\beta C_{t-1}
+
(1-\beta)C_{\mathrm{batch}},
$$

with default

$$
\beta=0.95.
$$

RoPE acts on coordinate pairs, so the required statistics are only \(2\times2\) matrices for each head and frequency pair.

The relative-position expectation is approximated over

$$
\mathcal D
=
\{1,2,4,8,16,32,64,128\}.
$$

Thus

$$
M_{Q,f}
=
\frac{1}{|\mathcal D|}
\sum_{\Delta\in\mathcal D}
R_f(\Delta)
C_{K,f}
R_f(\Delta)^\top,
$$

$$
M_{K,f}
=
\frac{1}{|\mathcal D|}
\sum_{\Delta\in\mathcal D}
R_f(\Delta)^\top
C_{Q,f}
R_f(\Delta).
$$

## Muon candidate update

ORBIT does not replace the matrix update used by Muon.

For a matrix gradient \(G_t\), the implementation first forms a momentum-smoothed look-ahead direction and applies the same five-step quintic Newton-Schulz polar iteration used by the Muon path.

Conceptually, if

$$
G=U\Sigma V^\top,
$$

the polar direction is related to

$$
UV^\top,
$$

which reduces the dominance of singular-value magnitude in the raw gradient.

The resulting query/key candidates are denoted

$$
U_Q,\qquad U_K.
$$

## Function-space preconditioning

For the full ORBIT variant, each two-row RoPE update pair is transformed using

$$
\widehat U_Q=M_Q^{-p/2}U_Q,
\qquad
\widehat U_K=M_K^{-p/2}U_K,
$$

with default functional power

$$
p=1.
$$

The default method therefore uses the inverse square root \(M^{-1/2}\). Directions with large metric eigenvalues correspond to larger functional sensitivity and are attenuated more strongly.

## Joint Frobenius restoration

Metric preconditioning can change the total size of the Q/K update. To prevent ORBIT from gaining simply by taking a larger step, a single restore factor is computed from the pair:

$$
\rho
=
\sqrt{
\frac{
\lVert U_Q\rVert_F^2+
\lVert U_K\rVert_F^2
}{
\lVert \widehat U_Q\rVert_F^2+
\lVert \widehat U_K\rVert_F^2
}
}.
$$

The final updates are

$$
\widetilde U_Q=\rho\widehat U_Q,
\qquad
\widetilde U_K=\rho\widehat U_K.
$$

Hence

$$
\lVert \widetilde U_Q\rVert_F^2+
\lVert \widetilde U_K\rVert_F^2
=
\lVert U_Q\rVert_F^2+
\lVert U_K\rVert_F^2.
$$

ORBIT changes the geometry of the update while keeping the combined Q/K step budget fixed.

## Numerical implementation

Each local metric is a symmetric \(2\times2\) matrix. `inverse_metric_power` computes the inverse power from its closed-form spectrum instead of calling a general batched eigensolver.

Before applying the matrix power, the implementation:

- symmetrizes the metric;
- applies the configured numerical floor;
- caps the effective condition number;
- handles nearly repeated eigenvalues with an isotropic branch;
- falls back to identity preconditioning for non-finite auxiliary statistics.

Default numerical settings:

~~~text
momentum              0.95
Newton-Schulz steps   5
Q/K covariance EMA    0.95
metric epsilon        1e-5
condition cap         100
functional power      1.0
~~~

## Parameter routing

| Parameter class | Optimizer path |
| --- | --- |
| Query / Key matrices | Muon candidate + ORBIT preconditioner |
| Other hidden 2-D matrices | Muon |
| Token embedding / tied LM head | AdamW with decay |
| Vectors, biases, norms | AdamW without decay |

The Q/K covariance buffers are training-only. They introduce no new learned parameter and are not required at inference.

## Ablation variants

The public implementation includes four variants:

~~~python
Orbit(model, variant="orbit")
Orbit(model, variant="orbit_norope")
Orbit(model, variant="orbit_diag")
Orbit(model, variant="orbit_identity")
~~~

- `orbit`: full RoPE-transported \(2\times2\) metric.
- `orbit_norope`: uses Q/K covariances without relative-position transport.
- `orbit_diag`: removes off-diagonal coupling from the local metric.
- `orbit_identity`: keeps the statistics path but applies no functional preconditioner.

These variants are intended for mechanism isolation rather than as separate production optimizers.
