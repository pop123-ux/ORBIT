# Method

ORBIT modifies only the query/key part of a Muon step.

For a single attention head and one two-dimensional RoPE frequency pair, the pre-softmax score contribution is

[
s_{ij,f}=q_{i,f}^{	op}R_f(i-j)k_{j,f}.
]

The optimizer tracks exponential-moving-average second moments of the unrotated Q/K pair:

[
C_{Q,f}=mathbb E[q_f q_f^	op],qquad
C_{K,f}=mathbb E[k_f k_f^	op].
]

For a set of relative displacements (mathcal D), the local metrics are

[
M_{Q,f}=rac{1}{|mathcal D|}sum_{Deltainmathcal D}
R_f(Delta)C_{K,f}R_f(Delta)^	op,
]

[
M_{K,f}=rac{1}{|mathcal D|}sum_{Deltainmathcal D}
R_f(Delta)^	op C_{Q,f}R_f(Delta).
]

The default displacement set is

[
mathcal D={1,2,4,8,16,32,64,128}.
]

Muon first produces the candidate matrix updates (U_Q) and (U_K). ORBIT reshapes them into RoPE pairs and applies

[
widehat U_Q=M_Q^{-1/2}U_Q,qquad
widehat U_K=M_K^{-1/2}U_K.
]

The final step uses one shared restoration factor

[
ho=
sqrt{
rac{|U_Q|_F^2+|U_K|_F^2}
{|widehat U_Q|_F^2+|widehat U_K|_F^2}
},
]

so that

[
widetilde U_Q=howidehat U_Q,qquad
widetilde U_K=howidehat U_K.
]

This keeps the combined Q/K update norm fixed. The preconditioner changes the direction and allocation of the step rather than increasing its total magnitude.

## Numerical details

The local metrics are symmetric 2x2 matrices. `inverse_metric_power` computes the inverse metric power from the closed-form spectrum, adds a small diagonal regularizer, caps the effective condition number, and falls back to the identity for non-finite auxiliary statistics.

Default values:

```text
momentum              0.95
Newton-Schulz steps   5
Q/K covariance EMA    0.95
metric epsilon        1e-5
condition cap         100
functional power      1.0
```

## Parameter routing

Q/K matrices use the Muon candidate followed by ORBIT preconditioning. Other hidden two-dimensional matrices use the same Muon candidate without ORBIT. Token embeddings and the tied LM head use the auxiliary AdamW path with decay; vectors, biases, and normalization parameters use AdamW without decay.

The covariance buffers are training-only. No ORBIT state is needed for inference.
