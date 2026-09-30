# ORBIT-v0 method freeze

**Status:** research candidate, not a novelty or performance claim.

ORBIT is an optimizer-only extension for rotary attention. It keeps ordinary Q/K/V/O
weights and changes no inference computation. The first frozen candidate tests one
specific hypothesis: Q and K should be updated according to the *functional movement
of RoPE attention logits*, not only according to parameter-space norms.

For one head and one two-dimensional RoPE frequency pair, the score contribution is

\[
s_{ij,f}=q_{i,f}^{\top}R_f(i-j)k_{j,f}.
\]

Let `C_q` and `C_k` be EMA 2x2 covariances of the unrotated query/key pair. ORBIT
forms the local output-space metrics

\[
M_{Q,f}=\mathbb E_{\Delta}[R_f(\Delta)C_{K,f}R_f(\Delta)^\top],\qquad
M_{K,f}=\mathbb E_{\Delta}[R_f(\Delta)^\top C_{Q,f}R_f(\Delta)].
\]

The expectation is approximated with a fixed logarithmic relative-position grid
`{1,2,4,8,16,32,64,128}`. A standard Muon candidate direction is produced first.
Each two-row Q/K frequency block is then left-preconditioned by
`M^{-p/2}`. The combined Q+K Frobenius norm is restored afterwards. Thus ORBIT-v0
cannot win merely by taking a larger aggregate Q/K step.

The initial paper candidate uses `p=1` unless an ablation falsifies it. Hyperparameter
search must **not** give ORBIT more free tuning dimensions than the baselines; the
functional exponent and relative-position grid are mechanism choices, not hidden
search knobs.

## Required falsification controls

- `orbit_identity`: same code/routing but no functional preconditioner; this is the
  direct Muon-style control.
- `orbit_norope`: use Q/K covariances but remove the relative-position rotations.
- `orbit_diag`: retain per-frequency sensitivity but remove off-diagonal 2x2 phase
  coupling.
- tuned AdamW, Muon, NorMuon and faithful AdaMuon reference from the inherited
  paper campaign.

## Kill criteria

Do not promote ORBIT to the expensive campaign unless all are true:

1. no NaN/Inf failures in the smoke and 124M discovery runs;
2. median discovery loss is not worse than tuned Muon at comparable runtime;
3. at least one genuine ORBIT mechanism (`orbit` vs `identity/norope/diag`) has a
   repeatable effect larger than run-to-run noise;
4. held-out seeds retain the direction of the discovery result;
5. overhead is plausibly production-compatible (target <15% wall-clock vs Muon);
6. the exact frozen equation survives a final prior-art audit before paper claims.

## Production path

The research model is GPT-2-sized but uses RoPE because the proposed geometry is
undefined for learned absolute position embeddings. The optimizer keeps standard
weights, so there is zero inference overhead. After 124M confirmation, the required
transfer target is a modern RoPE decoder with GQA; QK-norm support is a later gate.
