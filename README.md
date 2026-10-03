# ORBIT

[![CI](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml/badge.svg)](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![License](https://img.shields.io/badge/license-MIT-black)

**Function-Space Optimization for Rotary Query-Key Interactions**

ORBIT modifies the Muon candidate update for rotary Query/Key projections using a compact optimizer-side geometry built from RoPE frequency structure and opposite-side Q/K second moments. The canonical implementation in this repository matches the optimizer architecture used for the paper's primary 124M experiments: non-negative causal separations are transported with the optimizer orientation $R(+\Delta)$, followed by inverse-square-root preconditioning and joint Q/K Frobenius restoration.

[Method](docs/METHOD.md) · [Architecture](docs/ARCHITECTURE.md) · [Training](docs/TRAINING.md) · [Experiment status](docs/EXPERIMENTS.md)

## Definition

For a query at position $i$ attending to a causal key at $j\le i$, define

```math
\Delta=i-j\ge0.
```

The forward RoPE attention interaction is

```math
(R_f(i)q_{i,f})^\top(R_f(j)k_{j,f})
=
q_{i,f}^\top R_f(-\Delta)k_{j,f}.
```

ORBIT deliberately uses the inverse orientation as an **optimizer-side rotary transport**. Let the unrotated per-frequency uncentered second moments be

```math
S_{Q,f}=\mathbb E[q_fq_f^\top],
\qquad
S_{K,f}=\mathbb E[k_fk_f^\top].
```

The paper-trained ORBIT metric is

```math
M_{Q,f}
=
\mathbb E_{\Delta}
\left[
R_f(+\Delta)S_{K,f}R_f(+\Delta)^\top
\right],
```

```math
M_{K,f}
=
\mathbb E_{\Delta}
\left[
R_f(+\Delta)^\top S_{Q,f}R_f(+\Delta)
\right].
```

The expectation is approximated by a uniform average over

```math
\mathcal D=\{1,2,4,8,16,32,64,128\}.
```

This transport is a designed optimization geometry, not an exact Fisher/Hessian or exact pullback of the causal attention score. It uses the known RoPE coordinate structure and opposite-side activation statistics to reshape the Q/K update.

Muon produces candidates $U_Q,U_K$. ORBIT applies

```math
\widehat U_Q=M_Q^{-1/2}U_Q,
\qquad
\widehat U_K=M_K^{-1/2}U_K,
```

then uses one shared scalar $\rho$ so that

```math
\lVert\widetilde U_Q\rVert_F^2+
\lVert\widetilde U_K\rVert_F^2
=
\lVert U_Q\rVert_F^2+
\lVert U_K\rVert_F^2.
```

The result is a direction-changing Q/K preconditioner with the same aggregate Q/K step budget as the underlying Muon candidates. See [`docs/METHOD.md`](docs/METHOD.md) for the full definition.

## Install

```bash
git clone https://github.com/pop123-ux/ORBIT.git
cd ORBIT
pip install -e ".[dev]"
pytest -q
```

## Minimal use

```python
import torch
from orbit import Orbit, OrbitGPT, OrbitGPTConfig

model = OrbitGPT(
    OrbitGPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=512,
    )
)

optimizer = Orbit(
    model,
    lr=0.02,
    adamw_lr=3e-4,
    weight_decay=0.05,
)

x = torch.randint(0, model.config.vocab_size, (2, 512))
loss = model(x, labels=x).loss
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

The included `OrbitGPT` is the reference model interface. Integrating a different RoPE model requires exposing its Q/K pairs and the modules that construct the ORBIT metrics; see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Correctness contract

The regression suite fixes the implementation choices that are easy to make ambiguous:

- `deltas` are non-negative causal distances $\Delta=i-j$;
- forward attention uses $R(-\Delta)$ while ORBIT's optimizer-side transport uses $R(+\Delta)$;
- the $2\times2$ inverse metric is checked against a float64 eigendecomposition over multiple scales and powers;
- the joint Q/K Frobenius budget is tested directly;
- `orbit_identity` is checked against the matched Muon control;
- checkpoint recomputation contributes no duplicate second-moment update in the reusable reference implementation;
- DDP ranks aggregate sufficient statistics before the EMA;
- second-moment state survives checkpoint/resume;
- ordinary PyTorch LR schedulers scale the matrix and auxiliary paths together;
- diagnostic host synchronization is deferred until `diagnostics()` is requested.

The checkpoint/DDP/state handling above is engineering hardening for reuse. It preserves the paper-trained $R(+\Delta)$ optimizer geometry while making the package safer for longer or distributed runs.

## Variants

```python
Orbit(model, variant="orbit")
Orbit(model, variant="orbit_norope")
Orbit(model, variant="orbit_diag")
Orbit(model, variant="orbit_identity")
```

- `orbit` — full $R(+\Delta)$ transported $2\times2$ metric.
- `orbit_norope` — opposite-side second moment without rotary transport.
- `orbit_diag` — transported metric with off-diagonal coupling removed.
- `orbit_identity` — statistics path retained, functional preconditioner disabled.

## Repository layout

```text
src/orbit/
    optimizer.py
    baselines.py
    model.py

examples/
    quickstart.py

experiments/
    run.py
    matched.py
    README.md

docs/
    METHOD.md
    ARCHITECTURE.md
    TRAINING.md
    EXPERIMENTS.md

tests/
    test_orbit.py
    test_muon_control.py
    test_correctness.py
    test_docs.py
```

## Experiment status

The paper's primary 124M experiments evaluated the $R(+\Delta)$ optimizer-side transport defined above. The repository therefore treats that architecture as canonical. A later audit clarified that the forward RoPE score itself contains $R(-\Delta)$; the manuscript and repository now state explicitly that ORBIT's opposite orientation is an optimizer design choice rather than an exact causal-score pullback.

The reusable implementation additionally suppresses duplicate statistic updates during gradient-checkpoint recomputation and persists ORBIT statistics across checkpoints. Because the historical 355M exploratory run used the earlier checkpoint behavior, it is not treated as release evidence for the reusable package. The primary 124M matched comparison, ablations, crossed-recipe experiment, broad 124M context, and 124M long-horizon transfer do not use gradient checkpointing and correspond to the canonical optimizer geometry.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the evidence boundary and reproduction notes.

## Citation

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE)
