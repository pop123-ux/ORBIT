# ORBIT

[![CI](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml/badge.svg)](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![License](https://img.shields.io/badge/license-MIT-black)

**Function-Space Optimization for Rotary Query-Key Interactions**

ORBIT modifies the Muon candidate update for rotary Query/Key projections using a local metric derived from the causal RoPE score. The paper is the method specification; this repository mirrors its sign convention, second-moment definition, preconditioner, restoration rule, and training-state semantics.

[Method](docs/METHOD.md) · [Architecture](docs/ARCHITECTURE.md) · [Training](docs/TRAINING.md) · [Experiment status](docs/EXPERIMENTS.md)

## Definition

For a query at position $i$ attending to a causal key at $j\le i$, define

```math
\Delta=i-j\ge0.
```

The RoPE convention used here gives

```math
(R_f(i)q_{i,f})^\top(R_f(j)k_{j,f})
=
q_{i,f}^\top R_f(-\Delta)k_{j,f}.
```

Let the unrotated per-frequency **uncentered second moments** be

```math
S_{Q,f}=\mathbb E[q_fq_f^\top],
\qquad
S_{K,f}=\mathbb E[k_fk_f^\top].
```

ORBIT defines

```math
M_{Q,f}
=
\mathbb E_{\Delta}
\left[
R_f(-\Delta)S_{K,f}R_f(-\Delta)^\top
\right],
```

```math
M_{K,f}
=
\mathbb E_{\Delta}
\left[
R_f(-\Delta)^\top S_{Q,f}R_f(-\Delta)
\right].
```

The expectation is approximated by a uniform average over

```math
\mathcal D=\{1,2,4,8,16,32,64,128\}.
```

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

The inverse square root whitens the local quadratic metric; the joint Frobenius restoration fixes the aggregate Q/K parameter-space step magnitude. The derivation is in [`docs/METHOD.md`](docs/METHOD.md).

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

- `deltas` are non-negative causal distances $i-j$ and metric transport uses $R(-\Delta)$;
- an autograd score-gradient test verifies the same sign convention independently of the metric code;
- the $2\times2$ inverse metric is checked against a float64 eigendecomposition over multiple scales and powers;
- the joint Q/K Frobenius budget is tested directly;
- `orbit_identity` is checked against the matched Muon control;
- checkpoint recomputation contributes no duplicate second-moment update;
- DDP ranks aggregate sufficient statistics before the EMA;
- second-moment state survives checkpoint/resume;
- ordinary PyTorch LR schedulers scale the matrix and auxiliary paths together;
- diagnostic host synchronization is deferred until `diagnostics()` is requested.

## Variants

```python
Orbit(model, variant="orbit")
Orbit(model, variant="orbit_norope")
Orbit(model, variant="orbit_diag")
Orbit(model, variant="orbit_identity")
```

- `orbit` — full transported $2\times2$ metric.
- `orbit_norope` — opposite-side second moment without relative-position transport.
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

The pre-release audit changed the implementation that generated the earlier paper campaign. Those historical numerical results are **not release evidence for the corrected method**.

The corrected primary protocol is baseline-selected: Muon alone is tuned over ten deterministic 124M/900-step candidates, its lowest-loss recipe is frozen, and that same recipe is then used for the held-out Muon–ORBIT comparison and ORBIT ablations. [`experiments/matched.py`](experiments/matched.py) reproduces the candidate grid and selection rule; [`experiments/run.py`](experiments/run.py) executes individual runs. Corrected per-seed records will be committed only after the post-audit campaign is complete. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Citation

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE)
