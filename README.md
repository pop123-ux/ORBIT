# ORBIT

[![CI](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml/badge.svg)](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![License](https://img.shields.io/badge/license-MIT-black)

**Function-Space Optimization for Rotary Query-Key Interactions**

ORBIT modifies the Muon candidate update only for rotary Query/Key projections. Its local metric is derived from the causal RoPE score itself, so the update geometry follows the computation performed by the Q/K block rather than matrix shape alone.

Documentation: [method](docs/METHOD.md) · [architecture](docs/ARCHITECTURE.md) · [training](docs/TRAINING.md) · [experiment status](docs/EXPERIMENTS.md)

## Core definition

For a causal query at position $i$ attending to a key at position $j\le i$, define

```math
\Delta=i-j\ge 0.
```

With the rotation convention used by the implementation,

```math
s_{ij,f}
=
q_{i,f}^{\top}R_f(-\Delta)k_{j,f}.
```

For unrotated per-frequency covariances

```math
C_{Q,f}=\mathbb E[q_fq_f^\top],
\qquad
C_{K,f}=\mathbb E[k_fk_f^\top],
```

ORBIT uses

```math
M_{Q,f}
=
\mathbb E_{\Delta}
\left[
R_f(-\Delta)C_{K,f}R_f(-\Delta)^\top
\right],
```

```math
M_{K,f}
=
\mathbb E_{\Delta}
\left[
R_f(-\Delta)^\top C_{Q,f}R_f(-\Delta)
\right].
```

The Muon candidates $U_Q,U_K$ are transformed by

```math
\widehat U_Q=M_Q^{-1/2}U_Q,
\qquad
\widehat U_K=M_K^{-1/2}U_K,
```

then jointly rescaled so that

```math
\lVert\widetilde U_Q\rVert_F^2+
\lVert\widetilde U_K\rVert_F^2
=
\lVert U_Q\rVert_F^2+
\lVert U_K\rVert_F^2.
```

The full derivation is in [`docs/METHOD.md`](docs/METHOD.md).

## Correctness invariants

The implementation makes the conventions that matter for ORBIT explicit:

- `deltas` are non-negative **causal distances** $i-j$; negative values are rejected.
- RoPE transport uses $R(-\Delta)$, matching the actual score produced by `_apply_rope`.
- Gradient-checkpoint recomputation does not update the Q/K covariance EMA a second time.
- DDP ranks aggregate Q/K sufficient statistics before the EMA, so every replica applies the same ORBIT metric.
- Q/K covariance state and the observation counter are persistent model buffers and survive checkpoint/resume.
- The auxiliary learning rate is stored as a fixed ratio to the matrix learning rate, so a standard PyTorch LR scheduler scales both paths together.
- Diagnostic device-to-host transfers occur only when `optimizer.diagnostics()` is requested.

These properties are covered by regression tests.

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

The included `OrbitGPT` is the reference interface. A different RoPE model must expose its Q/K pairs and the attention modules that construct the ORBIT metrics; see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Variants

```python
Orbit(model, variant="orbit")
Orbit(model, variant="orbit_norope")
Orbit(model, variant="orbit_diag")
Orbit(model, variant="orbit_identity")
```

`orbit_identity` preserves the same statistics path while disabling the functional preconditioner. `orbit_norope` removes relative-position transport. `orbit_diag` removes off-diagonal coupling inside each $2\times2$ metric.

## Repository layout

```text
src/orbit/
    optimizer.py
    baselines.py
    model.py

examples/
    quickstart.py

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

## Empirical status

A pre-release audit found that the earlier experiment campaign transported the RoPE metric with the opposite relative-position sign, and that gradient checkpointing caused duplicate covariance updates in the 355M runs. The implementation and documentation in this repository correct both issues.

The earlier numerical paper results are therefore **not treated as evidence for the corrected implementation**. Corrected matched, ablation, and transfer runs must complete before the manuscript is released. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) records exactly which evidence is affected.

## Citation

Citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE)
