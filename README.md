# ORBIT

[![CI](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml/badge.svg)](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-black)

ORBIT is a RoPE-aware extension of Muon for Transformer query/key updates. It keeps Muon's matrix update as the base direction, then preconditions Q/K in the local geometry induced by rotary attention. The model itself is unchanged at inference.

The repository contains the optimizer, a matched Muon control, the RoPE GPT reference model used for development, tests, and compact notes on the method and reported experiments.

## Install

```bash
git clone https://github.com/pop123-ux/ORBIT.git
cd ORBIT
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Quick start

```python
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
```

A Muon control with the same parameter routing is available as `orbit.Muon`.

## Method

For one RoPE frequency pair, the attention contribution is

```text
q^T R(delta) k
```

ORBIT keeps exponential-moving-average 2x2 covariances for Q and K, transports the opposite-side covariance through the RoPE rotation, and builds a local metric

```text
M_Q = E_delta[R C_K R^T]
M_K = E_delta[R^T C_Q R]
```

The Muon candidate is preconditioned with `M^(-1/2)`. Q and K are then rescaled jointly so their combined Frobenius norm matches the original Muon candidate. Non-Q/K hidden matrices stay on the Muon path; embeddings, biases, and normalization parameters use the auxiliary AdamW path.

The implementation uses closed-form 2x2 matrix algebra rather than a batched eigensolver.

More detail: [method](docs/METHOD.md) · [implementation](docs/ARCHITECTURE.md)

## Variants

```python
Orbit(model, variant="orbit")
Orbit(model, variant="orbit_norope")
Orbit(model, variant="orbit_diag")
Orbit(model, variant="orbit_identity")
```

`orbit_identity` keeps the ORBIT statistics path but disables functional preconditioning. The other two variants remove RoPE transport or off-diagonal coupling.

## Reported experiments

The matched 124M comparison used the same selected hyperparameters for Muon and ORBIT and ten held-out paired seeds.

| method | mean validation loss |
| --- | ---: |
| Muon | 5.4951 |
| ORBIT | 5.4887 |

Mean paired difference: **-0.006396 nats**, 95% CI **[-0.009838, -0.002955]**, with ORBIT lower on 9/10 seeds. The measured runtime overhead was about 7% in that setting; peak allocated CUDA memory was effectively unchanged.

Ablations and the 124M/2700-step and 355M/900-step transfer runs are summarized in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Layout

```text
src/orbit/
    optimizer.py    ORBIT update and 2x2 metric algebra
    baselines.py    matched Muon control
    model.py        RoPE GPT reference model

examples/
    quickstart.py

docs/
    METHOD.md
    ARCHITECTURE.md
    EXPERIMENTS.md
    TRAINING.md

tests/
```

## Citation

See [CITATION.cff](CITATION.cff).
