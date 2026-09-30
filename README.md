# ORBIT

[![CI](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml/badge.svg)](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-black)

**Function-Space Optimization for Rotary Query-Key Interactions**

ORBIT is a research optimizer for RoPE-based Transformer language models. It keeps Muon's matrix update as the base step, then modifies only query/key (Q/K) updates using tiny RoPE-aware local metrics derived from the function those matrices perform inside attention.

> **Core idea:** optimize Q/K in a geometry induced by rotary attention, not only by matrix shape.

[Paper source](docs/paper/main.tex) · [Method](docs/METHOD.md) · [Reproduction guide](docs/REPRODUCE.md) · [Paper results](results/paper-v1/results.json)

---

## Primary result

The paper's primary comparison uses **identical hyperparameters** for Muon and ORBIT, chosen from the same 10-candidate grid before evaluation on 10 unseen paired seeds.

| Method | Mean validation loss | Mean runtime | Peak CUDA |
|---|---:|---:|---:|
| Muon | 5.4951 | 802.9 s | 4.710 GB |
| ORBIT | **5.4887** | 859.3 s | 4.700 GB |

**Paired ORBIT − Muon:** **−0.006396 nats**  
**95% CI:** **[−0.009838, −0.002955]**  
**Wins:** **9 / 10** held-out paired runs  
**Runtime overhead:** ~**7.0%** in the measured 124M/900-step setting  
**Inference overhead:** **none**; ORBIT is training-only.

The broader independently tuned benchmark and transfer experiments are included for context, but the matched comparison above is the paper's primary mechanism-isolated estimate.

---

## What ORBIT changes

For an ordinary Muon candidate update, ORBIT leaves all non-Q/K hidden matrices on the standard Muon path. For each attention Q/K projection it:

1. collects exponential-moving-average Q/K covariance statistics;
2. transports those statistics through RoPE relative-position rotations;
3. constructs a local symmetric \(2\times2\) metric for every RoPE frequency pair;
4. applies an analytic inverse-square-root preconditioner;
5. jointly restores the Q/K Frobenius norm so the method cannot win simply by taking a larger step.

The model architecture, learned parameterization, forward pass at inference, and inference graph remain unchanged.

---

## Install

```bash
git clone https://github.com/pop123-ux/ORBIT.git
cd ORBIT
pip install -e .
```

For paper reproduction:

```bash
pip install -e ".[experiments]"
```

To rebuild the manuscript and figures directly from the committed frozen evidence, with **no GPU training**:

```bash
python scripts/rebuild_frozen_paper.py
```

---

## Minimal usage

ORBIT currently expects a model exposing the Q/K pairs and RoPE statistics used by the optimizer. The repository includes the exact RoPE GPT implementation used in the paper.

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
    lr=0.0181771645661641,
    adamw_lr=0.0181771645661641 * 0.49061509693684174,
    weight_decay=0.005835070036773646,
)
```

A pure Muon control using the same parameter-routing policy is also provided:

```python
from orbit import Muon
optimizer = Muon(model, lr=0.0181771645661641)
```

---

## Repository map

```text
ORBIT/
├── src/orbit/                  # ORBIT optimizer, Muon control, RoPE GPT model
├── scripts/                    # exact campaign/orchestration scripts used for the paper
├── configs/                    # frozen paper protocol and matched configs
├── results/paper-v1/           # frozen raw runs + paper-level statistics
├── docs/                       # method, terminology, reproduction, experiment map
├── docs/paper/                 # LaTeX manuscript source
├── tests/                      # optimizer and metric tests
└── .github/workflows/          # CI
```

---

## Reproducing the paper

There are three intended levels.

### 1. Smoke test

```bash
pytest -q
```

This checks the closed-form \(2\times2\) inverse metric, RoPE metric construction, finite optimizer steps, statistics collection, and the identity control.

### 2. Primary matched Muon–ORBIT result

Prepare the 12.4M-token FineWeb-Edu cache described in [docs/REPRODUCE.md](docs/REPRODUCE.md), then run the matched phase from the exact campaign harness in `scripts/`.

The frozen selected configuration is committed in:

```text
configs/matched_best_configs.json
```

The held-out seeds are 500–509.

### 3. Full paper campaign

The full campaign contains:

- broad independently tuned optimizer context;
- optimizer × recipe cross-over;
- shared-grid matched tuning;
- 10-seed matched confirmation;
- 10-seed mechanism ablation;
- 124M / 2700-step transfer;
- 355M / 900-step transfer;
- ASTRO transfer cells.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the evidence hierarchy and exact scripts.

---

## Mechanism variants

The optimizer exposes the controls used in the paper:

```python
Orbit(model, variant="orbit")           # full method
Orbit(model, variant="orbit_norope")    # no relative-position rotation
Orbit(model, variant="orbit_diag")      # diagonal 2x2 metric
Orbit(model, variant="orbit_identity")  # statistics path, no preconditioning
```

These controls are important because they separate generic Q/K adaptation from RoPE-aware conditioning and full off-diagonal coupling.

---

## Paper protocol

- **Data:** FineWeb-Edu, GPT-2 tokenizer
- **Context:** 512
- **Fixed stream:** 400,000 validation tokens + 12,000,000 training tokens
- **124M:** 12 layers, 12 heads, width 768, batch 8
- **355M:** 24 layers, 16 heads, width 1024, batch 4, gradient checkpointing
- **Schedule:** 10% warmup + cosine decay to 0.1× peak LR
- **Muon momentum:** 0.95
- **Newton–Schulz iterations:** 5
- **ORBIT covariance EMA:** 0.95
- **Metric regularizer:** \(10^{-5}I\)
- **Condition cap:** 100
- **RoPE displacement grid:** {1, 2, 4, 8, 16, 32, 64, 128}

The 355M experiment uses a smaller batch size, so it is treated as transfer evidence rather than a clean scaling law.

---

## Evidence policy

The repository deliberately distinguishes:

- **primary confirmatory evidence** — matched Muon vs ORBIT, same frozen hyperparameters, 10 paired held-out seeds;
- **mechanism evidence** — identity / no-RoPE / diagonal controls;
- **secondary transfer evidence** — longer horizon and 355M, two seeds each;
- **broader context** — independently tuned optimizers, which mixes update-rule and recipe effects.

This distinction is important: the much larger independently tuned ORBIT–Muon gaps are **not** interpreted as pure algorithmic effects.

---

## Citation

If you use ORBIT, please cite the paper and this repository. A machine-readable citation file is included as [CITATION.cff](CITATION.cff).

---

## Status

This is the public research companion repository for the ORBIT paper. The paper establishes the method on RoPE-modified GPT-2-style decoders; larger controlled scaling studies, grouped-query attention, QK normalization, mixture-of-experts models, and lower-overhead implementations remain open directions.

Independent replication and extension are encouraged.
