# ORBIT

<img width="1448" height="1086" alt="53da1bac-27cc-42c6-a4fe-4c999ceab145" src="https://github.com/user-attachments/assets/d51e258e-feaf-44ff-9209-7650a02d34e4" />

[![CI](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml/badge.svg)](https://github.com/pop123-ux/ORBIT/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![License](https://img.shields.io/badge/license-MIT-black)

**Function-Space Optimization for Rotary Query-Key Interactions**

ORBIT modifies the Muon candidate update for rotary Query/Key projections using a compact optimizer-side geometry built from RoPE frequency structure and opposite-side Q/K second moments. The canonical implementation in this repository matches the optimizer architecture used for the paper's primary 124M experiments: non-negative causal separations are transported with the optimizer orientation $R(+\Delta)$, followed by inverse-square-root preconditioning and joint Q/K Frobenius restoration.

[Method](docs/METHOD.md) · [Architecture](docs/ARCHITECTURE.md) · [Adapters](docs/ADAPTERS.md) · [Paper baselines](docs/BASELINES.md) · [Training](docs/TRAINING.md) · [Experiment status](docs/EXPERIMENTS.md)

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

## Adapting another RoPE model

ORBIT is function-aware, so it needs architectural information that cannot be inferred safely from shape alone: which matrices form Q/K pairs, how those outputs map to heads and RoPE frequency pairs, where the unrotated activations appear, and which 2-D parameters should stay on the auxiliary AdamW path.

The repository therefore provides an explicit adapter rather than silently guessing those semantics:

```python
from orbit import Orbit, OrbitAdapter, QKPairSpec

base_model = ...
pairs = [
    QKPairSpec(
        q_proj=layer.self_attn.q_proj,
        k_proj=layer.self_attn.k_proj,
        n_head=base_model.config.num_attention_heads,
        rope_base=getattr(base_model.config, "rope_theta", 10_000.0),
    )
    for layer in base_model.layers
]

model = OrbitAdapter(base_model, pairs).to(device)
optimizer = Orbit(model, lr=0.02, adamw_lr=3e-4, weight_decay=0.05)
```

The v0.1 generic adapter targets separate Q/K linear projections with ordinary multi-head attention, matching the architecture evaluated in the paper. GQA/MQA, fused QKV, unusual RoPE layouts, QK normalization, or other architecture-specific transformations should use a native integration or purpose-built adapter instead of a shape-only approximation. This explicit boundary is deliberate: extending function-aware optimization requires encoding the computation a parameter actually performs.

See [`docs/ADAPTERS.md`](docs/ADAPTERS.md) for the integration contract, checkpointing notes, and research-extension guidance.

## Paper-tested baselines

The package also exposes the non-ASTRO optimizer definitions used around ORBIT in the paper: AdamW, Muon, NorMuon, and the published/reference AdaMuon variant.

```python
from orbit import build_paper_baseline

optimizer = build_paper_baseline(
    "normuon",
    model,
    lr=0.02,
    scalar_lr_mult=0.1,
    weight_decay=0.05,
)
```

For a quick API-level exercise rather than a paper reproduction:

```bash
python examples/toy_baseline_step.py
python examples/toy_compare.py --steps 10
```

ASTRO is intentionally not vendored here. See [`docs/BASELINES.md`](docs/BASELINES.md) for the exact distinctions among the included baseline update rules.

## Exact primary-paper reproduction

The frozen primary recipe and seed lists are stored in [`configs/paper_124m_matched.json`](configs/paper_124m_matched.json). Users do **not** need to rerun the ten-candidate tuning grid merely to reproduce the headline held-out comparison.

Prepare the fixed FineWeb-Edu token pool:

```bash
pip install -e ".[experiments]"
python experiments/run.py \
  --prepare-data \
  --token-cache /path/to/fineweb_edu_v1.0.0_12p4m.pt
```

Then run one frozen primary cell directly:

```bash
python experiments/paper_run.py \
  --optimizer orbit \
  --seed 500 \
  --token-cache /path/to/fineweb_edu_v1.0.0_12p4m.pt \
  --output results/paper/orbit-seed500.json
```

Repeat for seeds `500` through `509` and for `--optimizer muon` to reproduce the paired primary comparison. The exact selected `shared-04` learning rate, weight decay, auxiliary multiplier, data specification, and study seed sets are machine-readable in the config file.

The original ten-candidate selector remains in [`experiments/matched.py`](experiments/matched.py) for users who want to audit or rerun the selection procedure itself.

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
- the external-model adapter is tested for statistic collection, training, persistence, parameter routing, and rejection of ambiguous GQA/MQA mappings;
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
    adapter.py
    baselines.py
    paper_baselines.py
    model.py

examples/
    quickstart.py
    toy_baseline_step.py
    toy_compare.py

experiments/
    run.py
    paper_run.py
    matched.py
    README.md

configs/
    orbit.json
    paper_124m_matched.json

docs/
    METHOD.md
    ARCHITECTURE.md
    ADAPTERS.md
    BASELINES.md
    TRAINING.md
    EXPERIMENTS.md

tests/
    test_adapter.py
    test_orbit.py
    test_muon_control.py
    test_paper_baselines.py
    test_correctness.py
    test_experiment_protocol.py
    test_docs.py
```

## Experiment status

The paper's primary 124M experiments evaluated the $R(+\Delta)$ optimizer-side transport defined above. The repository therefore treats that architecture as canonical. A later audit clarified that the forward RoPE score itself contains $R(-\Delta)$; the manuscript and repository now state explicitly that ORBIT's opposite orientation is an optimizer design choice rather than an exact causal-score pullback.

The reusable implementation additionally suppresses duplicate statistic updates during gradient-checkpoint recomputation and persists ORBIT statistics across checkpoints. Because the historical 355M exploratory run used the earlier checkpoint behavior, it is not treated as release evidence for the reusable package. The primary 124M matched comparison, ablations, crossed-recipe experiment, broad 124M context, and 124M long-horizon transfer do not use gradient checkpointing and correspond to the canonical optimizer geometry.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the evidence boundary and reproduction notes.

## Release and citation

Package version `0.1.0` is the first public-release target. [`CHANGELOG.md`](CHANGELOG.md) records the release contents, and [`RELEASE.md`](RELEASE.md) contains the final public/tag checklist. The immutable `v0.1.0` tag should be created from the final manually reviewed public commit.

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE)
