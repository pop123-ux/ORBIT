# Paper-tested baselines

The standalone package includes the non-ASTRO optimizer definitions used in the ORBIT paper: AdamW, Muon, NorMuon, and the published/reference AdaMuon variant. These are included so readers can inspect and exercise the actual update rules compared around ORBIT without vendoring the full research campaign.

## Definitions

- **AdamW** — PyTorch AdamW with the paper's GPT decay routing: 2-D parameters decay, while vectors, biases, and normalization parameters do not.
- **Muon** — Nesterov momentum at 0.95, five-step Newton--Schulz orthogonalization, Muon's aspect-ratio scaling, and an auxiliary AdamW path for non-matrix parameters.
- **NorMuon** — the same Muon path followed by a row-wise second moment after orthogonalization and Frobenius-norm restoration before the usual Muon scale.
- **AdaMuon** — the published/reference definition used in the paper: sign before Newton--Schulz, elementwise post-polar second moment without bias correction, and 0.2-RMS alignment; non-matrix parameters use the shared auxiliary AdamW path.

ASTRO is intentionally not vendored into this repository; it belongs to a separate optimizer research line.

## Use

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

Accepted names are `adamw`, `muon`, `normuon`, and `adamuon`; `adamuon_ref` is retained as an alias for the exact reference AdaMuon variant used by the historical campaign.

The same definitions are available from `experiments/run.py` for lightweight follow-up runs on the repository's shared model/data loop. This is intentionally a convenience experiment path rather than a reconstruction of the full broad tuning campaign; see [`../experiments/README.md`](../experiments/README.md).

Two smaller synthetic examples are also provided:

```bash
python examples/toy_baseline_step.py
python examples/toy_compare.py --steps 10
```

These examples are intentionally **toy runs**. They demonstrate the implementations and shared API; they are not a reproduction of the broad paper campaign and should not be used to recover reported paper numbers. The frozen primary Muon--ORBIT protocol remains under [`../experiments/`](../experiments/).
