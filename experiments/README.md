# Experiments

This directory contains the standalone reproduction path for the paper's primary Muon–ORBIT comparison and ORBIT mechanism ablations. The paper protocol evaluates the same ten deterministic 124M/900-step candidate recipes for both Muon and ORBIT on tuning seed 0. Each optimizer selects its own lowest-loss candidate; in the reported campaign both selected the same shared recipe, which was then frozen before the held-out paired comparison.

Install the optional dependencies:

```bash
pip install -e ".[experiments]"
```

Build the fixed FineWeb-Edu token pool used by the paper:

```bash
python experiments/run.py \
  --prepare-data \
  --token-cache /path/to/fineweb_edu_v1.0.0_12p4m.pt
```

The cache contains 12.4M GPT-2 tokens from `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, revision `v1.0.0`: 400k validation tokens followed by a 12M-token training pool.

## Direct reproduction of the primary comparison

The exact frozen recipe and paper seed sets are checked into [`configs/paper_124m_matched.json`](../configs/paper_124m_matched.json). To reproduce the headline comparison, users do not need to spend compute re-running the tuning grid.

Run one ORBIT cell:

```bash
python experiments/paper_run.py \
  --optimizer orbit \
  --seed 500 \
  --token-cache /path/to/fineweb_edu_v1.0.0_12p4m.pt \
  --output results/paper/orbit-seed500.json
```

Run the paired Muon cell with the same seed:

```bash
python experiments/paper_run.py \
  --optimizer muon \
  --seed 500 \
  --token-cache /path/to/fineweb_edu_v1.0.0_12p4m.pt \
  --output results/paper/muon-seed500.json
```

Repeat for primary confirmation seeds `500` through `509`. `paper_run.py` reads the exact frozen `shared-04` recipe from the config and rejects non-primary seeds unless `--allow-nonprimary-seed` is given explicitly.

The frozen primary optimizer values are:

```text
matrix lr        0.0181771645661641
weight decay     0.005835070036773646
aux multiplier   0.49061509693684174
aux AdamW lr     0.008917991355665525
```

## Auditing the tuning procedure

To regenerate the exact ten shared candidates:

```bash
python experiments/matched.py candidates \
  --output results/matched/candidates.json
```

Run each candidate once for Muon and once for ORBIT with `--size 124M --steps 900 --seed 0`, using the corresponding `lr`, `scalar_lr_mult`, and `weight_decay`.

Then verify that the independently selected winners coincide and freeze the shared recipe:

```bash
python experiments/matched.py select \
  --muon results/matched/tune/muon-trial-*.json \
  --orbit results/matched/tune/orbit-trial-*.json \
  --output results/matched/selected_config.json
```

The reported paper campaign selected `shared-04` for both optimizers. The frozen recipe is then used unchanged for all held-out Muon/ORBIT confirmation runs.

Supported methods in `run.py` are `muon`, `orbit`, `orbit_identity`, `orbit_norope`, and `orbit_diag`.

The canonical ORBIT package uses the same $R(+\Delta)$ optimizer-side transport as the paper's non-checkpointed 124M experiments. The package additionally makes second-moment state persistent, synchronizes statistics across DDP ranks, and suppresses duplicate EMA updates during checkpoint recomputation. Those engineering safeguards are intended for downstream use and do not change the primary 124M optimizer geometry.

Secondary ASTRO/NorMuon/broad-campaign cells remain part of the full paper campaign because they use implementations outside this standalone package. They are not required to reproduce the primary matched Muon–ORBIT comparison.
