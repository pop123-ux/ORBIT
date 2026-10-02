# Experiments

This directory contains the standalone reproduction path for the paper's **primary Muon–ORBIT comparison and ORBIT mechanism ablations**. The paper is authoritative: Muon alone is tuned over ten deterministic candidates, its lowest-loss recipe is frozen, and that one recipe is then used for the held-out Muon–ORBIT comparison and the ORBIT ablations. This directory deliberately contains no manuscript-generation code or JSONL campaign machinery.

Install the optional dependencies:

```bash
pip install -e ".[experiments]"
```

Build the exact fixed FineWeb-Edu token pool used by the paper:

```bash
python experiments/run.py \
  --prepare-data \
  --token-cache /path/to/fineweb_edu_v1.0.0_12p4m.pt
```

The cache contains 12.4M GPT-2 tokens from `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, revision `v1.0.0`: 400k validation tokens followed by a 12M-token training pool.

Generate the exact ten primary Muon candidates:

```bash
python experiments/matched.py candidates \
  --output results/corrected/matched/candidates.json
```

Run each candidate with `--optimizer muon --size 124M --steps 900 --seed 0`, using the corresponding `lr`, `scalar_lr_mult`, and `weight_decay`. Store the ten run records as ordinary JSON files, then freeze the lowest-loss Muon candidate:

```bash
python experiments/matched.py select \
  results/corrected/matched/tune/muon-trial-*.json \
  --output results/corrected/matched/selected_config.json
```

Only after that file is frozen should held-out Muon and ORBIT confirmation runs be launched with the selected configuration.

A single run is written as one JSON file:

```bash
python experiments/run.py \
  --optimizer orbit \
  --size 124M \
  --steps 900 \
  --seed 500 \
  --lr <frozen-matched-lr> \
  --scalar-lr-mult <frozen-matched-multiplier> \
  --weight-decay <frozen-matched-weight-decay> \
  --token-cache /path/to/fineweb_edu_v1.0.0_12p4m.pt \
  --output results/corrected/matched/orbit-seed500.json
```

Supported methods are `muon`, `orbit`, `orbit_identity`, `orbit_norope`, and `orbit_diag`.

The final matched hyperparameters are intentionally not hard-coded here yet. The corrected campaign must first rerun the ten-candidate **Muon-only** search and freeze its winner. ORBIT tuning results are not part of the primary selection rule. At release, the selected configuration and corrected per-seed JSON records should be committed under `results/corrected/`.

Secondary ASTRO/NorMuon/broad-campaign cells remain part of the canonical paper campaign because they use implementations outside this standalone package. They are not needed to reproduce the paper's primary mechanism-isolated claim.
