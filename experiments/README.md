# Experiments

This directory contains the standalone reproduction path for the paper's **primary Muon–ORBIT comparison and ORBIT mechanism ablations**. It deliberately does not contain manuscript-generation code or JSONL campaign machinery.

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

The final matched hyperparameters are intentionally not hard-coded here yet. The causal-RoPE correction changes the ORBIT update rule, so the shared matched search must be rerun before those values can be frozen. At release, the corrected per-seed JSON records and the selected matched configuration should be committed under `results/corrected/`.

Secondary ASTRO/NorMuon/broad-campaign cells remain part of the canonical paper campaign because they use implementations outside this standalone package. They are not needed to reproduce the paper's primary mechanism-isolated claim.
