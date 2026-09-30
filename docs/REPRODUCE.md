# Reproducing ORBIT

This repository contains the exact optimizer/model implementation and the campaign scripts used for the paper.

## Environment

The final paper runs were produced on Tesla T4 GPUs with a representative software stack of Python 3.13, PyTorch 2.11 + CUDA 12.8, Transformers 4.57.6, and Datasets 4.8.5. Exact environment metadata is stored inside each raw JSONL run record.

Install:

```bash
git clone https://github.com/pop123-ux/ORBIT.git
cd ORBIT
pip install -e ".[experiments]"
```

## Data contract

The paper uses FineWeb-Edu with the GPT-2 tokenizer.

- first 400,000 tokens: validation pool
- next 12,000,000 tokens: training pool
- context length: 512
- fixed token stream shared across runs
- each training step samples a random contiguous training window from the fixed pool
- evaluation uses a separate seeded generator and 20 validation batches

Choose a cache location first (the original harness keeps the historical environment variable name so the experiment digest remains identical):

```bash
export ASTRO_PAPER_TOKEN_CACHE=/path/to/fineweb_edu_v1.0.0_12p4m.pt
```

Prepare the cache once:

```bash
python scripts/orbit_campaign.py \
  --phase discovery \
  --work-dir /path/to/orbit_paper \
  --prepare-data
```

The campaign runner reuses the paper harness's deterministic cache logic and stores the data outside git.

## Campaign structure

The original paper campaign was sharded across four logical workers. A shard is only a task partition; it does not change the experiment.

Example:

```bash
python scripts/orbit_campaign.py \
  --phase discovery \
  --work-dir /path/to/orbit_paper \
  --shard-id 0 \
  --num-shards 4
```

Run shard IDs 0, 1, 2, and 3, then freeze discovery:

```bash
python scripts/orbit_merge.py \
  --work-dir /path/to/orbit_paper \
  --phase discovery
```

The frozen best configurations are then used by later phases.

## Core campaign

The core phases are:

```text
discovery  -> 124M, 300 steps, 5 candidates / optimizer
confirm    -> 124M, 900 steps, 5 seeds / optimizer
ablation   -> 124M, 900 steps, pilot mechanism controls
horizon    -> 124M, 2700 steps, 2 seeds
scale      -> 355M, 900 steps, 2 seeds
```

For each phase:

```bash
for SHARD in 0 1 2 3; do
  python scripts/orbit_campaign.py \
    --phase PHASE \
    --work-dir /path/to/orbit_paper \
    --shard-id $SHARD \
    --num-shards 4
done

python scripts/orbit_merge.py \
  --work-dir /path/to/orbit_paper \
  --phase PHASE
```

## Post-scale confirmatory campaign

The confirmatory hierarchy used by the final paper is produced by `orbit_postscale.py`.

```text
xconfig         -> optimizer x frozen-recipe cross-over
matched_tune    -> shared 10-candidate grid for Muon and ORBIT
matched_confirm -> 10 unseen paired seeds, 500-509
ablation_ext    -> 10 seeds per mechanism control, 600-609
astro_horizon   -> ASTRO transfer cell at 124M/2700
astro_scale     -> ASTRO transfer cell at 355M/900
```

Example:

```bash
for SHARD in 0 1 2 3; do
  python scripts/orbit_postscale.py \
    --phase matched_confirm \
    --work-dir /path/to/orbit_paper \
    --shard-id $SHARD \
    --num-shards 4
done

python scripts/orbit_postscale_merge.py \
  --work-dir /path/to/orbit_paper \
  --phase matched_confirm
```

Run the post-scale phases in the order listed above because later phases consume previously frozen configurations.

## Freeze paper evidence

After all required merged files exist:

```bash
python scripts/orbit_paper_artifacts.py \
  --work-dir /path/to/orbit_paper
```

This validates the exact expected seed/config structure before producing paper-level statistics. It refuses to silently promote incomplete evidence.

## Regenerate figures and PDF

For a full PDF build, install a TeX distribution with `latexmk`, `texlive-latex-extra`, `texlive-fonts-recommended`, and `texlive-science`.

From a completed campaign:

```bash
python scripts/orbit_plot.py \
  --work-dir /path/to/orbit_paper

python scripts/orbit_build_paper.py \
  --work-dir /path/to/orbit_paper
```

The compiled manuscript is written to:

```text
docs/paper/main.pdf
```

The manuscript source and generated PDF live in `docs/paper/`.

## Frozen evidence

The exact merged records used for the public paper snapshot are committed under:

```text
results/paper-v1/raw/
```

The compact statistics used for the README and paper discussion are in:

```text
results/paper-v1/results.json
```

This lets readers inspect the evidence without rerunning GPU experiments.

## No-training rebuild from committed evidence

If you only want to verify the paper figures/tables from the public snapshot, do not rerun training:

```bash
python scripts/rebuild_frozen_paper.py
```

This copies the immutable `paper-v1` JSONL evidence into a temporary work directory, revalidates the experiment structure, regenerates the plots/tables, and compiles `docs/paper/main.pdf`.
