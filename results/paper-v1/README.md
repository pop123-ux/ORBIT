# Frozen paper evidence

This directory is the public evidence snapshot corresponding to the ORBIT paper.

- `results.json`: compact paper-level statistics and interpretation notes.
- `raw/confirm.jsonl`: broad independently tuned 124M confirmation.
- `raw/xconfig.jsonl`: optimizer × frozen-recipe cross-over.
- `raw/matched_tune.jsonl`: shared 10-candidate Muon/ORBIT selection.
- `raw/matched_confirm.jsonl`: primary 10-seed matched confirmation.
- `raw/ablation_ext.jsonl`: merged 10-seed mechanism ablation.
- `raw/horizon_with_astro.jsonl`: 124M/2700 transfer including ASTRO.
- `raw/scale_with_astro.jsonl`: 355M/900 transfer including ASTRO.

The JSONL records are intentionally committed so the headline numbers can be independently recomputed without rerunning the expensive training campaign.
