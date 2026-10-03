# Changelog

## 0.1.0

Initial ORBIT research release candidate.

- canonical paper-trained optimizer-side `R(+Delta)` rotary transport;
- Muon-based Q/K candidate updates with local `2 x 2` inverse-square-root preconditioning;
- joint Q/K Frobenius restoration;
- identity, no-RoPE, and diagonal ablation variants;
- persistent Q/K statistic state, DDP aggregation, checkpoint-safe reference integration, and scheduler-safe auxiliary learning-rate coupling;
- standalone 124M primary-paper reproduction runner and exact frozen matched configuration;
- exact non-ASTRO optimizer definitions used in the paper: AdamW, Muon, NorMuon, and published/reference AdaMuon;
- lightweight synthetic baseline examples for API and update-rule experimentation, intentionally separate from the full research campaign;
- explicit adapter API for compatible external RoPE models;
- regression tests for forward-vs-optimizer rotary orientation, metric numerics, Muon control parity, paper baselines, checkpointing, distributed statistics, state persistence, adapters, and paper protocol selection.

The GitHub `v0.1.0` tag/release should be created from the final verified public commit so that the paper points to an immutable software snapshot.
