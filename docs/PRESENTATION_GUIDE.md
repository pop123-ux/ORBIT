# Presentation guide

## One-sentence explanation

**ORBIT starts from Muon's matrix update, then reshapes only the attention query/key step using the local geometry induced by RoPE, while preserving the total Q/K step norm and leaving inference unchanged.**

## 30-second explanation

Muon knows that Transformer weights are matrices, but it largely treats same-shaped matrices with the same update geometry. Q and K are special: together they create attention scores, and RoPE rotates their coordinates according to relative token position. ORBIT uses running Q/K statistics and those known RoPE rotations to build tiny 2x2 local metrics. It preconditions the Muon Q/K update with those metrics, then restores the original combined update norm. The matched 124M experiment gives a mean validation-loss improvement of 0.0064 nats over Muon, with a 95% CI excluding zero and ORBIT winning 9 of 10 paired held-out runs.

## The key contrast with Muon

Muon asks: **what is a good matrix-shaped update?**

ORBIT additionally asks: **for this particular Q/K matrix, how does that update move the attention function?**

## What not to claim

Do not say:
- the large independently tuned ORBIT-vs-Muon gap is purely algorithmic;
- ORBIT is proven to beat ASTRO at 355M;
- the advantage is proven to grow with parameter count;
- full off-diagonal coupling is absolutely necessary;
- the two-seed transfer cells are high-powered statistical evidence.

Say instead:
- the matched comparison isolates a reproducible ORBIT effect over Muon;
- ablations support functional Q/K conditioning and an additional RoPE-aware contribution;
- longer-horizon and 355M experiments show transfer in the tested settings;
- broader scaling remains future work.
