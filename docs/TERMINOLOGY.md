# Terminology

This page gives a short translation between paper language and practical meaning.

| Term | Meaning |
|---|---|
| **Optimizer** | The algorithm that turns gradients into parameter updates during training. |
| **Gradient** | Local direction telling us how the loss changes if a parameter moves. |
| **Muon** | A matrix optimizer that transforms matrix gradients using an approximate orthogonal/polar update rather than treating every scalar independently. |
| **Q / K** | Query and key projections in attention. Their dot products determine which tokens attend to which other tokens. |
| **RoPE** | Rotary Positional Embedding. It rotates 2D pairs of Q/K coordinates by position-dependent angles so attention contains relative-position information. |
| **Function space** | Looking at what a parameter does to the model's computation, not only at the numerical shape of the parameter matrix. |
| **Covariance** | A compact description of how two quantities vary together. ORBIT stores 2x2 covariance matrices for each RoPE pair. |
| **Metric** | A local rule for measuring how large a functional movement is. ORBIT uses it to reshape Q/K updates. |
| **Preconditioner** | A transform applied to an update before stepping; it rescales/rotates the update according to local geometry. |
| **Inverse square root** | For a positive matrix M, M^{-1/2} reduces movement in high-sensitivity directions and allows relatively more movement in low-sensitivity directions. |
| **EMA** | Exponential moving average; a smoothed running statistic. |
| **Frobenius norm** | Matrix analogue of Euclidean length: square root of the sum of squared entries. |
| **Joint restoration** | After preconditioning Q and K, ORBIT rescales them together so their combined update norm matches the original Muon candidate budget. |
| **Ablation** | An experiment that removes one component to test whether that component matters. |
| **Paired seed** | Two methods run under the same random seed/setup so their difference is less contaminated by unrelated randomness. |
| **95% confidence interval** | An uncertainty interval for the estimated mean paired effect. |
| **Validation loss** | Loss on held-out data; lower is better. |
| **nats** | Natural-log units used by cross-entropy loss. |
| **Inference graph** | Computation used when the trained model generates output. ORBIT does not change it. |
