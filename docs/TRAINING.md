# Training notes

ORBIT is a PyTorch optimizer with one additional requirement: the model must expose the query/key pairs and the small RoPE statistics used to build the local metric.

The included `OrbitGPT` model is the reference implementation.

## Smoke run

~~~bash
python examples/quickstart.py
~~~

The example trains a small RoPE decoder for a few synthetic steps and prints optimizer diagnostics. It does not write experiment artifacts.

## Reference 124M setup

The primary matched experiments used:

| Setting | Value |
| --- | ---: |
| Sequence length | 512 |
| Layers | 12 |
| Attention heads | 12 |
| Embedding width | 768 |
| Batch size | 8 |
| Muon momentum | 0.95 |
| Newton-Schulz steps | 5 |
| Q/K covariance EMA | 0.95 |
| Metric epsilon | \(10^{-5}\) |
| Condition cap | 100 |
| Gradient clipping | 1.0 |

The RoPE displacement set was

$$
\mathcal D=\{1,2,4,8,16,32,64,128\}.
$$

The matched configuration selected by both Muon and ORBIT used:

| Hyperparameter | Value |
| --- | ---: |
| Matrix learning rate | 0.0181771645661641 |
| Weight decay | 0.005835070036773646 |
| Auxiliary LR multiplier | 0.49061509693684174 |

The corresponding auxiliary AdamW learning rate is

$$
0.0181771645661641\times0.49061509693684174
\approx
0.008918.
$$

A 10% linear warmup was followed by cosine decay to \(0.1\times\) the peak learning rate. The same schedule multiplier was applied to the matrix and auxiliary learning rates.

## Optimizer construction

Reference-scale construction:

~~~python
optimizer = Orbit(
    model,
    lr=0.0181771645661641,
    adamw_lr=0.008918,
    weight_decay=0.005835070036773646,
    momentum=0.95,
    ns_steps=5,
    functional_power=1.0,
    metric_eps=1e-5,
    metric_condition_cap=100.0,
    deltas=(1, 2, 4, 8, 16, 32, 64, 128),
)
~~~

The values above reproduce the matched experimental recipe. They are not intended as universal defaults.

## Integrating another model

A new model needs to expose:

~~~python
model.orbit_qk_pairs()
model.set_orbit_stat_collection(enabled)
~~~

The attention implementation must maintain the unrotated Q/K pair statistics and provide an `orbit_metrics(...)` method.

The current `Orbit` class also assumes the token embedding and LM head are available as

~~~python
model.wte
model.lm_head
~~~

for auxiliary parameter routing.

If a model uses different names or a different parameter organization, adapt that routing explicitly rather than relying on shape alone.

## Diagnostics

After optimizer steps, the full ORBIT variant exposes averaged diagnostic values:

~~~python
optimizer.diagnostics()
~~~

The current diagnostics include mean query metric condition, mean key metric condition, and the joint Frobenius restore factor.

## Inference

No optimizer state is needed for inference. The trained checkpoint can be used with the ordinary model forward pass; ORBIT does not add a layer, parameter, or inference-time kernel.
