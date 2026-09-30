# Training notes

The package does not prescribe a training framework. `Orbit` is a normal PyTorch optimizer whose only extra requirement is access to the Q/K pairs and the small RoPE statistics used to build the local metric.

The included `OrbitGPT` model is the reference implementation.

## Smoke run

```bash
python examples/quickstart.py
```

The example creates a small RoPE decoder, runs a few optimizer steps on synthetic tokens, and prints ORBIT diagnostics. It does not write run artifacts.

## Reference settings

The main 124M runs used:

```text
sequence length       512
layers                 12
heads                  12
embedding width       768
batch size              8
Muon momentum        0.95
Newton-Schulz steps     5
Q/K covariance EMA   0.95
metric epsilon       1e-5
condition cap          100
RoPE offsets     1,2,4,8,16,32,64,128
```

The matched configuration used:

```text
matrix learning rate       0.0181771645661641
weight decay               0.005835070036773646
auxiliary LR multiplier    0.49061509693684174
```

A 10% linear warmup was followed by cosine decay to 0.1 times the peak learning rate. The same schedule multiplier was applied to the auxiliary AdamW learning rate.

## Integrating another model

A model needs to expose its Q/K parameter pairs and enable or disable collection of the covariance buffers. See `OrbitGPT.orbit_qk_pairs` and `OrbitGPT.set_orbit_stat_collection` for the expected interface.

The optimizer does not modify the inference graph. After training, only the model weights are needed.
