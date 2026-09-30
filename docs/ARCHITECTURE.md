# Implementation notes

The public implementation is intentionally small. The optimizer-specific code is in `src/orbit/optimizer.py`; the reference attention module that exposes the required statistics is in `src/orbit/model.py`.

### Muon candidate

`newton_schulz` applies the five-step quintic polar iteration used by the Muon path. `Orbit._spectral_candidate` adds momentum and the aspect-ratio scaling before the Q/K-specific stage.

The matched control in `src/orbit/baselines.py` uses the same matrix update and parameter routing but does not collect ORBIT statistics.

### Q/K statistics

`RotaryAttention._update_covariances` stores one 2x2 covariance per head and RoPE frequency pair for Q and K. These buffers are updated only during training and only when ORBIT enables collection.

`RotaryAttention.orbit_metrics` rotates the opposite-side covariance over the configured relative-position offsets. It returns one metric per head/frequency pair.

### Preconditioning

`inverse_metric_power` handles the symmetric 2x2 inverse power analytically. It avoids a general eigensolver, clamps the condition number, and returns identity if the auxiliary metric is invalid.

`Orbit._precondition_pair` reshapes each candidate update to

```text
[head, frequency, 2, input_dimension]
```

applies the local transform, then restores the joint Q/K Frobenius norm.

### Model interface

`Orbit` expects the model to provide:

```python
model.orbit_qk_pairs()
model.set_orbit_stat_collection(enabled)
```

Each entry returned by `orbit_qk_pairs()` must contain the Q parameter, K parameter, and an attention module exposing `orbit_metrics(...)`.

`OrbitGPT` is a complete reference implementation of that interface. Integrating ORBIT into another RoPE Transformer mainly requires exposing the same Q/K pairing and metric collection hooks.
