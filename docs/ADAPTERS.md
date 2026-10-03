# Adapting ORBIT to other RoPE models

ORBIT is intentionally **function-aware**: its update rule depends on what a parameter does in the model, not only on the tensor's shape. That is why a completely automatic "wrap any Transformer" adapter would be unsafe.

For ORBIT to construct the same optimizer geometry as the paper, it must know all of the following:

1. which two matrices form a Query/Key pair;
2. how each projection output is partitioned into attention heads;
3. how each head is partitioned into RoPE frequency pairs;
4. the RoPE frequency convention/base used by that attention module;
5. where the **unrotated** Q/K projection outputs can be observed;
6. which 2-D parameters should remain on the auxiliary AdamW path rather than Muon's spectral path.

Those are architectural semantics. Two tensors can have identical shapes while implementing different computations, so ORBIT deliberately does not guess them from parameter names or dimensions alone. Making these choices explicit is also consistent with the broader research motivation of function-aware optimization: future optimizers may need different local geometries for different computational roles.

## Generic adapter

The package provides `OrbitAdapter` and `QKPairSpec` for external RoPE models whose attention modules expose separate `nn.Linear` query and key projections.

```python
from orbit import Orbit, OrbitAdapter, QKPairSpec

base_model = ...

pairs = []
for layer in base_model.layers:
    attn = layer.self_attn
    pairs.append(
        QKPairSpec(
            q_proj=attn.q_proj,
            k_proj=attn.k_proj,
            n_head=base_model.config.num_attention_heads,
            rope_base=getattr(base_model.config, "rope_theta", 10_000.0),
            name="self_attn",
        )
    )

model = OrbitAdapter(base_model, pairs)
model = model.to(device)

optimizer = Orbit(
    model,
    lr=0.02,
    adamw_lr=3e-4,
    weight_decay=0.05,
)

loss = model(input_ids, labels=input_ids).loss
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Use the **adapter wrapper** for forward passes after construction. The underlying model is available as `model.model`.

If the external model implements Hugging Face-style `get_input_embeddings()` and `get_output_embeddings()`, the adapter automatically keeps those weights on ORBIT's auxiliary AdamW path. Otherwise pass them explicitly:

```python
model = OrbitAdapter(
    base_model,
    pairs,
    aux_decay_parameters=[base_model.embed_tokens, base_model.lm_head],
)
```

## What the adapter records

Forward hooks on each declared `q_proj` and `k_proj` collect the unrotated projection outputs before RoPE is applied. For every declared attention pair, the adapter stores persistent per-head, per-frequency `2 x 2` Q/K second moments and exposes the same `orbit_metrics(...)` contract as the reference `OrbitGPT` implementation.

The optimizer then performs exactly the same high-level path as in the paper:

```text
Muon Q/K candidate
  -> opposite-side Q/K second moments
  -> optimizer-side R(+Delta) transport
  -> local inverse-square-root preconditioning
  -> joint Q/K Frobenius restoration
  -> parameter update
```

The adapter also preserves the distinction between the forward RoPE score, which contains `R(-Delta)`, and ORBIT's optimizer-side transport, which uses `R(+Delta)`.

## Supported generic-adapter boundary

The v0.1 generic adapter deliberately targets the geometry actually evaluated in the paper:

- separate `nn.Linear` Q and K projections;
- ordinary multi-head attention with equal Q/K projection widths;
- even per-head dimension;
- standard pairwise RoPE frequency layout;
- Q/K projection outputs shaped `[batch, time, width]`.

It rejects ambiguous GQA/MQA Q/K width mappings instead of silently choosing one. Fused-QKV attention, grouped-query attention, query/key normalization inserted before RoPE, unusual RoPE layouts, recurrent/shared attention modules, or architectures that expose Q/K only after an additional transformation should use a **native integration** or a purpose-built adapter.

That restriction is intentional. A function-aware optimizer should not pretend that architectural semantics are interchangeable when they are not.

## Gradient checkpointing

The reference `OrbitGPT` has a native checkpoint-aware integration that suppresses duplicate statistic updates during recomputation.

The generic adapter deduplicates the normal one-attention-module-per-top-level-forward case by associating statistics with the adapter's top-level forward call. This is sufficient for the common forward/backward loop used by the reference path, but models with unusual shared-layer execution, delayed backward across multiple top-level forwards, or custom checkpoint engines should prefer a native integration and explicitly use `suspend_orbit_stat_collection()` around recomputation.

For experiments intended as publication evidence, verify the statistic-update count for the target architecture rather than assuming checkpoint behavior is identical across frameworks.

## Native integration contract

For maximum control, a model can implement ORBIT directly instead of using hooks. The optimizer expects:

```python
model.orbit_qk_pairs()
model.set_orbit_stat_collection(enabled)
```

Each Q/K pair must provide

```python
{
    "q": q_parameter,
    "k": k_parameter,
    "module": metric_provider,
}
```

where `metric_provider` exposes

```python
metric_provider.n_head
metric_provider.n_freq
metric_provider.orbit_metrics(deltas, rotate=True, eps=...)
```

A model may additionally expose

```python
model.orbit_aux_decay_parameters()
```

to identify 2-D parameters such as embeddings or tied output heads that should use the auxiliary AdamW path.

`OrbitGPT` is the reference native integration; `OrbitAdapter` is the reusable bridge for compatible external models.

## Research direction

The adapter is intentionally explicit because ORBIT is only one example of a larger design space. A normalization scale, routing matrix, Q/K projection, convolutional kernel, or recurrent state transition each participates in a different computation. Function-aware optimization asks whether the optimizer should exploit those differences rather than treating all same-shaped tensors identically.

The repository therefore treats adapters not merely as compatibility glue, but as a place to encode and test architectural hypotheses. Extensions for GQA/MQA, QK normalization, learned/modified RoPE, MoE routing, or other parameter-specific geometries should make their semantic mapping explicit and add regression tests for the claimed computation.
