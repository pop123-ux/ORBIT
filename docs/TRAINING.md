# Training

Only ORBIT-specific training behavior is documented here. Final paper hyperparameters are intentionally not frozen in this file until the corrected post-audit campaign has been rerun.

## Construction

```python
optimizer = Orbit(
    model,
    lr=0.02,
    adamw_lr=3e-4,
    weight_decay=0.05,
    momentum=0.95,
    ns_steps=5,
    functional_power=1.0,
    metric_eps=1e-5,
    metric_condition_cap=100.0,
    deltas=(1, 2, 4, 8, 16, 32, 64, 128),
)
```

`deltas` are non-negative causal distances $\Delta=i-j$; the metric transports them with $R(-\Delta)$.

## Scheduling

At construction time,

```math
r_{\mathrm{aux}}
=
\frac{\eta_{\mathrm{aux},0}}
     {\eta_{\mathrm{matrix},0}}
```

is stored as a fixed ratio. At step $t$,

```math
\eta_{\mathrm{aux},t}
=
r_{\mathrm{aux}}\eta_{\mathrm{matrix},t}.
```

Therefore a normal PyTorch scheduler can operate on the optimizer’s `lr` field:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=0.1 * optimizer.param_groups[0]["lr"],
)
```

The current auxiliary rate is available through `optimizer.auxiliary_lr()`.

## Gradient checkpointing

Enable checkpointing in the reference model with

```python
OrbitGPTConfig(gradient_checkpointing=True)
```

Checkpoint recomputation runs with ORBIT second-moment collection suspended, so the EMA cadence is unchanged.

## DDP

When `torch.distributed` is initialized, each attention layer reduces its unnormalized Q/K second-moment sums and sample count across ranks before applying the EMA. No separate post-step metric synchronization is required.

## Checkpoint/resume

The Q/K second moments and `orbit_stats_seen` are persistent model buffers. Save model and optimizer state together:

```python
torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    },
    path,
)
```

Restore the model state before continuing optimization:

```python
state = torch.load(path, map_location="cpu")
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
scheduler.load_state_dict(state["scheduler"])
```

`Orbit.load_state_dict` also migrates the pre-audit optimizer metadata that stored an independent `adamw_lr` field.

## Diagnostics

```python
metrics = optimizer.diagnostics()
```

performs the device-to-host conversion for the accumulated metric-condition and joint-restoration diagnostics. The training step itself does not transfer those diagnostics to CPU.

## Smoke run

```bash
python examples/quickstart.py
```

The example uses a deterministic next-token pattern so its loss can demonstrate learning.
