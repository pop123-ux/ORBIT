# Training

Only ORBIT-specific training details are listed here.

## Reference construction

```python
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
```

The `deltas` tuple contains non-negative causal distances $i-j$.

## Scheduling

`adamw_lr` is converted at construction time into a fixed ratio to `lr`:

```math
r_{\mathrm{aux}}
=
\frac{\eta_{\mathrm{aux},0}}{\eta_{\mathrm{matrix},0}}.
```

The effective auxiliary learning rate at every step is

```math
\eta_{\mathrm{aux},t}
=
r_{\mathrm{aux}}\eta_{\mathrm{matrix},t}.
```

Therefore ordinary PyTorch schedulers work as expected:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=0.1 * optimizer.param_groups[0]["lr"],
)

# optimizer.step()
# scheduler.step()
```

Use `optimizer.auxiliary_lr()` when the current effective auxiliary rate needs to be logged.

## Gradient checkpointing

The reference model can enable

```python
OrbitGPTConfig(gradient_checkpointing=True)
```

without changing the Q/K EMA cadence. Checkpoint recomputation is run under an ORBIT-statistics suspension context, so each logical forward/backward contributes one statistics update per layer.

## DDP

No special optimizer call is required. If `torch.distributed` is initialized, each attention layer globally reduces its unnormalized Q/K second moments and sample count before applying the EMA.

All ranks must use the same ORBIT model structure and the same `deltas`.

## Checkpoint/resume

The ORBIT covariance buffers and `orbit_stats_seen` are persistent model state. Save the model and optimizer together:

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
}
torch.save(checkpoint, path)
```

and restore the model before continuing optimization:

```python
state = torch.load(path, map_location="cpu")
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
scheduler.load_state_dict(state["scheduler"])
```

## Diagnostics

```python
metrics = optimizer.diagnostics()
```

returns the accumulated mean metric condition numbers and joint restoration factor. Calling it performs the device-to-host synchronization; the training step itself does not call `.cpu()` for these diagnostics.

## Smoke run

```bash
python examples/quickstart.py
```

The example repeats a deterministic next-token pattern so that the printed loss demonstrates learning rather than measuring fresh random targets at every step.
