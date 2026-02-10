# bayesflow-calibration: Add-on Package for Calibrated NPE Training

## Context

**Problem**: BayesFlow 2.x trains normalizing flows via maximum likelihood (negative log-density). The resulting posteriors can be **miscalibrated** — credible intervals don't have the advertised coverage. Currently `rctbp_bf_training` only detects this post-training via `validation.py`.

**Solution**: Implement the differentiable calibration loss from [Falkner et al. (NeurIPS 2023)](https://arxiv.org/abs/2310.13402) as a **standalone add-on package** (`bayesflow-calibration`) that plugs into BayesFlow's training loop. The loss penalizes coverage errors during training, producing well-calibrated posteriors by construction.

**Why a separate package?**
- Reusable across projects (not just rctbp_bf_training)
- Clean dependency: `bayesflow-calibration` depends on `bayesflow>=2.0` + `keras>=3.0`
- `rctbp_bf_training` adds it as an optional dependency

---

## Integration Strategy: Subclass `ContinuousApproximator`

BayesFlow's training flow:
```
train_step() → compute_metrics(batch) → returns {"loss": tensor, ...} → loss.backward() → optimizer.step()
```

**Our approach**: Subclass `ContinuousApproximator`, override `compute_metrics()`:
```python
def compute_metrics(self, ...):
    metrics = super().compute_metrics(...)        # Base NLL loss
    cal_loss = self._calibration_loss(...)        # Our addition
    metrics["loss"] = metrics["loss"] + self.gamma_schedule.current * cal_loss
    metrics["calibration_loss"] = cal_loss
    return metrics
```

**Why not a Keras callback?** Callbacks run *after* `train_step` completes — they can't modify the loss for the current batch's gradient. Subclassing `compute_metrics` is the only way to inject a loss term into the same backward pass.

**Performance**: No slower than a native module. The subclass calls `super()` for the base loss (identical code path) and adds tensor operations for the calibration term. Same optimizer, same backward pass, one extra forward pass through the inference network for rank computation.

---

## Package Structure

```
bayesflow-calibration/
├── pyproject.toml
├── README.md
├── src/
│   └── bayesflow_calibration/
│       ├── __init__.py              # Public API exports
│       ├── approximator.py          # CalibratedContinuousApproximator
│       ├── losses.py                # calibration_loss, coverage_error, ste_indicator
│       ├── schedules.py             # GammaSchedule (constant, linear warmup, cosine)
│       └── diagnostics.py           # Training-time calibration monitoring callback
└── tests/
    ├── test_losses.py               # Unit tests for loss functions
    ├── test_approximator.py         # Integration tests with BayesFlow
    └── test_schedules.py            # Schedule tests
```

---

## Module Details

### 1. `losses.py` — Core Math (~120 lines)

**`ste_indicator(x)`** — Straight-Through Estimator
```python
def ste_indicator(x):
    """1.0 where x > 0, 0.0 elsewhere. Gradients flow through via STE."""
    hard = keras.ops.cast(x > 0, dtype=x.dtype)
    return x + keras.ops.stop_gradient(hard - x)
```
Backend-agnostic via `keras.ops`. Forward: hard indicator. Backward: identity.

**`compute_ranks(log_prob_true, log_probs_prior)`** — Differentiable ranking
- `log_prob_true`: `(batch, 1, [param_dim])` — log-density of true θ
- `log_probs_prior`: `(batch, n_rank_samples, [param_dim])` — log-density of prior samples
- Returns: `(batch, [param_dim])` — fractional rank in [0, 1]
- For multi-param: computes marginal ranks per parameter dimension
```python
def compute_ranks(log_prob_true, log_probs_prior):
    diff = log_probs_prior - log_prob_true           # (batch, n_samples, [dim])
    indicators = ste_indicator(diff)                  # STE: 1 where prior sample has higher density
    ranks = keras.ops.mean(indicators, axis=1)        # (batch, [dim]) fraction with higher density
    return ranks
```

**`coverage_error(ranks, mode=0.0, n_levels=20)`** — Calibration loss
- `ranks`: `(batch, [param_dim])` tensor
- `mode`: 0.0=conservativeness (ReLU), 1.0=calibration (both directions), or mixture
- Returns: scalar loss
```python
def coverage_error(ranks, mode=0.0, n_levels=20):
    # Sort ranks along batch dimension
    sorted_ranks = keras.ops.sort(ranks, axis=0)       # (batch, [dim])
    batch_size = keras.ops.shape(sorted_ranks)[0]

    # Expected coverage at each quantile
    expected = keras.ops.linspace(0.0, 1.0, batch_size)  # (batch,)
    # Broadcast for multi-param
    expected = keras.ops.expand_dims(expected, axis=-1) if sorted_ranks.ndim > 1 else expected

    # Empirical vs expected
    diff = sorted_ranks - expected

    if mode == 0.0:
        # Conservativeness: penalize under-coverage only
        error = keras.ops.relu(-diff)
    elif mode == 1.0:
        # Calibration: penalize both directions
        error = diff
    else:
        # Mixture
        error = (1 - mode) * keras.ops.relu(-diff) + mode * diff

    return keras.ops.mean(error ** 2)
```

### 2. `schedules.py` — Gamma Scheduling (~80 lines)

The `gamma` weight should be **schedulable** (user requirement):

```python
@dataclass
class GammaSchedule:
    """Schedule for calibration loss weight gamma."""
    schedule_type: str = "constant"   # "constant", "linear_warmup", "cosine", "step"
    gamma_max: float = 100.0
    warmup_epochs: int = 0            # Epochs before gamma reaches gamma_max
    total_epochs: int = 200           # For cosine schedule
    gamma_min: float = 0.0            # Floor value

    def __call__(self, epoch: int) -> float:
        if self.schedule_type == "constant":
            return self.gamma_max
        elif self.schedule_type == "linear_warmup":
            if epoch < self.warmup_epochs:
                return self.gamma_min + (self.gamma_max - self.gamma_min) * epoch / max(self.warmup_epochs, 1)
            return self.gamma_max
        elif self.schedule_type == "cosine":
            # Cosine annealing from gamma_min to gamma_max
            progress = min(epoch / max(self.total_epochs, 1), 1.0)
            return self.gamma_min + 0.5 * (self.gamma_max - self.gamma_min) * (1 + cos(pi * (1 - progress)))
        elif self.schedule_type == "step":
            return self.gamma_max if epoch >= self.warmup_epochs else self.gamma_min
```

Rationale: Start training with pure NLL (gamma=0) to learn a reasonable posterior first, then ramp up calibration pressure. Prevents calibration loss from dominating early when the network is random.

### 3. `approximator.py` — CalibratedContinuousApproximator (~200 lines)

```python
class CalibratedContinuousApproximator(bf.approximators.ContinuousApproximator):
    def __init__(
        self,
        *,
        prior_fn: Callable,              # (n_samples) -> (n_samples, param_dim)
        gamma_schedule: GammaSchedule = GammaSchedule(),
        calibration_mode: float = 0.0,   # 0=conservativeness, 1=calibration
        n_rank_samples: int = 100,       # Prior samples for ranking
        subsample_size: int | None = None,  # If set, subsample batch for cal loss
        n_coverage_levels: int = 20,
        **kwargs,                        # All ContinuousApproximator args
    ):
        super().__init__(**kwargs)
        self.prior_fn = prior_fn
        self.gamma_schedule = gamma_schedule
        self.calibration_mode = calibration_mode
        self.n_rank_samples = n_rank_samples
        self.subsample_size = subsample_size
        self.n_coverage_levels = n_coverage_levels
        self._current_epoch = 0
```

**`prior_fn` contract**: A callable that takes `(n_samples: int)` and returns a numpy array of shape `(n_samples, param_dim)` — i.i.d. samples from the **marginal prior** p(θ).

Following the original paper (Falkner et al.), we use a **global marginal prior**, not a per-observation conditional prior. The same set of prior samples is used for all observations in the batch. This is simpler and the SBC uniformity guarantee still holds in expectation.

For models with context-dependent priors (like ANCOVA where prior depends on `prior_df`, `prior_scale`), the marginal prior integrates over the meta-parameter distribution:
p(θ) = ∫ p(θ | prior_df, prior_scale) · p(prior_df, prior_scale) d(prior_df, prior_scale)

In practice, sample random meta-params first, then sample θ:
```python
def ancova_prior_fn(n_samples):
    samples = np.zeros((n_samples, 1))
    for j in range(n_samples):
        meta = meta_fn(config, rng)  # random prior_df, prior_scale
        samples[j, 0] = sample_t_or_normal(meta["prior_df"], meta["prior_scale"], rng)
    return samples
```

The prior samples are **broadcast** across the batch dimension inside `_calibration_loss()` — all observations share the same reference set.

**`compute_metrics()` override**:
```python
def compute_metrics(self, inference_variables, summary_variables=None,
                    inference_conditions=None, sample_weight=None, stage="training"):
    # 1. Get base metrics (NLL loss from normalizing flow)
    metrics = super().compute_metrics(
        inference_variables, summary_variables,
        inference_conditions, sample_weight, stage
    )

    # 2. Only add calibration during training
    if stage != "training":
        return metrics

    gamma = self.gamma_schedule(self._current_epoch)
    if gamma <= 0:
        return metrics

    # 3. Subsample batch if configured
    batch_size = keras.ops.shape(inference_variables)[0]
    if self.subsample_size and self.subsample_size < batch_size:
        idx = keras.random.shuffle(keras.ops.arange(batch_size))[:self.subsample_size]
        inf_vars_sub = keras.ops.take(inference_variables, idx, axis=0)
        cond_sub = keras.ops.take(inference_conditions, idx, axis=0) if inference_conditions is not None else None
        summ_sub = keras.ops.take(summary_variables, idx, axis=0) if summary_variables is not None else None
    else:
        inf_vars_sub, cond_sub, summ_sub = inference_variables, inference_conditions, summary_variables

    # 4. Compute calibration loss
    cal_loss = self._calibration_loss(inf_vars_sub, summ_sub, cond_sub)

    # 5. Add to total loss
    metrics["loss"] = metrics["loss"] + gamma * cal_loss
    metrics["calibration_loss"] = cal_loss
    metrics["gamma"] = gamma

    return metrics
```

**`_calibration_loss()` implementation**:
```python
def _calibration_loss(self, theta_true, summary_variables, inference_conditions):
    sub_batch = keras.ops.shape(theta_true)[0]

    # Build conditions for inference network (same as in super().compute_metrics)
    if self.summary_network is not None and summary_variables is not None:
        summary_out = self.summary_network(summary_variables, training=True)
        conditions = keras.ops.concatenate([inference_conditions, summary_out], axis=-1)
    else:
        conditions = inference_conditions

    # Log-prob of true θ given x
    log_prob_true = self.inference_network.log_prob(theta_true, conditions=conditions)
    # Shape: (sub_batch,) or (sub_batch, param_dim)

    # Sample from marginal prior (numpy → tensor), shared across batch
    prior_samples_np = self.prior_fn(self.n_rank_samples)  # (n_rank_samples, param_dim)
    prior_samples = keras.ops.convert_to_tensor(prior_samples_np, dtype=theta_true.dtype)

    # Broadcast prior: (1, n_rank_samples, param_dim) → (sub_batch, n_rank_samples, param_dim)
    prior_expanded = keras.ops.broadcast_to(
        keras.ops.expand_dims(prior_samples, axis=0),
        (sub_batch, self.n_rank_samples, keras.ops.shape(prior_samples)[-1])
    )

    # Broadcast conditions: (sub_batch, 1, cond_dim) → (sub_batch, n_rank_samples, cond_dim)
    cond_expanded = keras.ops.broadcast_to(
        keras.ops.expand_dims(conditions, axis=1),
        (sub_batch, self.n_rank_samples, keras.ops.shape(conditions)[-1])
    )

    # Flatten for batch forward pass through inference network
    prior_flat = keras.ops.reshape(prior_expanded, (-1, keras.ops.shape(prior_samples)[-1]))
    cond_flat = keras.ops.reshape(cond_expanded, (-1, keras.ops.shape(cond_expanded)[-1]))

    # Log-prob of all prior samples under each observation's posterior
    log_probs_prior_flat = self.inference_network.log_prob(prior_flat, conditions=cond_flat)
    log_probs_prior = keras.ops.reshape(log_probs_prior_flat, (sub_batch, self.n_rank_samples))

    # Compute ranks via STE
    log_prob_true_expanded = keras.ops.expand_dims(log_prob_true, axis=1)
    ranks = compute_ranks(log_prob_true_expanded, log_probs_prior)

    # Compute coverage error
    return coverage_error(ranks, mode=self.calibration_mode, n_levels=self.n_coverage_levels)
```

**Key simplification**: Prior samples are drawn once per batch and shared across all observations (broadcast). This reduces the prior sampling cost and matches the original paper's approach.

### 4. `diagnostics.py` — Training Monitor Callback (~50 lines)

```python
class CalibrationMonitorCallback(keras.callbacks.Callback):
    """Log calibration_loss and gamma to training history."""
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, '_current_epoch'):
            self.model._current_epoch = epoch
```

This callback updates the epoch counter so the gamma schedule works.

---

## Integration with `rctbp_bf_training`

In `rctbp_bf_training`, add `bayesflow-calibration` as optional dependency:

```toml
# pyproject.toml
[project.optional-dependencies]
calibration = ["bayesflow-calibration>=0.1"]
```

Usage in `create_optimization_objective()` or `train_until_threshold()`:

```python
from bayesflow_calibration import CalibratedContinuousApproximator, GammaSchedule

# Instead of bf.BasicWorkflow(...), build approximator directly:
approximator = CalibratedContinuousApproximator(
    inference_network=inference_net,
    summary_network=summary_net,
    prior_fn=my_prior_fn,
    gamma_schedule=GammaSchedule(schedule_type="linear_warmup", gamma_max=100, warmup_epochs=20),
    calibration_mode=0.0,
    n_rank_samples=100,
    subsample_size=80,
)
approximator.compile(optimizer=opt)
approximator.fit(simulator=simulator, ...)
```

---

## Multi-Parameter Support

The implementation supports arbitrary `param_dim` from the start:
- `inference_network.log_prob()` returns per-dimension or scalar log-probs
- `compute_ranks()` works per-dimension when log-probs are multi-dimensional
- `coverage_error()` averages across parameter dimensions
- `prior_fn` returns `(batch, n_samples, param_dim)` for any `param_dim`

For 1D (ANCOVA `b_group`): `param_dim=1`, ranks are scalar per item.
For multi-dim: marginal calibration per dimension, averaged into single loss.

---

## Performance & Scheduling

| Config | Overhead | Use case |
|--------|----------|----------|
| `n_rank_samples=50, subsample_size=32` | ~2x | Fast experimentation |
| `n_rank_samples=100, subsample_size=80` | ~6x | **Recommended default** |
| `n_rank_samples=200, subsample_size=None` | ~60x | Maximum calibration signal |

**Scheduling strategies** (via `GammaSchedule`):
- `constant`: Fixed gamma throughout (simple baseline)
- `linear_warmup`: Start at 0, ramp to gamma_max over N epochs (recommended)
- `cosine`: Smooth cosine ramp-up
- `step`: Binary on/off after warmup_epochs

Recommended: `linear_warmup` with `warmup_epochs=20-50` so the network first learns a reasonable posterior via NLL, then the calibration loss refines coverage.

---

## Implementation Order

1. **`losses.py`** — Pure math, no BayesFlow dependency for core functions. Test with synthetic tensors.
2. **`schedules.py`** — Standalone dataclass + tests.
3. **`approximator.py`** — Core integration. Requires BayesFlow for testing.
4. **`diagnostics.py`** — Simple callback.
5. **`__init__.py`** — Public API.
6. **`tests/`** — Unit + integration tests.
7. **Integration into `rctbp_bf_training`** — Add optional dependency, update optimization objective.

---

## Verification Plan

1. **Unit tests (`test_losses.py`)**:
   - `ste_indicator`: correct forward values, gradients exist and are nonzero
   - `compute_ranks`: uniform prior samples → ranks ≈ uniform
   - `coverage_error`: uniform ranks → loss ≈ 0; degenerate ranks → loss > 0
   - Modes: conservativeness vs calibration produce different losses for same input

2. **Unit tests (`test_schedules.py`)**:
   - Constant schedule returns fixed value
   - Linear warmup: 0 at epoch 0, gamma_max at warmup_epochs
   - Cosine schedule is smooth and bounded

3. **Integration tests (`test_approximator.py`)**:
   - `CalibratedContinuousApproximator` can be instantiated with mock prior_fn
   - `compute_metrics` returns dict with `calibration_loss` and `gamma` keys
   - Training runs for 2 epochs without error (smoke test)
   - `gamma=0` produces identical loss to base approximator

4. **End-to-end** (in `rctbp_bf_training` notebook):
   - Train ANCOVA model with `CalibratedContinuousApproximator`
   - Compare coverage metrics vs standard training
   - Verify `calibration_loss` metric appears in training history and decreases

---

## Files Summary

| File | Lines | Description |
|------|-------|-------------|
| `src/bayesflow_calibration/__init__.py` | ~15 | Public API exports |
| `src/bayesflow_calibration/losses.py` | ~120 | `ste_indicator`, `compute_ranks`, `coverage_error` |
| `src/bayesflow_calibration/schedules.py` | ~80 | `GammaSchedule` dataclass |
| `src/bayesflow_calibration/approximator.py` | ~200 | `CalibratedContinuousApproximator` |
| `src/bayesflow_calibration/diagnostics.py` | ~50 | `CalibrationMonitorCallback` |
| `tests/test_losses.py` | ~100 | Loss function unit tests |
| `tests/test_approximator.py` | ~80 | Integration tests |
| `tests/test_schedules.py` | ~50 | Schedule tests |
| `pyproject.toml` | ~40 | Package config |
