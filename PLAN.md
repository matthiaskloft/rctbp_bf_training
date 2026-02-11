# Implementation Plan: Targeted Improvements

## Overview

Four improvements, implemented as **two PRs** to keep reviews manageable:

- **PR 1** (structural): Split `optimization.py`, clean up `validation.py`, consolidate `diagnostics.py`
- **PR 2** (tests): Comprehensive unit tests for all modules

PR 1 first because the module split changes imports, and tests should be written against the final structure.

---

## PR 1: Structural Refactoring

### Step 1: Split `optimization.py` (1,612 LOC → 5 focused modules)

Create a git worktree `refactor-modules`, then split into:

#### `core/objectives.py` (~150 LOC)
Extract objective computation and parameter normalization:
- `PARAM_COUNT_LOG_SCALE` constant
- `FAILED_TRIAL_CAL_ERROR`, `FAILED_TRIAL_PARAM_SCORE` constants
- `get_param_count()`
- `estimate_param_count()`
- `compute_composite_objective()`
- `normalize_param_count()`
- `denormalize_param_count()`
- `extract_objective_values()`

#### `core/results.py` (~320 LOC)
Extract results analysis and optimization visualization:
- `get_pareto_trials()`
- `trials_to_dataframe()`
- `summarize_best_trials()`
- `plot_optimization_results()`
- `plot_pareto_front()`

These import `denormalize_param_count` from `objectives.py`.

#### `core/threshold.py` (~270 LOC)
Extract threshold-based training (orthogonal to Optuna):
- `QualityThresholds` dataclass
- `check_thresholds()`
- `train_until_threshold()`
- `create_strict_validation_grid()`

These import `cleanup_trial` from `optimization.py`.

#### `core/dashboard.py` (~120 LOC)
Extract dashboard CLI (zero optimization logic dependencies):
- `launch_dashboard()`
- `_cli_main()`
- `if __name__ == "__main__"` block

#### `core/optimization.py` (~600 LOC, reduced)
Keep the core optimization logic:
- `create_study()`
- `run_optimization()`
- `create_optimization_objective()`
- `sample_hyperparameters()`
- `HyperparameterSpace` dataclass
- `OptunaReportCallback` class
- `cleanup_trial()`

**Plus backward-compat re-exports** at the bottom:
```python
# Backward compatibility: re-export from split modules
from rctbp_bf_training.core.objectives import (  # noqa: F401
    PARAM_COUNT_LOG_SCALE, FAILED_TRIAL_CAL_ERROR, FAILED_TRIAL_PARAM_SCORE,
    get_param_count, estimate_param_count,
    compute_composite_objective, normalize_param_count,
    denormalize_param_count, extract_objective_values,
)
from rctbp_bf_training.core.results import (  # noqa: F401
    get_pareto_trials, trials_to_dataframe, summarize_best_trials,
    plot_optimization_results, plot_pareto_front,
)
from rctbp_bf_training.core.threshold import (  # noqa: F401
    QualityThresholds, check_thresholds, train_until_threshold,
    create_strict_validation_grid,
)
from rctbp_bf_training.core.dashboard import launch_dashboard  # noqa: F401
```

#### Update `core/__init__.py`
Add wildcard imports for new modules:
```python
from rctbp_bf_training.core.objectives import *
from rctbp_bf_training.core.results import *
from rctbp_bf_training.core.threshold import *
from rctbp_bf_training.core.dashboard import *
```

The top-level `__init__.py` stays unchanged — it imports from `core.optimization` which re-exports everything.

### Step 2: Clean up `validation.py`

1. **Fix bug at line 817**: Change `conditions_list` → `condition_grid` (the parameter is named `condition_grid` in the function signature).

2. **Add deprecation warning to `extract_calibration_metrics()`**:
   ```python
   import warnings
   warnings.warn(
       "extract_calibration_metrics() is deprecated, "
       "use run_validation_pipeline() instead.",
       DeprecationWarning,
       stacklevel=2,
   )
   ```

### Step 3: Consolidate `diagnostics.py`

1. **Remove dead `plot_coverage_diff` (lines 22-125)**: The second definition (line 847) overrides it. The second version is strictly better (handles flexible input shapes, returns Axes). Remove the first; keep the second and move it to the top of the file after imports.

2. **Extract `_create_condition_grid()` helper**: The 5 grid-plot functions (`plot_sbc_by_condition`, `plot_recovery_by_condition`, `plot_coverage_by_condition`, `plot_histogram_by_condition`, `plot_ecdf_by_condition`) all repeat:
   - Get unique conditions, limit to max_conditions
   - Compute n_cols/n_rows
   - Create fig + axes with figsize
   - `np.atleast_2d(axes)` handling
   - Hide empty subplots

   Extract to:
   ```python
   def _create_condition_grid(
       n_conditions: int,
       max_conditions: int = 16,
       max_cols: int = 4,
       figsize_per_plot: tuple = (3, 3),
       suptitle: str = "",
   ) -> Tuple[plt.Figure, np.ndarray, int, int]:
       """Create a subplot grid for condition-level plots.

       Returns (fig, axes_2d, n_rows, n_cols).
       """
   ```

   Each grid function then becomes ~15-20 LOC shorter.

### Step 4: Verify everything works
- Run `ruff check src/`
- Run `mypy src/`
- Run `pytest` (existing test_package.py should still pass)
- Verify all existing imports work (test that `from rctbp_bf_training.core.optimization import X` still works for all X)

---

## PR 2: Comprehensive Unit Tests

### `tests/test_core/test_utils.py` (~80 LOC)
- `test_loguniform_int_bounds`: output in [low, high]
- `test_loguniform_int_alpha_skew`: alpha=0.5 skews low, alpha=2.0 skews high (statistical test over 1000 samples)
- `test_loguniform_float_bounds`: output in [low, high]
- `test_loguniform_float_reproducibility`: same rng seed → same result
- `test_sample_t_or_normal_df_zero`: df=0 → Normal
- `test_sample_t_or_normal_df_positive`: df=5 → Student-t (heavier tails)
- `test_sample_t_or_normal_df_large`: df=200 → Normal
- `test_moving_average_early_stopping_patience`: stops after patience epochs of no improvement (mock Keras model with `get_weights`/`set_weights`/`stop_training`)
- `test_moving_average_early_stopping_window`: moving average smooths noise
- `test_moving_average_early_stopping_restore_weights`: best weights restored

### `tests/test_core/test_infrastructure.py` (~120 LOC)
- `test_summary_network_config_defaults`: verify default values
- `test_inference_network_config_defaults`: verify default values
- `test_training_config_defaults`: verify default values
- `test_workflow_config_roundtrip`: `to_dict()` → `from_dict()` preserves all fields
- `test_workflow_config_partial_dict`: `from_dict({})` uses defaults
- `test_adapter_spec_fields`: verify required/optional fields
- `test_params_dict_to_workflow_config`: known params dict → correct WorkflowConfig
- `test_save_load_metadata_roundtrip`: save metadata JSON to tmp dir, load it back, verify contents

### `tests/test_core/test_objectives.py` (NEW, ~90 LOC)
- `test_normalize_param_count_known_values`: 10K→~0.67, 100K→~0.83, 1M→1.0
- `test_normalize_param_count_inverse`: normalize → denormalize roundtrip
- `test_denormalize_param_count_known_values`: inverse of above
- `test_extract_objective_values`: given mock metrics dict + param count, returns expected tuple
- `test_compute_composite_objective`: known inputs → expected weighted sum
- `test_get_param_count_returns_negative_one`: pass mock unbuilt model
- `test_estimate_param_count`: known config → reasonable estimate range

### `tests/test_core/test_optimization.py` (~100 LOC)
- `test_hyperparameter_space_defaults`: HyperparameterSpace() has sensible ranges
- `test_sample_hyperparameters`: mock Optuna trial, verify all keys present
- `test_create_study_single_objective`: creates study with correct direction
- `test_create_study_multi_objective`: creates study with 2 directions
- `test_optuna_report_callback_init`: callback initializes correctly
- `test_cleanup_trial_no_gpu`: runs without error when torch/tf not available

### `tests/test_core/test_results.py` (NEW, ~70 LOC)
- `test_trials_to_dataframe_empty`: empty study → empty DataFrame
- `test_trials_to_dataframe_with_trials`: mock completed trials → correct DataFrame shape
- `test_summarize_best_trials`: mock study → sorted by cal_error
- `test_get_pareto_trials`: wrapper returns study.best_trials

### `tests/test_core/test_threshold.py` (NEW, ~60 LOC)
- `test_quality_thresholds_defaults`: verify default values
- `test_check_thresholds_all_pass`: good metrics → all pass
- `test_check_thresholds_partial_fail`: one metric fails → not passed
- `test_create_strict_validation_grid`: returns expected number of conditions (product of all parameter ranges)
- `test_create_strict_validation_grid_has_required_keys`: each condition has id_cond, N, p_alloc, etc.

### `tests/test_core/test_validation.py` (~120 LOC)
- `test_compute_sbc_uniformity_tests_uniform`: uniform ranks → high p-values
- `test_compute_sbc_uniformity_tests_nonuniform`: biased ranks → low p-values
- `test_compute_sbc_uniformity_tests_empty`: empty array → NaN results
- `test_compute_sbc_c2st_uniform`: uniform ranks → accuracy ~0.5
- `test_compute_sbc_c2st_biased`: biased ranks → accuracy > 0.6
- `test_compute_batch_metrics_shapes`: verify output DataFrame columns and row count
- `test_compute_batch_metrics_coverage`: known draws → correct coverage
- `test_aggregate_metrics_structure`: verify output dict has required keys
- `test_aggregate_metrics_recovery`: synthetic perfect recovery → r≈1.0
- `test_extract_calibration_metrics_deprecation_warning`: verify DeprecationWarning is raised

### `tests/test_models/test_ancova/test_model.py` (~120 LOC)
- `test_prior_config_defaults`: PriorConfig() default values
- `test_meta_config_defaults`: MetaConfig() default values
- `test_ancova_config_roundtrip`: `to_dict()` → `from_dict()` preserves all fields
- `test_prior_output_shape`: prior() returns dict with shape-(1,) arrays
- `test_prior_output_keys`: prior() returns b_covariate and b_group
- `test_likelihood_output_shape`: likelihood() returns correct shapes given N
- `test_likelihood_output_keys`: outcome, covariate, group keys present
- `test_meta_output_keys`: meta() returns N, p_alloc, prior_df, prior_scale
- `test_meta_ranges`: meta() values within MetaConfig ranges
- `test_simulate_cond_batch_shapes`: batch simulation → correct array shapes
- `test_create_validation_grid`: verify grid size and required keys
- `test_get_ancova_adapter_spec`: returns AdapterSpec with correct set_keys, param_keys, context_keys

### `tests/test_plotting/test_diagnostics.py` (~80 LOC)
- `test_plot_coverage_diff_returns_axes`: returns matplotlib Axes
- `test_plot_coverage_diff_with_ax`: passes existing ax, returns same ax
- `test_plot_sbc_rank_histogram_returns_axes`: returns Axes
- `test_plot_sbc_ecdf_diff_returns_axes`: returns Axes
- `test_plot_recovery_returns_axes`: returns Axes
- `test_plot_sbc_by_condition_returns_figure`: returns Figure
- `test_plot_histogram_by_condition_returns_figure`: returns Figure
- `test_create_condition_grid_dimensions`: helper returns correct grid dims
- Close all figures in teardown to avoid memory leaks (`plt.close('all')`)

### Test infrastructure
- Use `pytest` fixtures for common test data (rng, sample draws, condition grids)
- Use `matplotlib.use('Agg')` backend in conftest.py for headless plotting
- Mock Keras model for MovingAverageEarlyStopping tests (simple object with `get_weights`/`set_weights`/`stop_training`)
- Mock Optuna trial for hyperparameter sampling tests
- All tests must run without GPU/BayesFlow model training

---

## Implementation Order

1. Create worktree for PR 1
2. Split optimization.py → objectives.py, results.py, threshold.py, dashboard.py
3. Update core/__init__.py
4. Add backward-compat re-exports in optimization.py
5. Fix validation.py bug (line 817) + add deprecation warning
6. Consolidate diagnostics.py (remove dead code, extract grid helper)
7. Run ruff + mypy + pytest
8. Commit & create PR 1
9. Create worktree for PR 2 (branched from PR 1)
10. Write tests/conftest.py with shared fixtures
11. Write all test files
12. Run pytest, iterate until green
13. Commit & create PR 2
