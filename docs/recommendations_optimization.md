# Recommendations for Speeding Up Simulation and Inference

## 1. R-Side Simulation Optimizations

### 1.1 Avoid repeated allocation of large matrices

Reuse large matrices (outcome, covariate, group) instead of allocating
new ones for each value of `n_total`.

### 1.2 Use faster RNG functions

Replace: - `sample(0:1, ...)` with `rbinom()` for binary arm
assignment. - `rnorm(total_elements)` with faster bulk RNG from `Rfast`
if available.

### 1.3 Avoid unnecessary copies from `matrix()`

Use:

    x <- rnorm(total_elements)
    dim(x) <- c(n_sims, n_total)

This avoids internal copying.

### 1.4 Eliminate temporary intermediate matrices

Compute the outcome directly without allocating `mean_mat`.

------------------------------------------------------------------------

## 2. Reduce Reticulate Conversion Overhead

### 2.1 Convert R matrices to NumPy arrays only once

Convert full matrices to NumPy once and pass NumPy slices to BayesFlow.
Avoid per-batch `r_to_py()` calls.

### 2.2 Avoid repeated concatenation in Python

Preallocate a full NumPy results array and fill slices directly.

### 2.3 Optionally move simulation into Python

A NumPy-based simulation eliminates R→Python data movement and is
generally faster.

------------------------------------------------------------------------

## 3. BayesFlow Inference Improvements

### 3.1 Maximize batch size

Larger batches reduce fixed overhead of each model call.

### 3.2 Ensure no retracing of the model

If the model retraces on every call, ensure it is traced once or wrapped
in a static inference function.

------------------------------------------------------------------------

## 4. R-Side Postprocessing

Postprocessing is already efficient using `matrixStats`. Only small
gains remain.

------------------------------------------------------------------------

## 5. Highest Impact Changes

1.  Use `rbinom()` for group assignment.
2.  Avoid `matrix()` copying and use `dim(x) <- ...`.
3.  Remove intermediate temporary matrices.
4.  Convert R matrices to NumPy once; slice in Python.
5.  Preallocate final NumPy results array.
6.  Consider implementing simulation in NumPy for maximal speed.
