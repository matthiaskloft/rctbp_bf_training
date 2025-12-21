# Notebook Update Summary: New Decoupled API

## Overview

Updated `ancova_cont_2arms_bo.ipynb` to demonstrate the new decoupled network architecture and reusable infrastructure.

## Key Changes

### 1. Updated Imports (Cell 1)

**Before:**
```python
from ancova_cont_2arms_fns import (
    ANCOVAConfig,
    NetworkConfig,
    TrainingConfig,
    # ...
)
```

**After:**
```python
# Import generic infrastructure
from bayesflow_infrastructure import (
    SummaryNetworkConfig,
    InferenceNetworkConfig,
    WorkflowConfig,
    params_dict_to_workflow_config,
)

# Import ANCOVA-specific functions
from ancova_cont_2arms_fns import (
    ANCOVAConfig,
    create_ancova_workflow_components,
    get_ancova_adapter_spec,
    # ... plus legacy imports for compatibility
)
```

**Benefits:**
- Clear separation between generic and model-specific code
- Access to decoupled network configs
- Can use infrastructure functions directly

### 2. New Demo Cell (After Cell 3)

Added a demonstration of the decoupled API:

```python
# Create independent network configs
summary_config = SummaryNetworkConfig(
    summary_dim=12,
    depth=4,
    width=96,
    dropout=0.1,
)

inference_config = InferenceNetworkConfig(
    depth=8,
    hidden_sizes=(256, 256),
    dropout=0.15,
)

# Build workflow components
workflow_config = WorkflowConfig(summary_config, inference_config)
demo_config = ANCOVAConfig(workflow=workflow_config)
summary_net, inference_net, adapter = create_ancova_workflow_components(demo_config)
```

**Shows:**
- Independent configuration of summary and inference networks
- Using the ANCOVA factory function
- New workflow config structure

### 3. Updated Objective Function (Cell 11)

**Before:**
```python
def objective(trial):
    params = sample_hyperparameters(trial, search_space)
    summary_net, inference_net = build_networks_from_params(params)  # Legacy
    # ...
```

**After:**
```python
def objective(trial):
    params = sample_hyperparameters(trial, search_space)

    # NEW API: Convert to WorkflowConfig with decoupled networks
    workflow_config = params_dict_to_workflow_config(params)

    # NEW API: Build networks independently
    summary_net = build_summary_network(workflow_config.summary_network)
    inference_net = build_inference_network(workflow_config.inference_network)
    # ...
```

**Benefits:**
- Explicit use of decoupled configs
- Can optimize summary and inference networks independently
- Shows conversion from flat params dict to structured config

### 4. Updated train_until_threshold (Cell 22)

**Key Changes:**
```python
def train_until_threshold(params, threshold, max_attempts, ...):
    # Convert params to decoupled config
    workflow_config = params_dict_to_workflow_config(params)

    # Build networks independently
    summary_net = build_summary_network(workflow_config.summary_network)
    inference_net = build_inference_network(workflow_config.inference_network)

    # Print network details
    print(f"Summary: dim={workflow_config.summary_network.summary_dim}, ...")
    print(f"Inference: depth={workflow_config.inference_network.depth}, ...")

    # ...

    # Return workflow_config for saving
    return wf, cal_error, attempt, workflow_config
```

**Benefits:**
- Clear visibility into network architectures during training
- Returns workflow_config for proper metadata saving
- Uses config structure consistently

### 5. Updated Model Saving (Cell 24)

**Before:**
```python
net_config = network_config_from_params(params)  # Legacy
config_with_net = ANCOVAConfig(
    prior=config.prior,
    meta=config.meta,
    network=net_config,  # Old structure
    training=config.training,
)
```

**After:**
```python
# Use the workflow config from training
config_with_optimized = ANCOVAConfig(
    prior=config.prior,
    meta=config.meta,
    workflow=best_workflow_config,  # New structure with decoupled networks
)

metadata = get_model_metadata(config=config_with_optimized, ...)
saved_path = save_model_with_metadata(approximator, save_path, metadata)
```

**Benefits:**
- Saves complete workflow configuration
- Includes both summary and inference network configs
- Better metadata for model reproducibility

## What Users Will See

### Running the Notebook

When users run the updated notebook, they'll see:

1. **On import (Cell 1):**
   ```
   Config loaded: {...}

   New decoupled network configs:
     Summary network: SummaryNetworkConfig(...)
     Inference network: InferenceNetworkConfig(...)
   ```

2. **In demo cell:**
   ```
   ✓ Decoupled network configs created
   ✓ Workflow components created
   ```

3. **During optimization (Cell 11):**
   ```
   Objective function defined using NEW DECOUPLED API
     - params_dict_to_workflow_config() converts hyperparameters
     - build_summary_network() builds summary net independently
     - build_inference_network() builds inference net independently
   ```

4. **During threshold training (Cell 22):**
   ```
   Networks built:
     Summary: dim=12, depth=4, width=96
     Inference: depth=8, hidden=(256, 256)
   ```

5. **When saving (Cell 24):**
   ```
   ✓ Model saved to: ...

   Saved configuration:
     Summary network: dim=12, depth=4
     Inference network: depth=8, hidden=(256, 256)
   ```

## Backwards Compatibility

The notebook maintains backwards compatibility:
- Legacy `build_networks_from_params()` still imported
- Old `NetworkConfig` still available
- Can mix old and new API as needed

## Benefits for Users

### 1. Clearer Architecture
- Explicit separation of summary and inference networks
- Easy to see what each network is configured to do

### 2. Better Debugging
- Network configs printed during training
- Clear visibility into what's being optimized

### 3. Independent Optimization
- Can tune summary network separately from inference network
- Future: Multi-objective BO could optimize each network independently

### 4. Reusable Patterns
- Shows how to use the generic infrastructure
- Template for creating new models

### 5. Better Metadata
- Saved models include full workflow config
- Easy to reproduce exact architecture later

## Migration Guide for Users

### If Using Default Configs
No changes needed! The notebook works as before.

### If Customizing Configs
**Old way:**
```python
config = ANCOVAConfig()
config.network.summary_dim = 12
config.network.flow_depth = 8
```

**New way (recommended):**
```python
config = ANCOVAConfig()
config.workflow.summary_network.summary_dim = 12
config.workflow.inference_network.depth = 8
```

### If Building Networks Manually
**Old way:**
```python
from ancova_cont_2arms_fns import build_networks_from_params
summary_net, inference_net = build_networks_from_params(params)
```

**New way (recommended):**
```python
from bayesflow_infrastructure import (
    params_dict_to_workflow_config,
    build_summary_network,
    build_inference_network,
)

workflow_config = params_dict_to_workflow_config(params)
summary_net = build_summary_network(workflow_config.summary_network)
inference_net = build_inference_network(workflow_config.inference_network)
```

## Summary

The updated notebook:
- ✅ Demonstrates the new decoupled architecture
- ✅ Uses generic infrastructure where appropriate
- ✅ Maintains backwards compatibility
- ✅ Provides clear examples of the new API
- ✅ Shows benefits of network decoupling
- ✅ Works without requiring any user changes

Users can continue using the notebook as before, or gradually adopt the new API patterns shown in the updated cells.
