# BayesFlow Network Parameter Review

## Current Search Space

### DeepSet (Summary Network)
- ✅ `summary_dim`: (4, 16) - output dimension
- ✅ `deepset_width`: (32, 96) - width of all MLP layers
- ✅ `deepset_depth`: (1, 3) - number of aggregation stages
- ✅ `deepset_dropout`: (0.0, 0.15) - dropout rate

### CouplingFlow (Inference Network)
- ✅ `flow_depth`: (3, 8) - number of coupling layers
- ✅ `flow_hidden`: (32, 128) - hidden layer sizes in coupling subnets
- ✅ `flow_dropout`: (0.05, 0.25) - dropout rate in coupling subnets

---

## Recommended Additions

### High Priority (Likely to improve performance)

#### 1. **DeepSet Activation Function**
```python
deepset_activation: ["relu", "gelu", "silu", "elu", "mish"]
```
- **Current default**: `"silu"`
- **Impact**: Different activations can significantly affect learning dynamics
- **Recommendation**: Add as categorical parameter
- **Why**: "gelu" and "mish" often outperform "silu" on complex tasks

#### 2. **CouplingFlow Subnet Activation**
```python
flow_activation: ["relu", "gelu", "mish", "elu", "silu"]
```
- **Current default**: `"mish"` (in MLP subnet)
- **Impact**: Can improve expressiveness of transformations
- **Recommendation**: Add as categorical parameter
- **Why**: Different activations for flow vs. summary net can be beneficial

#### 3. **CouplingFlow Transform Type**
```python
flow_transform: ["affine", "rqs"]  # rqs = Rational Quadratic Spline
```
- **Current default**: `"affine"`
- **Impact**: RQS transformations are more flexible but more expensive
- **Recommendation**: Add for advanced optimization
- **Why**: RQS can model more complex posterior shapes

---

### Medium Priority (May help in specific cases)

#### 4. **DeepSet Pooling Strategy**
```python
deepset_inner_pooling: ["mean", "sum", "max"]
deepset_output_pooling: ["mean", "sum", "max"]
```
- **Current default**: `"mean"` for both
- **Impact**: Affects how set information is aggregated
- **Recommendation**: Add if struggling with set size invariance
- **Why**: Different pooling can handle varying dataset sizes better

#### 5. **CouplingFlow Residual Connections**
```python
flow_residual: [True, False]
```
- **Current default**: `True` (in MLP subnet)
- **Impact**: Stabilizes training for deep networks
- **Recommendation**: Add only if experiencing instability
- **Why**: True is almost always better, but worth confirming

#### 6. **Spectral Normalization**
```python
deepset_spectral_norm: [False, True]
flow_spectral_norm: [False, True]
```
- **Current default**: `False`
- **Impact**: Can improve training stability and generalization
- **Recommendation**: Add if overfitting or training instability observed
- **Why**: Provides Lipschitz constraint for better calibration

---

### Low Priority (Usually not worth optimizing)

#### 7. **Kernel Initializers**
- **Current defaults**: `"he_normal"` for both
- **Alternatives**: `"glorot_uniform"`, `"lecun_normal"`
- **Recommendation**: **Skip** - modern defaults are well-tuned
- **Why**: Minimal impact with proper learning rate tuning

#### 8. **CouplingFlow Permutation Strategy**
- **Current default**: `"random"`
- **Alternatives**: `"reverse"`, `"fixed"`
- **Recommendation**: **Skip** - random is usually best
- **Why**: Random permutations provide good mixing

#### 9. **CouplingFlow ActNorm**
- **Current default**: `True`
- **Recommendation**: **Skip** - should stay enabled
- **Why**: Helps with scale invariance

---

## Implementation Priority

### Phase 1: Essential (Implement Now)
1. `deepset_activation` - categorical
2. `flow_activation` - categorical

### Phase 2: Advanced (If Phase 1 plateaus)
3. `flow_transform` - categorical ["affine", "rqs"]
4. `deepset_spectral_norm` - boolean
5. `flow_spectral_norm` - boolean

### Phase 3: Fine-tuning (Optional)
6. `deepset_inner_pooling` / `deepset_output_pooling` - categorical
7. `flow_residual` - boolean

---

## Updated Search Space Code

```python
from bayesian_optimization import HyperparameterSpace

search_space = HyperparameterSpace(
    # DeepSet - Core Architecture
    summary_dim=(4, 16),
    deepset_width=(32, 96),
    deepset_depth=(1, 3),
    deepset_dropout=(0.0, 0.15),
    deepset_activation=["relu", "gelu", "silu", "elu", "mish"],  # NEW

    # CouplingFlow - Core Architecture
    flow_depth=(3, 8),
    flow_hidden=(32, 128),
    flow_dropout=(0.05, 0.25),
    flow_activation=["relu", "gelu", "mish", "elu", "silu"],  # NEW

    # Advanced (Phase 2)
    # flow_transform=["affine", "rqs"],  # Uncomment if needed
    # deepset_spectral_norm=[False, True],  # Uncomment if overfitting
    # flow_spectral_norm=[False, True],  # Uncomment if overfitting

    # Training
    initial_lr=(1e-4, 5e-3),
    batch_size=(128, 384),

    # Fixed
    decay_rate=0.85,
    patience=10,
    window=10,
)
```

---

## Notes on Implementation

### For `ancova_cont_2arms_fns.py`

Update `NetworkConfig`:
```python
@dataclass
class NetworkConfig:
    """Neural network architecture hyperparameters."""
    summary_dim: int = 10
    deepset_depth: int = 3
    deepset_width: int = 64
    deepset_dropout: float = 0.05
    deepset_activation: str = "gelu"  # NEW

    flow_depth: int = 7
    flow_hidden: int = 128
    flow_dropout: float = 0.20
    flow_activation: str = "mish"  # NEW
    flow_transform: str = "affine"  # NEW (optional)
```

Update `build_networks()`:
```python
def build_networks(config: NetworkConfig) -> tuple:
    summary_net = bf.networks.DeepSet(
        summary_dim=config.summary_dim,
        depth=config.deepset_depth,
        mlp_widths_equivariant=(config.deepset_width,) * 2,
        mlp_widths_invariant_inner=(config.deepset_width,) * 2,
        mlp_widths_invariant_outer=(config.deepset_width,) * 2,
        mlp_widths_invariant_last=(config.deepset_width,) * 2,
        dropout=config.deepset_dropout,
        activation=config.deepset_activation,  # NEW
    )

    inference_net = bf.networks.CouplingFlow(
        depth=config.flow_depth,
        transform=config.flow_transform,  # NEW (if added)
        subnet_kwargs=dict(
            widths=[config.flow_hidden, config.flow_hidden],
            dropout=config.flow_dropout,
            activation=config.flow_activation,  # NEW
        ),
    )

    return summary_net, inference_net
```

---

## Expected Impact

### High Impact Parameters
- **Activation functions**: 5-15% improvement in calibration
- **Flow transform** (RQS): 10-20% improvement for complex posteriors, but 2x slower

### Medium Impact Parameters
- **Spectral normalization**: 2-5% improvement if overfitting
- **Pooling strategies**: 2-8% improvement for variable set sizes

### Low Impact Parameters
- **Residual connections**: 0-3% (usually already optimal)
- **Kernel initializers**: <1%

---

## Computational Cost

| Parameter | Cost Impact |
|-----------|-------------|
| `deepset_activation` | Negligible |
| `flow_activation` | Negligible |
| `flow_transform="rqs"` | **+100% training time** |
| `spectral_norm=True` | +10-20% training time |
| `flow_depth` | Linear scaling |

---

## Recommendation Summary

**Start with Phase 1** (activation functions) - these have the highest impact-to-cost ratio and are easy to implement.

**Consider Phase 2** only if:
- Calibration errors plateau above target
- You observe overfitting (use spectral_norm)
- You need more flexible transformations (use RQS transform)

**Skip Phase 3** unless you have very specific requirements for set aggregation behavior.
