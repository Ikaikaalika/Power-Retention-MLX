# Metal Kernel Implementation Details

This document provides technical details about the custom Metal GPU kernels used in Power Retention MLX.

## Overview

Power Retention uses three custom Metal kernels for GPU-accelerated computation:

1. **Phi Kernel** - Quadratic feature expansion
2. **State Update Kernel** - Recurrent state updates with gating
3. **Output Kernel** - Matrix-vector products for output computation

All kernels are implemented using `mlx.core.fast.metal_kernel` and are JIT-compiled on first use.

## Kernel 1: Phi Feature Computation

### Purpose
Computes the quadratic feature expansion phi(x) = x⊗x for power p=2.

### Algorithm
For input dimension d, computes d*(d+1)/2 features representing all pairs (i,j) where i≤j:
- Diagonal terms: x[i] * x[i]
- Off-diagonal terms: sqrt(2) * x[i] * x[j]

### Grid Configuration
- **Grid dimensions**: (num_tokens, expanded_dim, 1)
- **Threadgroup size**: (1, 32, 1)
- **Parallelization**: Each thread computes one feature for one token

### Metal Code Highlights
```metal
// Map feature index to (i,j) pair
uint i = 0, cum = 0, prev_cum = 0;
while (cum + (dim - i) <= feat_idx) {
    prev_cum = cum;
    cum += (dim - i);
    i++;
}
uint j = i + (feat_idx - prev_cum);

// Compute feature with sqrt(2) scaling for off-diagonal
T val = x_i * x_j;
if (i != j) val *= metal::sqrt(T(2.0));
```

### Performance
- Time complexity: O(d²) per token (parallelized)
- Memory: Input [seq, d] -> Output [seq, d*(d+1)/2]

## Kernel 2: State Update

### Purpose
Updates recurrent state: S_t = g_t * phi(k_t) ⊗ v_t + S_{t-1}

### Algorithm
Performs outer product between phi features and values, scaled by gate g, then adds to current state.

### Grid Configuration
- **Grid dimensions**: (expanded_dim, dim, 1)
- **Threadgroup size**: (16, 16, 1)
- **Parallelization**: Each thread updates one element of the state matrix

### Metal Code Highlights
```metal
T phi_k_val = phi_k[row];
T v_val = v[col];
T g_val = g[0];  // Scalar gate value
T update = g_val * phi_k_val * v_val;

// Read old state, add update
out_state[row * dim + col] = in_state[row * dim + col] + update;
```

### Key Design Choice
Uses separate input/output buffers (in_state, out_state) to avoid Metal's restrictions on read-write buffers.

### Performance
- Time complexity: O(expanded_dim * dim) per token (parallelized)
- Memory: State shape [expanded_dim, dim]

## Kernel 3: Output Computation

### Purpose
Computes output vector: y_t = phi(q_t)^T @ S_t

### Algorithm
Matrix-vector product between query features and state matrix.

### Grid Configuration
- **Grid dimensions**: (dim, 1, 1)
- **Threadgroup size**: (32, 1, 1)
- **Parallelization**: Each thread computes one output dimension

### Metal Code Highlights
```metal
T sum = T(0.0);
for (uint row = 0; row < expanded_dim; ++row) {
    sum += phi_q[row] * state[row * dim + out_idx];
}
output[out_idx] = sum;
```

### Performance
- Time complexity: O(expanded_dim * dim) per token
- Uses serial reduction within each thread (could be optimized with shared memory)

## Normalization

After each output computation, we apply normalization:

```python
weight_sum = mx.sum(phi_q * phi_k) + 1e-6
output = y / weight_sum
```

This approximates the attention normalization and stabilizes training.

## Performance Characteristics

### Comparison to Attention

| Operation | Attention | Power Retention |
|-----------|-----------|-----------------|
| Time per token | O(seq_len * dim) | O(dim²) |
| Memory | O(seq_len²) | O(dim²) |
| State size | Grows with seq | Fixed |

### Optimizations

1. **JIT Compilation**: Kernels compiled once, cached for subsequent calls
2. **Grid tuning**: Threadgroup sizes chosen for Apple Silicon architecture
3. **Memory layout**: Row-major layout for cache-friendly access
4. **Constant folding**: dim and expanded_dim baked into kernel source

### Tuning Parameters

Adjust these in the kernel initialization:

```python
# Phi kernel
threadgroup=(1, 32, 1)  # Increase second dim for larger expanded_dim

# State update
threadgroup=(16, 16, 1)  # Balance based on expanded_dim and dim

# Output kernel
threadgroup=(32, 1, 1)  # Increase for larger dim
```

## Future Optimizations

Potential improvements:

1. **Shared memory**: Use threadgroup memory for reductions
2. **Atomics**: Enable parallel state updates across tokens
3. **Batch parallelism**: Process multiple batches in single kernel call
4. **Fused ops**: Combine phi computation with state update
5. **Half precision**: Use float16 for memory bandwidth optimization

## Debugging

Enable Metal shader debugging:

```python
# In kernel source, add debug output:
if (thread_position_in_grid.x == 0) {
    // Print first thread's values
}
```

Check Metal compilation errors:
```bash
# MLX will print detailed Metal compiler errors to stderr
python3 your_script.py 2>&1 | grep -A 10 "error:"
```

## References

- [MLX Custom Metal Kernels Documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Power Retention Paper](https://github.com/m-a-n-i-f-e-s-t/retention)
