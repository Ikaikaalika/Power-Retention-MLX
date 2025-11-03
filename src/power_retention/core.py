import mlx.core as mx
import mlx.nn as nn
import mlx.core.fast as fast

class PowerRetention(nn.Module):
    def __init__(self, dim: int, power: int = 2, chunk_size: int = 128, use_metal_kernels: bool = True):
        super().__init__()
        self.dim = dim
        self.power = power
        self.chunk_size = chunk_size
        self.use_metal_kernels = use_metal_kernels

        if power != 2:
            raise ValueError("Only p=2 supported; extend for higher.")
        self.expanded_dim = (dim * (dim + 1)) // 2

        # Learnable projections
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)

        # Initialize Metal kernels (optional)
        if use_metal_kernels:
            self._init_kernels()

        # Initialize state
        self._state = None
        self.reset_state()

    def _init_kernels(self):
        """Initialize custom Metal kernels for GPU acceleration."""

        # Kernel 1: Compute phi features (quadratic expansion)
        phi_source = f"""
        uint token_idx = thread_position_in_grid.x;
        uint feat_idx = thread_position_in_grid.y;

        const uint dim = {self.dim};
        const uint expanded_dim = {self.expanded_dim};

        if (feat_idx >= expanded_dim) return;

        // Map feat_idx to (i, j) where i <= j
        uint i = 0;
        uint cum = 0;
        uint prev_cum = 0;
        while (cum + (dim - i) <= feat_idx) {{
            prev_cum = cum;
            cum += (dim - i);
            i++;
        }}
        uint j = i + (feat_idx - prev_cum);

        // Get input values
        T x_i = input[token_idx * dim + i];
        T x_j = input[token_idx * dim + j];

        // Compute phi feature
        T val = x_i * x_j;
        if (i != j) {{
            val *= metal::sqrt(T(2.0));
        }}

        output[token_idx * expanded_dim + feat_idx] = val;
        """

        self.phi_kernel = fast.metal_kernel(
            name="compute_phi",
            input_names=["input"],
            output_names=["output"],
            source=phi_source,
        )

        # Kernel 2: Update state with gating
        state_update_source = f"""
        uint row = thread_position_in_grid.x;
        uint col = thread_position_in_grid.y;

        const uint expanded_dim = {self.expanded_dim};
        const uint dim = {self.dim};

        if (row >= expanded_dim || col >= dim) return;

        T phi_k_val = phi_k[row];
        T v_val = v[col];
        T g_val = g[0];  // Dereference scalar
        T update = g_val * phi_k_val * v_val;

        out_state[row * dim + col] = in_state[row * dim + col] + update;
        """

        self.state_update_kernel = fast.metal_kernel(
            name="state_update",
            input_names=["in_state", "phi_k", "v", "g"],
            output_names=["out_state"],
            source=state_update_source,
        )

        # Kernel 3: Compute output from state
        output_source = f"""
        uint out_idx = thread_position_in_grid.x;

        const uint expanded_dim = {self.expanded_dim};
        const uint dim = {self.dim};

        if (out_idx >= dim) return;

        T sum = T(0.0);
        for (uint row = 0; row < expanded_dim; ++row) {{
            sum += phi_q[row] * state[row * dim + out_idx];
        }}

        output[out_idx] = sum;
        """

        self.output_kernel = fast.metal_kernel(
            name="compute_output",
            input_names=["phi_q", "state"],
            output_names=["output"],
            source=output_source,
        )

    def reset_state(self):
        self._state = mx.zeros((self.expanded_dim, self.dim), dtype=mx.float32)

    def _compute_phi_mlx(self, x: mx.array) -> mx.array:
        """
        Compute phi features using pure MLX operations (supports autodiff).

        Args:
            x: Input of shape [batch, seq, dim] or [seq, dim]

        Returns:
            phi: Features of shape [batch, seq, expanded_dim] or [seq, expanded_dim]
        """
        original_shape = x.shape
        if len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])

        batch, seq, dim = x.shape

        # Create indices for upper triangular part (i <= j)
        indices = []
        for i in range(dim):
            for j in range(i, dim):
                indices.append((i, j))

        # Pre-compute sqrt(2) constant
        sqrt_2 = 1.4142135623730951  # sqrt(2)

        # Compute phi features: x[i] * x[j] for all pairs (i,j) with i<=j
        phi_list = []
        for i, j in indices:
            if i == j:
                # Diagonal terms
                phi_list.append(x[:, :, i:i+1] * x[:, :, j:j+1])
            else:
                # Off-diagonal terms (multiply by sqrt(2) for symmetry)
                phi_list.append(sqrt_2 * x[:, :, i:i+1] * x[:, :, j:j+1])

        phi = mx.concatenate(phi_list, axis=-1)

        # Restore original batch dimension if needed
        if len(original_shape) == 2:
            phi = phi.squeeze(0)

        return phi

    def _compute_phi_metal(self, x: mx.array) -> mx.array:
        """
        Compute phi features using Metal kernel.

        Args:
            x: Input of shape [seq, dim] (flattened batch)

        Returns:
            phi: Features of shape [seq, expanded_dim]
        """
        seq = x.shape[0]

        outputs = self.phi_kernel(
            inputs=[x],
            template=[("T", x.dtype)],
            grid=(seq, self.expanded_dim, 1),
            threadgroup=(1, 32, 1),
            output_shapes=[(seq, self.expanded_dim)],
            output_dtypes=[x.dtype],
        )

        return outputs[0]

    def _update_state_metal(self, state: mx.array, phi_k: mx.array, v: mx.array, g: float) -> mx.array:
        """
        Update state using Metal kernel.

        Args:
            state: Current state [expanded_dim, dim]
            phi_k: Phi features for key [expanded_dim]
            v: Value vector [dim]
            g: Gating value (scalar)

        Returns:
            Updated state [expanded_dim, dim]
        """
        g_array = mx.array([g], dtype=state.dtype)

        outputs = self.state_update_kernel(
            inputs=[state, phi_k, v, g_array],
            template=[("T", state.dtype)],
            grid=(self.expanded_dim, self.dim, 1),
            threadgroup=(16, 16, 1),
            output_shapes=[state.shape],
            output_dtypes=[state.dtype],
        )

        return outputs[0]

    def _compute_output_metal(self, phi_q: mx.array, state: mx.array) -> mx.array:
        """
        Compute output using Metal kernel.

        Args:
            phi_q: Phi features for query [expanded_dim]
            state: Current state [expanded_dim, dim]

        Returns:
            Output vector [dim]
        """
        outputs = self.output_kernel(
            inputs=[phi_q, state],
            template=[("T", state.dtype)],
            grid=(self.dim, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(self.dim,)],
            output_dtypes=[state.dtype],
        )

        return outputs[0]

    def __call__(self, x: mx.array, log_g: mx.array = None) -> mx.array:
        """
        Forward pass through power retention.

        Args:
            x: Input tensor of shape [batch, seq, dim]
            log_g: Optional gating values of shape [batch, seq]

        Returns:
            outputs: Output tensor of shape [batch, seq, dim]
        """
        # Use Metal kernels for inference, pure MLX for training
        if self.use_metal_kernels and not mx.metal.is_available():
            # Fallback to MLX if Metal not available
            return self._forward_mlx(x, log_g)
        elif self.use_metal_kernels:
            return self._forward_metal(x, log_g)
        else:
            return self._forward_mlx(x, log_g)

    def _forward_mlx(self, x: mx.array, log_g: mx.array = None) -> mx.array:
        """
        Forward pass using pure MLX operations (supports autodiff).
        """
        batch, seq, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        if log_g is None:
            log_g = mx.zeros((batch, seq), dtype=mx.float32)
        g = mx.exp(log_g)

        outputs = mx.zeros_like(q)

        for b in range(batch):
            chunk_state = mx.array(self._state)

            for start in range(0, seq, self.chunk_size):
                end = min(start + self.chunk_size, seq)

                q_chunk = q[b:b+1, start:end]
                k_chunk = k[b:b+1, start:end]
                v_chunk = v[b:b+1, start:end]
                g_chunk = g[b:b+1, start:end]

                # Compute phi features
                phi_q = self._compute_phi_mlx(q_chunk)  # [1, chunk_len, expanded_dim]
                phi_k = self._compute_phi_mlx(k_chunk)  # [1, chunk_len, expanded_dim]

                # Process each token in the chunk
                for i in range(end - start):
                    # Update state: S_t = g_t * phi(k_t) ⊗ v_t + S_{t-1}
                    g_val = g_chunk[0, i]
                    phi_k_i = phi_k[0, i:i+1]  # [1, expanded_dim]
                    v_i = v_chunk[0, i:i+1]    # [1, dim]

                    # Outer product and add to state
                    update = g_val * mx.matmul(phi_k_i.T, v_i)  # [expanded_dim, dim]
                    chunk_state = chunk_state + update

                    # Compute output: y_t = phi(q_t)^T @ S_t
                    phi_q_i = phi_q[0, i]  # [expanded_dim]
                    y_i = mx.matmul(phi_q_i.reshape(1, self.expanded_dim), chunk_state)  # [1, dim]

                    # Normalization by approximate weight sum
                    weight_sum = mx.sum(phi_q_i * phi_k[0, i]) + 1e-6
                    outputs[b, start + i] = (y_i / weight_sum).squeeze(0)

                # Clear cache after each chunk
                mx.clear_cache()

            self._state = chunk_state

        return outputs

    def _forward_metal(self, x: mx.array, log_g: mx.array = None) -> mx.array:
        """
        Forward pass using Metal kernels (faster but no gradients).
        """
        batch, seq, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        if log_g is None:
            log_g = mx.zeros((batch, seq), dtype=mx.float32)
        g = mx.exp(log_g)

        outputs = mx.zeros_like(q)

        for b in range(batch):
            chunk_state = mx.array(self._state)

            for start in range(0, seq, self.chunk_size):
                end = min(start + self.chunk_size, seq)

                q_chunk = q[b, start:end]
                k_chunk = k[b, start:end]
                v_chunk = v[b, start:end]
                g_chunk = g[b, start:end]

                # Compute phi features using Metal kernel
                phi_q = self._compute_phi_metal(q_chunk)  # [chunk_len, expanded_dim]
                phi_k = self._compute_phi_metal(k_chunk)  # [chunk_len, expanded_dim]

                # Process each token in the chunk
                for i in range(end - start):
                    g_val = float(g_chunk[i].item())

                    # Update state using Metal kernel
                    chunk_state = self._update_state_metal(
                        chunk_state,
                        phi_k[i],
                        v_chunk[i],
                        g_val
                    )

                    # Compute output using Metal kernel
                    y_i = self._compute_output_metal(phi_q[i], chunk_state)

                    # Normalization by approximate weight sum
                    weight_sum = mx.sum(phi_q[i] * phi_k[i]) + 1e-6
                    outputs[b, start + i] = y_i / weight_sum

            self._state = chunk_state

        return outputs

    def inference_step(self, x: mx.array, log_g: float = 0.0) -> mx.array:
        """
        Single-step recurrent inference with Metal kernels.

        Args:
            x: Input token of shape [dim] or [batch, dim]
            log_g: Gating value (scalar)

        Returns:
            y: Output of shape [dim] or [batch, dim]
        """
        original_shape = x.shape
        if len(x.shape) == 1:
            x = x.reshape(1, self.dim)

        batch = x.shape[0]
        outputs = mx.zeros((batch, self.dim), dtype=mx.float32)

        for b in range(batch):
            # Project current token
            q = self.w_q(x[b:b+1]).squeeze(0)
            k = self.w_k(x[b:b+1]).squeeze(0)
            v = self.w_v(x[b:b+1]).squeeze(0)

            g_val = float(mx.exp(mx.array([log_g]))[0].item())

            # Compute phi features using Metal kernel
            phi_q = self._compute_phi_metal(q.reshape(1, self.dim)).squeeze(0)
            phi_k = self._compute_phi_metal(k.reshape(1, self.dim)).squeeze(0)

            # Update state using Metal kernel
            self._state = self._update_state_metal(self._state, phi_k, v, g_val)

            # Compute output using Metal kernel
            y = self._compute_output_metal(phi_q, self._state)

            # Normalization
            weight_sum = mx.sum(phi_q * phi_k) + 1e-6
            outputs[b] = y / weight_sum

        return outputs.squeeze(0) if len(original_shape) == 1 else outputs
