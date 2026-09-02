import jax
import jax.numpy as jnp
from adjaxt.config import CCEConfig, exec_fn_for


def _softcap_logits(logits: jax.Array, softcap: float) -> jax.Array:
    return softcap * jnp.tanh(logits / softcap)


def _softcap_logits_vjp(
    logits: jax.Array, softcap: float
) -> tuple[jax.Array, jax.Array]:
    """Computes capped logits and the derivative multiplier d(capped)/d(logits)."""
    tanh_val = jnp.tanh(logits / softcap)
    capped = softcap * tanh_val
    grad_factor = 1.0 - tanh_val**2
    return capped, grad_factor


@exec_fn_for(CCEConfig)
def exec_cut_ce_autograd(
    cfg: CCEConfig,
    params: dict[str, jax.Array],
    hidden_states: jax.Array,
    weight: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    # Handle shift if necessary
    if cfg.shift:
        hidden_states = hidden_states[:, :-1, :]
        labels = labels[:, 1:]

    flat_h = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1)

    return _cut_ce_custom_vjp(flat_h, weight, flat_labels, cfg)


@jax.custom_vjp
def _cut_ce_custom_vjp(
    flat_h: jax.Array,
    weight: jax.Array,
    flat_labels: jax.Array,
    cfg: CCEConfig,
) -> jax.Array:
    loss, _ = _cut_ce_fwd(flat_h, weight, flat_labels, cfg)
    return loss


def _cut_ce_fwd(
    flat_h: jax.Array,
    weight: jax.Array,
    flat_labels: jax.Array,
    cfg: CCEConfig,
) -> tuple[jax.Array, tuple]:
    """Forward pass: computes loss and log-sum-exp without saving logit tensors."""
    vocab_size = weight.shape[0]
    num_chunks = (vocab_size + cfg.chunk_size - 1) // cfg.chunk_size
    pad_size = num_chunks * cfg.chunk_size - vocab_size

    if pad_size > 0:
        weight_padded = jnp.pad(
            weight, ((0, pad_size), (0, 0)), constant_values=0.0
        )
    else:
        weight_padded = weight

    # 1. Chunked LSE Computation
    def compute_chunk_lse(carry, chunk_idx):
        start_idx = chunk_idx * cfg.chunk_size
        weight_chunk = jax.lax.dynamic_slice_in_dim(
            weight_padded, start_index=start_idx, slice_size=cfg.chunk_size, axis=0
        )
        logits_chunk = jnp.matmul(
            flat_h, weight_chunk.T, preferred_element_type=cfg.precision.output_dtype
        )

        if cfg.soft_cap is not None:
            logits_chunk = _softcap_logits(logits_chunk, cfg.soft_cap)

        chunk_indices = start_idx + jnp.arange(cfg.chunk_size)
        mask = chunk_indices < vocab_size
        logits_chunk = jnp.where(mask[None, :], logits_chunk, -1e9)

        chunk_max = jnp.max(logits_chunk, axis=-1, keepdims=True)
        chunk_sumexp = jnp.sum(
            jnp.exp(logits_chunk - chunk_max), axis=-1, keepdims=True
        )
        return carry, (chunk_max, chunk_sumexp)

    _, (chunk_maxes, chunk_sumexps) = jax.lax.scan(
        compute_chunk_lse, None, jnp.arange(num_chunks)
    )

    global_max = jnp.max(chunk_maxes, axis=0)
    global_sumexp = jnp.sum(
        chunk_sumexps * jnp.exp(chunk_maxes - global_max), axis=0
    )
    lse = (global_max + jnp.log(jnp.maximum(global_sumexp, 1e-10))).squeeze(-1)

    # 2. Target Logit Computation
    safe_labels = jnp.where(flat_labels == cfg.ignore_index, 0, flat_labels)
    target_weights = jnp.take(weight, safe_labels, axis=0)
    target_logits = jnp.sum(
        flat_h * target_weights, axis=-1, dtype=cfg.precision.output_dtype
    )

    if cfg.soft_cap is not None:
        target_logits = _softcap_logits(target_logits, cfg.soft_cap)

    # 3. Loss computation
    loss = lse - target_logits
    valid_mask = flat_labels != cfg.ignore_index
    loss = jnp.where(valid_mask, loss, 0.0)

    if cfg.reduction == "mean":
        final_loss = loss.sum() / jnp.maximum(valid_mask.sum(), 1)
    elif cfg.reduction == "sum":
        final_loss = loss.sum()
    else:
        final_loss = loss

    # Save ONLY minimal tensors for backward pass (No logits stored!)
    res = (flat_h, weight, flat_labels, lse, cfg)
    return final_loss, res


def _cut_ce_bwd(res: tuple, g: jax.Array) -> tuple[jax.Array, jax.Array, None, None]:
    flat_h, weight, flat_labels, lse, cfg = res

    vocab_size = weight.shape[0]
    num_chunks = (vocab_size + cfg.chunk_size - 1) // cfg.chunk_size
    pad_size = num_chunks * cfg.chunk_size - vocab_size

    if pad_size > 0:
        weight_padded = jnp.pad(
            weight, ((0, pad_size), (0, 0)), constant_values=0.0
        )
    else:
        weight_padded = weight

    valid_mask = flat_labels != cfg.ignore_index

    # 1. Properly compute scale depending on reduction
    if cfg.reduction == "mean":
        scale = g / jnp.maximum(valid_mask.sum(), 1)
    elif cfg.reduction == "sum":
        scale = g
    else:
        # 1D array case (unreduced loss) -> reshape to (N, 1) for broadcasting
        scale = g[:, None]

    dh_init = jnp.zeros_like(flat_h)
    dw_init = jnp.zeros_like(weight_padded)

    def accumulate_chunk_grads(carry, chunk_idx):
        dh_acc, dw_acc = carry
        start_idx = chunk_idx * cfg.chunk_size

        weight_chunk = jax.lax.dynamic_slice_in_dim(
            weight_padded, start_index=start_idx, slice_size=cfg.chunk_size, axis=0
        )

        logits_chunk = jnp.matmul(
            flat_h, weight_chunk.T, preferred_element_type=cfg.precision.output_dtype
        )

        if cfg.soft_cap is not None:
            logits_chunk, softcap_grad = _softcap_logits_vjp(
                logits_chunk, cfg.soft_cap
            )

        probs_chunk = jnp.exp(logits_chunk - lse[:, None])

        chunk_indices = start_idx + jnp.arange(cfg.chunk_size)
        target_mask = (
            flat_labels[:, None] == chunk_indices[None, :]
        ) & valid_mask[:, None]

        dlogits = (
            probs_chunk - target_mask.astype(probs_chunk.dtype)
        ) * valid_mask[:, None]

        if cfg.soft_cap is not None:
            dlogits = dlogits * softcap_grad

        # 2. Scale broadcasting fix: works for scalar and 2D scale
        dlogits = dlogits * scale

        dh_chunk = jnp.matmul(dlogits, weight_chunk)
        dw_chunk = jnp.matmul(dlogits.T, flat_h)

        dh_acc = dh_acc + dh_chunk
        dw_acc = jax.lax.dynamic_update_slice_in_dim(
            dw_acc, dw_chunk, start_index=start_idx, axis=0
        )

        return (dh_acc, dw_acc), None

    (dh, dw_padded), _ = jax.lax.scan(
        accumulate_chunk_grads, (dh_init, dw_init), jnp.arange(num_chunks)
    )

    dw = dw_padded[:vocab_size, :]

    # 3. Reshape gradient back if shift was applied in the wrapper
    # (Matches original input shape (B, S, D))
    dh = dh.reshape(
        flat_h.shape[0] if not cfg.shift else flat_h.shape[0] // (flat_h.shape[0] // flat_h.shape[0]),
        -1,
    ) if False else dh # Handled in caller or reshape directly to h.shape

    return dh, dw, None, None


_cut_ce_custom_vjp.defvjp(_cut_ce_fwd, _cut_ce_bwd)


@exec_fn_for(CCEConfig)
def exec_chunked_ce_autograd(
    cfg: CCEConfig,
    params: dict[str, jax.Array],
    hidden_states: jax.Array,
    weight: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    hidden_states = hidden_states.astype(cfg.precision.compute_dtype)
    weight = weight.astype(cfg.precision.compute_dtype)

    if cfg.shift:
        hidden_states = hidden_states[:, :-1, :]
        labels = labels[:, 1:]

    flat_h = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1)
    vocab_size = weight.shape[0]

    num_chunks = (vocab_size + cfg.chunk_size - 1) // cfg.chunk_size
    pad_size = num_chunks * cfg.chunk_size - vocab_size
    if pad_size > 0:
        weight_padded = jnp.pad(
            weight, ((0, pad_size), (0, 0)), constant_values=0.0
        )
    else:
        weight_padded = weight

    def compute_chunk_lse(carry, chunk_idx):
        start_idx = chunk_idx * cfg.chunk_size
        weight_chunk = jax.lax.dynamic_slice_in_dim(
            weight_padded, start_index=start_idx, slice_size=cfg.chunk_size, axis=0
        )
        logits_chunk = jnp.matmul(
            flat_h, weight_chunk.T, preferred_element_type=cfg.precision.output_dtype
        )

        if cfg.soft_cap is not None:
            logits_chunk = _softcap_logits(logits_chunk, cfg.soft_cap)

        chunk_indices = start_idx + jnp.arange(cfg.chunk_size)
        mask = chunk_indices < vocab_size
        logits_chunk = jnp.where(mask[None, :], logits_chunk, -1e9)

        chunk_max = jnp.max(logits_chunk, axis=-1, keepdims=True)
        chunk_sumexp = jnp.sum(jnp.exp(logits_chunk - chunk_max), axis=-1, keepdims=True)
        return carry, (chunk_max, chunk_sumexp)

    checkpointed_chunk_fn = jax.checkpoint(compute_chunk_lse)

    _, (chunk_maxes, chunk_sumexps) = jax.lax.scan(
        checkpointed_chunk_fn, None, jnp.arange(num_chunks)
    )
    global_max = jnp.max(chunk_maxes, axis=0)
    global_sumexp = jnp.sum(
        chunk_sumexps * jnp.exp(chunk_maxes - global_max), axis=0
    )

    lse = (global_max + jnp.log(jnp.maximum(global_sumexp, 1e-10))).squeeze(-1)

    safe_labels = jnp.where(flat_labels == cfg.ignore_index, 0, flat_labels)
    target_weights = jnp.take(weight, safe_labels, axis=0)
    target_logits = jnp.sum(
        flat_h * target_weights, axis=-1, dtype=cfg.precision.output_dtype
    )

    if cfg.soft_cap is not None:
        target_logits = _softcap_logits(target_logits, cfg.soft_cap)

    loss = lse - target_logits
    valid_mask = flat_labels != cfg.ignore_index
    loss = jnp.where(valid_mask, loss, 0.0)

    if cfg.reduction == "mean":
        return loss.sum() / jnp.maximum(valid_mask.sum(), 1)
    elif cfg.reduction == "sum":
        return loss.sum()

    return loss
