from pandas.io.formats.format import re
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from adjaxt.config import CCEConfig, PrecisionPolicy
from adjaxt.loss import exec_chunked_ce_autograd, exec_cut_ce_autograd

def standard_jax_ce(
    hidden_states: jax.Array,
    weight: jax.Array,
    labels: jax.Array,
    shift: bool = True,
    ignore_index: int = -100,
    reduction: str = "mean",
    softcap: float | None = None,
) -> jax.Array:
    if shift:
        hidden_states = hidden_states[:, :-1, :]
        labels = labels[:, 1:]

    flat_h = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1)

    logits = jnp.matmul(flat_h, weight.T, preferred_element_type=jnp.float32)
    if softcap is not None:
        logits = softcap * jnp.tanh(logits / softcap)

    lse = jax.nn.logsumexp(logits, axis=-1)

    safe_labels = jnp.where(flat_labels == ignore_index, 0, flat_labels)
    target_probs = jnp.take_along_axis(logits, safe_labels[:, None], axis=-1).squeeze(-1)

    loss = lse - target_probs
    valid_mask = flat_labels != ignore_index
    loss = jnp.where(valid_mask, loss, 0.0)
    if reduction == "mean":
        return loss.sum() / jnp.maximum(valid_mask.sum(), 1)
    elif reduction == "sum":
        return loss.sum()
    return loss

@pytest.mark.parametrize("shift", [True, False])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("chunk_size", [128, 512, 1024])
@pytest.mark.parametrize("softcap", [None, 30.0])
def test_cces_vs_jax_ce(shift, reduction, chunk_size, softcap):
    batch_size, seq_len, d_model, vocab_size = 2, 16, 64, 1000

    key = jax.random.PRNGKey(42)

    k1, k2, k3 = jax.random.split(key, 3)
    h = jax.random.normal(k1, (batch_size, seq_len, d_model), dtype=jnp.float32)
    w = jax.random.normal(k2, (vocab_size, d_model), dtype=jnp.float32)
    labels = jax.random.randint(k3, (batch_size, seq_len), 0, vocab_size)

    # Ignore some idx for testing
    labels = labels.at[0, 5].set(-100)
    labels = labels.at[1, 10].set(-100)

    def ref_loss_fn(h_arg, w_arg):
        return standard_jax_ce(
            h_arg,
            w_arg,
            labels,
            shift=shift,
            ignore_index=-100,
            reduction=reduction,
            softcap=softcap,
        )

    ref_loss, (dh_ref, dw_ref) = jax.value_and_grad(ref_loss_fn, argnums=(0, 1))(h, w)

    cfg = CCEConfig(
        reduction=reduction,
        ignore_index=-100,
        shift=shift,
        chunk_size=chunk_size,
        soft_cap=softcap,
        precision=PrecisionPolicy(
            compute_dtype=jnp.float32, output_dtype=jnp.float32
        ),
    )

    def chunked_ce_loss_fn(h_arg, w_arg):
        return exec_chunked_ce_autograd(cfg, {}, h_arg, w_arg, labels)

    def cut_ce_loss_fn(h_arg, w_arg):
        return exec_cut_ce_autograd(cfg, {}, h_arg, w_arg, labels)

    chunked_ce_loss, (dh_chunked_ce, dw_chunked_ce) = jax.value_and_grad(chunked_ce_loss_fn, argnums=(0, 1))(h, w)
    cut_ce_loss, (dh_cut_ce, dw_cut_ce) = jax.value_and_grad(cut_ce_loss_fn, argnums=(0, 1))(h, w)

    np.testing.assert_allclose(
        np.array(chunked_ce_loss), np.array(ref_loss), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        np.array(dh_chunked_ce), np.array(dh_ref), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        np.array(dw_chunked_ce), np.array(dw_ref), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        np.array(cut_ce_loss), np.array(ref_loss), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        np.array(dh_cut_ce), np.array(dh_ref), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        np.array(dw_cut_ce), np.array(dw_ref), rtol=1e-3, atol=1e-3
    )
