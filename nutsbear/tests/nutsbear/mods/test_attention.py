#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_attention.py
#   Author: xyy15926
#   Created: 2025-06-17 15:58:08
#   Updated: 2026-08-09 19:32:30
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import torch
from pytest import mark
from torch import nn, optim
from torch.nn import functional as F

if __name__ == "__main__":
    from importlib import reload

    from nutsbear.mods import attention, fixture

    reload(fixture)
    reload(attention)

from nutsbear.mods.attention import (
    MultiheadAttention,
    SimpleMHA,
    _infer_mask_shapes,
    _init_sdpa_mask,
    _merge_to_bias,
    scaled_dot_product_attention,
)
from nutsbear.mods.fixture import (
    all_close,
    fkwargs_32_dml,
    fkwargs_64_cpu,
)

torch.autograd.set_detect_anomaly(False)


# %%
if fkwargs_32_dml:
    torch_fkwargs_params = [fkwargs_64_cpu, fkwargs_32_dml]
else:
    torch_fkwargs_params = [fkwargs_64_cpu]


@pytest.fixture(params=torch_fkwargs_params)
def torch_fkwargs(request):
    return request.param


# torch_fkwargs = fkwargs_32_dml
# torch_fkwargs = fkwargs_64_cpu


# %%
@mark.filterwarnings(
    "ignore: .*is not currently supported on the DML backend*"
)
def test_scaled_dot_product_attention(torch_fkwargs):
    _dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]
    query = torch.randn(3, 4, 5, **torch_fkwargs)
    key = torch.randn(3, 6, 5, **torch_fkwargs)
    value = torch.randn(3, 6, 5, **torch_fkwargs)
    attn_mask = torch.randint(0, 2, (4, 6), device=device).to(torch.bool)
    attn_mask[0] = 0
    attn_mask_3D = torch.randint(0, 2, (3, 4, 6), device=device).to(torch.bool)

    # Attention mask SDPA.
    outp, _ws = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask.logical_not()
    )
    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask
    )
    assert all_close(outp, foutp, 0, 1)

    outp, _ws = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask.logical_not(),
        safe_softmax=False,
    )
    assert all_close(outp, foutp, 1, 1)

    # 3D-Attention mask SDPA.
    outp, _ws = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask_3D.logical_not()
    )
    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask_3D
    )
    assert all_close(outp, foutp, 0, 1)

    outp, _ws = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask_3D.logical_not(),
        safe_softmax=False,
    )
    assert all_close(outp, foutp, 1, 1)

    # Causal attention mask SDPA.
    outp, _ws = scaled_dot_product_attention(query, key, value, is_causal=True)
    foutp = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    assert all_close(outp, foutp, 0, 1)

    outp, _ws = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, is_causal=True
    )
    # Construct `attn_mask` for `F.scaled_dot_product_attention` manually.
    F_attn_mask = (
        attn_mask
        + 1
        - torch.ones(4, 6, dtype=torch.int, device=device).tril(diagonal=0)
    )
    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=F_attn_mask.logical_not()
    )
    assert all_close(outp, foutp, 0, 1)


# %%
def test_scaled_dot_product_attention_backward_grad(torch_fkwargs):
    _dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]
    query = torch.randn(3, 4, 2, requires_grad=True, **torch_fkwargs)
    key = torch.randn(3, 6, 2, requires_grad=True, **torch_fkwargs)
    value = torch.randn(3, 6, 2, requires_grad=True, **torch_fkwargs)
    sgd = optim.SGD((query, key, value))
    # attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    # attn_mask[0, :] = False
    # attn_mask[:, 0] = False
    # fmt: off
    attn_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
    ]).to(dtype=torch.bool, device=device)
    # fmt: on

    # `NaN` will be return for all-NInf query and lead to NaN in `.grad`
    # in backward.
    output, _ws = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask.logical_not(),
        safe_softmax=False,
    )
    os = output.sum()
    os.backward()
    assert torch.all(query.grad[:, 0, :].isnan())
    assert torch.all(query.grad[:, 1:, :].isnan().logical_not())
    assert torch.all(key.grad.isnan())
    assert torch.all(value.grad.isnan())
    sgd.step()
    assert torch.all(query[:, 0, :].isnan())
    assert torch.all(key.isnan())
    assert torch.all(value.isnan())

    # So does the `scaled_dot_product_attention` with safe-softmax.
    query = torch.randn(3, 4, 2, requires_grad=True, **torch_fkwargs)
    key = torch.randn(3, 6, 2, requires_grad=True, **torch_fkwargs)
    value = torch.randn(3, 6, 2, requires_grad=True, **torch_fkwargs)
    sgd = optim.SGD((query, key, value))

    outp, _ws = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask.logical_not(), safe_softmax=True
    )
    os = outp.sum()
    os.backward()
    assert torch.all(query.grad[:, 0, :] == 0)
    assert torch.all(query.grad.isnan().logical_not())
    # Key and value's grad are 0 because no query attention to 2nd key.
    assert torch.all(key.grad[:, 2, :] == 0)
    assert torch.all(value.grad[:, 2, :] == 0)
    sgd.step()
    assert torch.all(query.isnan().logical_not())
    assert torch.all(key.isnan().logical_not())
    assert torch.all(value.isnan().logical_not())


# %%
def test_MultiHeadAttention(torch_fkwargs):
    _dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]
    query = torch.randn(3, 4, 8, **torch_fkwargs)
    key = torch.randn(3, 6, 8, **torch_fkwargs)
    value = torch.randn(3, 6, 8, **torch_fkwargs)

    nnmha = nn.MultiheadAttention(8, 2, batch_first=True, **torch_fkwargs)
    nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(8, 2, **torch_fkwargs)
    sd = {
        "in_proj.weight": nn_sd["in_proj_weight"],
        "in_proj.bias": nn_sd["in_proj_bias"],
        "out_proj.weight": nn_sd["out_proj.weight"],
        "out_proj.bias": nn_sd["out_proj.bias"],
    }
    mha.load_state_dict(sd)

    # Default forward.
    nnattn, _nnw = nnmha(query, key, value)
    attn, _attn_ws = mha(query, key, value)
    assert all_close(nnattn, attn)

    # Forward with `is_causal` only.
    # `is_causal` in `nn.MultiheadAttention` is just a hint and
    # `attn_mask`(`src_mask`) must be set if `is_causal` is set.
    causal_mask = (
        MultiheadAttention.merge_masks(None, None, 1, query, key)
        .squeeze()
        .to(query.dtype)
    )
    nnattn, _nnw = nnmha(
        query, key, value, attn_mask=causal_mask, is_causal=True
    )
    attn, _attn_ws = mha(query, key, value, is_causal=True)
    assert all_close(nnattn, attn)

    # Forward with key-padding-mask.
    key_padding_mask = torch.randint(0, 2, (3, 6)).to(
        dtype=torch.bool, device=device
    )
    key_padding_mask[0, :] = True
    nnattn, _nnw = nnmha(query, key, value, key_padding_mask=key_padding_mask)
    attn, _attn_ws = mha(query, key, value, key_padding_mask=key_padding_mask)
    assert torch.all(torch.isnan(nnattn[0]))
    assert not torch.any(torch.isnan(attn))
    assert all_close(nnattn, attn, 1, 0)

    nnattn, _nnw = nnmha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask.logical_not(),
        need_weights=False,
    )
    attn, _attn_ws = mha(
        query, key, value, key_padding_mask=key_padding_mask.logical_not()
    )
    assert all_close(nnattn, attn, 1, 0)

    # Forward with attention-mask only.
    attn_mask = torch.randint(0, 2, (4, 6)).to(dtype=torch.bool, device=device)
    attn_mask[0, :] = True
    nnattn, nnattn_w = nnmha(
        query,
        key,
        value,
        attn_mask=attn_mask,
        need_weights=True,
    )
    attn, _attn_ws = mha(
        query,
        key,
        value,
        attn_mask=attn_mask,
        need_weights=False,
    )
    attn_w, attn_ws_w = mha(
        query,
        key,
        value,
        attn_mask=attn_mask,
        need_weights=True,
    )
    assert torch.all(torch.isnan(nnattn_w[:, 0]))
    assert not torch.any(torch.isnan(attn_ws_w))
    assert all_close(nnattn_w, attn_ws_w, 1, 0)
    assert all_close(nnattn, attn, 1, 1)
    assert all_close(attn, attn_w, 1, 0)

    # Forward with attention-mask(or mixed mask).
    attn_mask = torch.randint(0, 2, (4, 6)).to(dtype=torch.bool, device=device)
    nnattn, _nnw = nnmha(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )
    attn, _attn_ws = mha(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
    )
    assert all_close(nnattn, attn, 1, 0)

    nnattn, _nnw = nnmha(
        query,
        key,
        value,
        attn_mask=attn_mask.logical_not(),
        key_padding_mask=key_padding_mask.logical_not(),
        need_weights=False,
    )
    attn, _attn_ws = mha(
        query,
        key,
        value,
        attn_mask=attn_mask.logical_not(),
        key_padding_mask=key_padding_mask.logical_not(),
    )
    assert all_close(nnattn, attn, 1, 0)


# %%
def test_MultiHeadAttention_qkv_diffsz(torch_fkwargs):
    _dtype, _device = torch_fkwargs["dtype"], torch_fkwargs["device"]
    query = torch.randn(3, 4, 8, **torch_fkwargs)
    key = torch.randn(3, 6, 16, **torch_fkwargs)
    value = torch.randn(3, 6, 16, **torch_fkwargs)

    nnmha = nn.MultiheadAttention(
        8,
        1,
        bias=True,
        kdim=16,
        vdim=16,
        batch_first=True,
        **torch_fkwargs,
    )
    nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(8, 1, ksz=16, vsz=16, **torch_fkwargs)
    qb, kb, vb = nn_sd["in_proj_bias"].chunk(3)
    sd = {
        "q_proj.weight": nn_sd["q_proj_weight"],
        "k_proj.weight": nn_sd["k_proj_weight"],
        "v_proj.weight": nn_sd["v_proj_weight"],
        "q_proj.bias": qb,
        "k_proj.bias": kb,
        "v_proj.bias": vb,
        "out_proj.weight": nn_sd["out_proj.weight"],
        "out_proj.bias": nn_sd["out_proj.bias"],
    }
    mha.load_state_dict(sd)

    # Default forward.
    nnattn, _nnw = nnmha(query, key, value)
    attn, _attn_ws = mha(query, key, value)
    assert all_close(nnattn, attn, 1, 0)


# %%
def test_MHA_merge_masks(torch_fkwargs):
    _dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]

    # 4D-QKV and mask will be used here so that
    # `F.scaled_dot_product_attention` won't raise RuntimeError.
    query = torch.randn(3, 1, 4, 2, requires_grad=True, **torch_fkwargs)
    key = torch.randn(3, 1, 6, 2, requires_grad=True, **torch_fkwargs)
    value = torch.randn(3, 1, 6, 2, requires_grad=True, **torch_fkwargs)

    key_padding_mask = torch.randint(0, 2, (3, 6), device=device).to(
        torch.bool
    )
    # fmt: off
    key_padding_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
    ], device=device).to(torch.bool).logical_not()
    # fmt: on
    attn_mask = torch.randint(0, 2, (4, 6), device=device).to(torch.bool)
    # fmt: off
    attn_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
    ], device=device).to(torch.bool).logical_not()
    # fmt: on

    # Causality from `is_causal` or `attn_mask` lead to the same result.
    def check_SDPA_and_causal(query, key, value, non_causal_mask, causal_mask):
        cn_ret, _c_ws = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            is_causal=False,
        )
        nc_ret, _nc_ws = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=non_causal_mask,
            is_causal=True,
        )
        cc_ret, _c_ws = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            is_causal=True,
        )
        cn_fret = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            is_causal=False,
        )

        assert all_close(cn_ret, nc_ret)
        assert all_close(nc_ret, cc_ret)
        assert all_close(cc_ret, cn_fret, 0, 1)

        if device == torch.device("cpu"):
            nc_fret = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=non_causal_mask,
                is_causal=True,
            )
            cc_fret = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask,
                is_causal=True,
            )
            assert all_close(cn_fret, nc_fret)
            assert all_close(nc_fret, cc_fret)

    # Check `.merge_masks` with different kinds of parameters.
    non_causal_mask = MultiheadAttention.merge_masks(
        key_padding_mask, attn_mask, False, query, key
    ).unsqueeze(1)
    causal_mask = MultiheadAttention.merge_masks(
        key_padding_mask, attn_mask, True, query, key
    ).unsqueeze(1)
    check_SDPA_and_causal(query, key, value, non_causal_mask, causal_mask)

    non_causal_mask = MultiheadAttention.merge_masks(
        key_padding_mask, None, False, query, key
    ).unsqueeze(1)
    causal_mask = MultiheadAttention.merge_masks(
        key_padding_mask, None, True, query, key
    ).unsqueeze(1)
    check_SDPA_and_causal(query, key, value, non_causal_mask, causal_mask)

    non_causal_mask = MultiheadAttention.merge_masks(
        None, attn_mask, False, query, key
    ).unsqueeze(1)
    causal_mask = MultiheadAttention.merge_masks(
        None, attn_mask, True, query, key
    ).unsqueeze(1)
    check_SDPA_and_causal(query, key, value, non_causal_mask, causal_mask)


# %%
def test_MultiHeadAttention_is_causal(torch_fkwargs):
    _dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]

    query = torch.randn(3, 4, 8, **torch_fkwargs)
    key = torch.randn(3, 6, 8, **torch_fkwargs)
    value = torch.randn(3, 6, 8, **torch_fkwargs)

    nnmha = nn.MultiheadAttention(8, 1, batch_first=True, **torch_fkwargs)
    nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(8, 1, **torch_fkwargs)
    sd = {
        "in_proj.weight": nn_sd["in_proj_weight"],
        "in_proj.bias": nn_sd["in_proj_bias"],
        "out_proj.weight": nn_sd["out_proj.weight"],
        "out_proj.bias": nn_sd["out_proj.bias"],
    }
    mha.load_state_dict(sd)
    attn_mask = torch.randint(0, 2, (4, 6), device=device).to(torch.bool)
    key_padding_mask = torch.randint(0, 2, (3, 6), device=device).to(
        torch.bool
    )

    # Construct `attn_mask` for `F.scaled_dot_product_attention` manually.
    F_attn_mask = attn_mask.logical_or(
        torch.ones(4, 6, dtype=torch.int, device=device)
        .tril(diagonal=0)
        .logical_not()
    )
    nnattn, _nnw = nnmha(
        query,
        key,
        value,
        attn_mask=F_attn_mask,
        key_padding_mask=key_padding_mask.logical_not(),
        need_weights=False,
    )
    # `is_causal` and `attn_mask` are merged.
    attn, _attn_ws = mha(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask.logical_not(),
        is_causal=True,
    )
    assert all_close(nnattn, attn, 1, 0)


# %%
def test_SimpleMHA(torch_fkwargs):
    _dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]

    bsz, slen, _tlen, mlen = 3, 4, 6, 5
    hn, esz = 1, 8
    query = torch.randn(bsz, slen, esz, **torch_fkwargs)
    key = torch.randn(bsz, mlen, esz, **torch_fkwargs)
    value = torch.randn(bsz, mlen, esz, **torch_fkwargs)

    mha = SimpleMHA(esz, hn, **torch_fkwargs)
    nnmha = nn.MultiheadAttention(esz, hn, batch_first=True, **torch_fkwargs)
    single_w = mha.state_dict()["attn_proj.weight"]
    nn_sd = nnmha.state_dict()
    # Attention: only slice on first demension is allowed, even single `:` for
    # the second demension is not allowed for the tensor assignment in GPU.
    nn_sd["in_proj_weight"][:8] = single_w
    nn_sd["in_proj_weight"][8:-8] = single_w
    torch.eye(8, out=nn_sd["in_proj_weight"][-8:], **torch_fkwargs)
    torch.eye(8, out=nn_sd["out_proj.weight"], **torch_fkwargs)
    nnmha.load_state_dict(nn_sd)

    nnattn, nnattn_ws = nnmha(query, key, value, need_weights=True)
    attn, attn_ws = mha(query, key, value)
    assert attn.size() == (bsz, slen, esz)
    assert attn_ws.size() == (bsz, slen, mlen)
    assert all_close(nnattn, attn, 1, 0)
    assert all_close(nnattn_ws, attn_ws, 1, 0)

    # Forward with key-padding-mask.
    key_padding_mask = torch.randint(0, 2, (bsz, mlen), device=device).to(
        torch.bool
    )
    nnattn, nnattn_ws = nnmha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    attn, attn_ws = mha(query, key, value, key_padding_mask=key_padding_mask)
    assert attn.size() == (bsz, slen, esz)
    assert attn_ws.size() == (bsz, slen, mlen)
    assert all_close(nnattn, attn, 1, 0)
    assert all_close(nnattn_ws, attn_ws, 1, 0)

    # Forward with attention-mask only.
    attn_mask = torch.randint(0, 2, (slen, mlen), device=device).to(torch.bool)
    nnattn, nnattn_ws = nnmha(
        query,
        key,
        value,
        attn_mask=attn_mask,
        need_weights=True,
    )
    attn, attn_ws = mha(
        query,
        key,
        value,
        attn_mask=attn_mask,
    )
    assert attn.size() == (bsz, slen, esz)
    assert attn_ws.size() == (bsz, slen, mlen)
    assert all_close(nnattn, attn, 1, 0)
    assert all_close(nnattn_ws, attn_ws, 1, 0)

    # Forward with attention-mask(or mixed mask).
    nnattn, nnattn_ws = nnmha(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    attn, attn_ws = mha(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
    )
    assert attn.size() == (bsz, slen, esz)
    assert attn_ws.size() == (bsz, slen, mlen)
    assert all_close(nnattn, attn, 1, 0)
    assert all_close(nnattn_ws, attn_ws, 1, 0)


# %%
def test_SimpleMHA_qkv_diffsz(torch_fkwargs):
    bsz, slen, _tlen, mlen = 3, 4, 6, 5
    _hn, qksz, vsz = 2, 8, 16
    query = torch.randn(bsz, slen, qksz, **torch_fkwargs)
    key = torch.randn(bsz, mlen, qksz, **torch_fkwargs)
    value = torch.randn(bsz, mlen, vsz, **torch_fkwargs)

    mha = SimpleMHA(8, 1, vsz=vsz, **torch_fkwargs)
    attn, attn_ws = mha(query, key, value)
    assert attn.size() == (bsz, slen, vsz)
    assert attn_ws.size() == (bsz, slen, mlen)

    mha = SimpleMHA(8, 1, vsz=vsz, out_proj=True, **torch_fkwargs)
    attn, attn_ws = mha(query, key, value)
    assert attn.size() == (bsz, slen, qksz)
    assert attn_ws.size() == (bsz, slen, mlen)


# %%
class TestInferMaskShapes:
    """Tests for _infer_mask_shapes."""

    def test_no_masks_no_causal(self):
        """All inputs None, no causal -> default shapes."""
        bsz, qslen, kslen, device = _infer_mask_shapes(
            None, None, False, None, None
        )
        assert bsz == 1
        assert qslen == 1
        assert kslen == 1
        assert device is None

    def test_key_padding_mask_infer_bsz_kslen(self):
        """key_padding_mask provides bsz and kslen."""
        kpm = torch.zeros(5, 8, dtype=torch.bool)
        bsz, qslen, kslen, _device = _infer_mask_shapes(
            kpm, None, False, None, None
        )
        assert bsz == 5
        assert qslen == 1
        assert kslen == 8

    def test_3d_attn_mask_infer_all(self):
        """3D attn_mask provides bsz, qslen, kslen."""
        attn = torch.zeros(3, 4, 6)
        bsz, qslen, kslen, _device = _infer_mask_shapes(
            None, attn, False, None, None
        )
        assert bsz == 3
        assert qslen == 4
        assert kslen == 6

    def test_2d_attn_mask_qslen_kslen(self):
        """2D attn_mask provides qslen, kslen but not bsz."""
        attn = torch.zeros(4, 6)
        bsz, qslen, kslen, _device = _infer_mask_shapes(
            None, attn, False, None, None
        )
        assert bsz == 1
        assert qslen == 4
        assert kslen == 6

    def test_causal_with_query(self):
        """is_causal=True, query provided -> qslen from query."""
        query = torch.randn(2, 7, 5)
        key = torch.randn(2, 9, 5)
        bsz, qslen, kslen, _device = _infer_mask_shapes(
            None, None, True, query, key
        )
        assert bsz == 1
        assert qslen == 7
        assert kslen == 9

    def test_device_from_query(self):
        """Device is inferred from the first non-None tensor."""
        query = torch.randn(2, 4, 5)
        _bsz, _qslen, _kslen, device = _infer_mask_shapes(
            None, None, False, query, None
        )
        assert device == query.device

    def test_key_padding_mask_priority(self):
        """key_padding_mask takes priority over attn_mask for bsz."""
        kpm = torch.zeros(7, 6, dtype=torch.bool)
        attn = torch.zeros(3, 4, 6)
        bsz, qslen, kslen, _device = _infer_mask_shapes(
            kpm, attn, False, None, None
        )
        assert bsz == 7
        assert qslen == 4
        assert kslen == 6


# %%
class TestMergeToBias:
    """Tests for _merge_to_bias."""

    def test_no_masks_no_causal(self):
        """No masks, no causal -> all zeros."""
        shapes = (2, 4, 6, torch.device("cpu"))
        bias = _merge_to_bias(None, None, False, shapes)
        assert bias.shape == (2, 4, 6)
        assert (bias == 0).all()

    def test_causal_mask(self):
        """is_causal=True -> lower-triangular -inf."""
        shapes = (1, 4, 4, torch.device("cpu"))
        bias = _merge_to_bias(None, None, True, shapes)
        assert bias.shape == (1, 4, 4)
        # Upper triangle should be -inf.
        assert bias[0, 0, 1] == float("-inf")
        assert bias[0, 0, 3] == float("-inf")
        assert bias[0, 1, 2] == float("-inf")
        # Diagonal and below should be 0.
        assert bias[0, 0, 0] == 0
        assert bias[0, 1, 0] == 0
        assert bias[0, 1, 1] == 0
        assert bias[0, 3, 3] == 0

    def test_bool_key_padding_mask(self):
        """Bool key_padding_mask fills with -inf."""
        shapes = (2, 4, 6, torch.device("cpu"))
        # fmt: off
        kpm = torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1],
        ], dtype=torch.bool)
        # fmt: on
        bias = _merge_to_bias(kpm, None, False, shapes)
        # Batch 0: no padding -> all zeros.
        assert (bias[0] == 0).all()
        # Batch 1: padded positions should be -inf.
        assert bias[1, 0, 0] == float("-inf")
        assert bias[1, 0, 1] == float("-inf")
        assert bias[1, 0, 3] == float("-inf")
        assert bias[1, 0, 4] == float("-inf")
        assert bias[1, 0, 5] == float("-inf")
        # Non-padded position should be 0.
        assert bias[1, 0, 2] == 0

    def test_bool_attn_mask(self):
        """Bool attn_mask fills with -inf."""
        shapes = (1, 3, 4, torch.device("cpu"))
        # fmt: off
        attn = torch.tensor([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.bool)
        # fmt: on
        bias = _merge_to_bias(None, attn, False, shapes)
        assert bias[0, 0, 1] == float("-inf")
        assert bias[0, 0, 3] == float("-inf")
        assert bias[0, 1, 0] == float("-inf")
        assert bias[0, 1, 2] == float("-inf")
        assert bias[0, 2, 3] == float("-inf")
        # Non-masked positions.
        assert bias[0, 0, 0] == 0
        assert bias[0, 0, 2] == 0
        assert bias[0, 2, 0] == 0

    def test_float_attn_mask_additive(self):
        """Float attn_mask is added to bias."""
        shapes = (1, 2, 3, torch.device("cpu"))
        attn = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bias = _merge_to_bias(None, attn, False, shapes)
        assert torch.allclose(bias, attn.unsqueeze(0))

    def test_combined_causal_and_padding(self):
        """Causal + key_padding_mask both applied."""
        shapes = (1, 4, 4, torch.device("cpu"))
        kpm = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
        bias = _merge_to_bias(kpm, None, True, shapes)
        # Causal: upper triangle -inf.
        assert bias[0, 0, 1] == float("-inf")
        assert bias[0, 0, 2] == float("-inf")
        assert bias[0, 0, 3] == float("-inf")
        assert bias[0, 1, 2] == float("-inf")
        assert bias[0, 1, 3] == float("-inf")
        assert bias[0, 2, 3] == float("-inf")
        # Padding: column 2 is -inf for all rows.
        assert bias[0, 0, 2] == float("-inf")
        assert bias[0, 1, 2] == float("-inf")
        assert bias[0, 2, 2] == float("-inf")
        assert bias[0, 3, 2] == float("-inf")
        # Diagonal (0,0), (1,1), (3,3) should be 0.
        assert bias[0, 0, 0] == 0
        assert bias[0, 1, 1] == 0
        assert bias[0, 3, 3] == 0


# %%
class TestInitSdpaMask:
    """Tests for _init_sdpa_mask."""

    def test_pass_through_non_bool(self):
        """Non-bool, non-causal attn_mask returned directly."""
        attn = torch.randn(4, 6)
        query = torch.randn(3, 4, 5)
        result = _init_sdpa_mask(
            attn, False, 4, 6, torch.float32, query.device, query
        )
        assert result is attn

    def test_none_mask_no_causal(self):
        """No mask, no causal -> all zeros."""
        query = torch.randn(3, 4, 5)
        result = _init_sdpa_mask(
            None, False, 4, 6, torch.float32, query.device, query
        )
        assert result.shape == (4, 6)
        assert (result == 0).all()

    def test_causal_only(self):
        """is_causal=True, no attn_mask -> lower-triangular -inf."""
        query = torch.randn(3, 4, 4)
        result = _init_sdpa_mask(
            None, True, 4, 4, torch.float32, query.device, query
        )
        assert result.shape == (4, 4)
        # Upper triangle should be -inf.
        assert result[0, 1] == float("-inf")
        assert result[0, 3] == float("-inf")
        assert result[1, 2] == float("-inf")
        # Diagonal should be 0.
        assert result[0, 0] == 0
        assert result[1, 1] == 0
        assert result[3, 3] == 0

    def test_bool_attn_mask(self):
        """Bool attn_mask -> masked positions get -inf."""
        query = torch.randn(3, 3, 4)
        # fmt: off
        attn = torch.tensor([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.bool)
        # fmt: on
        result = _init_sdpa_mask(
            attn, False, 3, 4, torch.float32, query.device, query
        )
        assert result[0, 1] == float("-inf")
        assert result[0, 3] == float("-inf")
        assert result[0, 0] == 0

    def test_3d_attn_mask(self):
        """3D attn_mask broadcast and merged."""
        query = torch.randn(2, 4, 5)
        attn = torch.randn(2, 4, 6)
        result = _init_sdpa_mask(
            attn, False, 4, 6, torch.float32, query.device, query
        )
        assert result.shape == (2, 4, 6)
        assert torch.allclose(result, attn)

    def test_causal_with_bool_attn_mask(self):
        """Causal + bool attn_mask both applied."""
        query = torch.randn(1, 3, 3)
        # fmt: off
        attn = torch.tensor([
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
        ], dtype=torch.bool)
        # fmt: on
        result = _init_sdpa_mask(
            attn, True, 3, 3, torch.float32, query.device, query
        )
        # Causal: upper triangle -inf.
        assert result[0, 1] == float("-inf")
        assert result[0, 2] == float("-inf")
        assert result[1, 2] == float("-inf")
        # attn_mask: (0,2) and (2,0) -inf.
        assert result[0, 2] == float("-inf")
        assert result[2, 0] == float("-inf")
        # Diagonal should be 0 (not masked by attn_mask).
        assert result[0, 0] == 0
        assert result[1, 1] == 0
        assert result[2, 2] == 0
