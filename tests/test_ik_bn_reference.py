import struct

import gguf
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from vllm_gguf_plugin.ik_types import (
    GGML_TYPE_IQ1_BN,
    GGML_TYPE_IQ2_BN,
    IQ1_BN_BLOCK_BYTES,
    IQ2_BN_BLOCK_BYTES,
    QK_IQ1BN,
    QK_IQ2BN,
)
from vllm_gguf_plugin.triton.dequantize.ik_bn_reference import (
    dequantize_iq1_bn_reference,
    dequantize_iq2_bn_reference,
)
from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton


def test_bn_gguf_geometry_is_not_registered_as_fixed_block_geometry():
    # GGUF's fixed (weights-per-block, bytes-per-block) API cannot represent
    # the per-row prefix without producing wrong tensor byte sizes.
    assert GGML_TYPE_IQ1_BN not in gguf.GGML_QUANT_SIZES
    assert GGML_TYPE_IQ2_BN not in gguf.GGML_QUANT_SIZES


def test_iq1_bn_uses_descending_digit_multipliers_and_extra_order():
    ql = bytes([1, 2, 3] * 4)
    raw = struct.pack("<e", 2.0) + ql + bytes([4])

    output = dequantize_iq1_bn_reference(
        torch.tensor(list(raw), dtype=torch.uint8), 1, 64, dtype=torch.float32
    )

    expected = []
    multipliers = (81, 27, 9, 3, 1)
    for group in range(4):
        for lane in range(16):
            if lane == 15:
                q, digit = 4, group
            else:
                q, digit = ql[3 * group + lane // 5], lane % 5
            value = (q * multipliers[digit] + (q * multipliers[digit] >> 1)) >> 7
            expected.append(2.0 * (value - 1))
    torch.testing.assert_close(output, torch.tensor([expected]))


def test_iq2_bn_decodes_grouped_codes_and_fp32_row_scale():
    raw = struct.pack("<f", 0.5) + bytes([0b11100100] * 16)
    output = dequantize_iq2_bn_reference(
        torch.tensor(list(raw), dtype=torch.uint8), 1, 64, dtype=torch.float32
    )
    expected = [
        0.5 * (((0b11100100 >> (2 * (lane // 16))) & 3) - 1) for lane in range(64)
    ]
    torch.testing.assert_close(output, torch.tensor([expected]))


@pytest.mark.parametrize(
    ("decoder", "prefix", "payload", "qk"),
    [
        (dequantize_iq1_bn_reference, 2, IQ1_BN_BLOCK_BYTES, QK_IQ1BN),
        (dequantize_iq2_bn_reference, 4, IQ2_BN_BLOCK_BYTES, QK_IQ2BN),
    ],
)
def test_bn_validates_exact_storage_for_multi_block_multi_row(
    decoder, prefix, payload, qk
):
    raw = torch.zeros(2 * (prefix + 2 * payload), dtype=torch.uint8)
    decoder(raw, 2, 2 * qk)
    with pytest.raises(ValueError, match="exactly"):
        decoder(raw[:-1], 2, 2 * qk)


@pytest.mark.parametrize(
    ("decoder", "prefix", "payload", "qk", "scales"),
    [
        (dequantize_iq1_bn_reference, 2, IQ1_BN_BLOCK_BYTES, QK_IQ1BN, (1.0, 2.0)),
        (dequantize_iq2_bn_reference, 4, IQ2_BN_BLOCK_BYTES, QK_IQ2BN, (1.0, 2.0)),
    ],
)
def test_bn_row_scales_broadcast_independently(
    decoder, prefix, payload, qk, scales
):
    rows = [
        struct.pack("<e", scale) if prefix == 2 else struct.pack("<f", scale)
        for scale in scales
    ]
    raw = torch.tensor(
        list(b"".join(row + bytes(payload) for row in rows)), dtype=torch.uint8
    )
    output = decoder(raw, 2, qk, dtype=torch.float32)
    torch.testing.assert_close(output[1], output[0] * 2)


@pytest.mark.parametrize(
    ("quant_type", "raw", "decoder"),
    [
        (
            GGML_TYPE_IQ1_BN,
            struct.pack("<e", 2.0) + bytes([0, 1, 2] * 4) + bytes([255]),
            dequantize_iq1_bn_reference,
        ),
        (
            GGML_TYPE_IQ2_BN,
            struct.pack("<f", 0.5) + bytes([0b11100100] * 16),
            dequantize_iq2_bn_reference,
        ),
    ],
)
def test_bn_type_ids_dispatch_through_dequantize_interface(quant_type, raw, decoder):
    weights = torch.tensor(list(raw), dtype=torch.uint8)

    output = ggml_dequantize_triton(weights, quant_type, 1, 64, dtype=torch.float32)

    expected = decoder(weights, 1, 64, dtype=torch.float32)
    assert output.shape == (1, 64)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("decoder", "payload", "prefix"),
    [
        (dequantize_iq1_bn_reference, IQ1_BN_BLOCK_BYTES, 2),
        (dequantize_iq2_bn_reference, IQ2_BN_BLOCK_BYTES, 4),
    ],
)
def test_bn_rejects_malformed_inputs(decoder, payload, prefix):
    row_bytes = prefix + payload
    with pytest.raises(ValueError, match="exactly"):
        decoder(torch.zeros(row_bytes - 1, dtype=torch.uint8), 1, 64)
    with pytest.raises(ValueError, match="positive"):
        decoder(torch.zeros(row_bytes, dtype=torch.uint8), 0, 64)
    with pytest.raises(TypeError, match="uint8"):
        decoder(torch.zeros(row_bytes, dtype=torch.float32), 1, 64)
    with pytest.raises(ValueError, match="contiguous"):
        decoder(torch.zeros(row_bytes * 2, dtype=torch.uint8)[::2], 1, 64)


@given(
    m=st.integers(min_value=1, max_value=3),
    blocks_per_row=st.integers(min_value=1, max_value=3),
    quant_type=st.sampled_from((GGML_TYPE_IQ1_BN, GGML_TYPE_IQ2_BN)),
)
def test_bn_reference_shape_finite_and_row_block_boundaries(
    m, blocks_per_row, quant_type
):
    payload = (
        IQ1_BN_BLOCK_BYTES if quant_type == GGML_TYPE_IQ1_BN else IQ2_BN_BLOCK_BYTES
    )
    prefix = 2 if quant_type == GGML_TYPE_IQ1_BN else 4
    row = bytes(prefix) + bytes(payload) * blocks_per_row
    raw = torch.tensor(list(row * m), dtype=torch.uint8)
    decoder = (
        dequantize_iq1_bn_reference
        if quant_type == GGML_TYPE_IQ1_BN
        else dequantize_iq2_bn_reference
    )
    output = decoder(raw, m, QK_IQ1BN * blocks_per_row, dtype=torch.float32)
    assert output.shape == (m, QK_IQ2BN * blocks_per_row)
    assert torch.isfinite(output).all()
