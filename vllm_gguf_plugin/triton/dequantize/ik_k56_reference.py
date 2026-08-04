"""CPU/PyTorch reference decoders for ik_llama IQ5_K and IQ6_K."""

from __future__ import annotations

import torch

from ...ik_types import GGML_TYPE_IQ5_K, GGML_TYPE_IQ6_K

QK = 256
IQ5_K_BLOCK_BYTES = 176
IQ6_K_BLOCK_BYTES = 212

_IQ5_CODEBOOK = torch.tensor(
    [
        -126,
        -114,
        -103,
        -92,
        -83,
        -74,
        -65,
        -57,
        -50,
        -43,
        -36,
        -30,
        -24,
        -18,
        -12,
        -6,
        -1,
        5,
        11,
        17,
        23,
        29,
        36,
        43,
        51,
        59,
        68,
        77,
        87,
        97,
        109,
        121,
        -124,
        -112,
        -101,
        -90,
        -81,
        -72,
        -63,
        -55,
        -48,
        -41,
        -34,
        -28,
        -22,
        -16,
        -10,
        -4,
        1,
        7,
        13,
        19,
        25,
        31,
        38,
        45,
        53,
        61,
        70,
        79,
        89,
        99,
        111,
        123,
    ],
    dtype=torch.float32,
)
_IQ6_CODEBOOK = torch.tensor(
    [
        -127,
        -121,
        -115,
        -109,
        -104,
        -98,
        -93,
        -88,
        -84,
        -79,
        -74,
        -70,
        -66,
        -62,
        -58,
        -54,
        -51,
        -47,
        -44,
        -40,
        -37,
        -34,
        -31,
        -28,
        -25,
        -22,
        -19,
        -16,
        -13,
        -11,
        -8,
        -5,
        -2,
        0,
        3,
        6,
        9,
        12,
        14,
        17,
        20,
        23,
        27,
        30,
        33,
        36,
        40,
        44,
        47,
        51,
        55,
        59,
        63,
        68,
        72,
        77,
        82,
        87,
        92,
        98,
        103,
        109,
        115,
        121,
        -126,
        -120,
        -114,
        -108,
        -103,
        -97,
        -92,
        -87,
        -83,
        -78,
        -73,
        -69,
        -65,
        -61,
        -57,
        -53,
        -50,
        -46,
        -43,
        -39,
        -36,
        -33,
        -30,
        -27,
        -24,
        -21,
        -18,
        -15,
        -12,
        -10,
        -7,
        -4,
        -1,
        1,
        4,
        7,
        10,
        13,
        15,
        18,
        21,
        24,
        28,
        31,
        34,
        37,
        41,
        45,
        48,
        52,
        56,
        60,
        64,
        69,
        73,
        78,
        83,
        88,
        93,
        99,
        104,
        110,
        116,
        122,
    ],
    dtype=torch.float32,
)


def _validate(W: torch.Tensor, m: int, n: int, block_bytes: int) -> torch.Tensor:
    if W.dtype is not torch.uint8:
        raise TypeError(f"Quantized weights must be torch.uint8, got {W.dtype}")
    if m <= 0 or n <= 0:
        raise ValueError(
            f"Reference dequant shape must have positive dimensions, got ({m}, {n})"
        )
    if n % QK:
        raise ValueError(
            f"Reference dequant columns must be divisible by {QK}, got {n}"
        )
    expected = m * (n // QK) * block_bytes
    if W.numel() != expected:
        raise ValueError(
            f"Quantized weights must contain exactly {expected} bytes, got {W.numel()}"
        )
    return W.contiguous().reshape(-1, block_bytes)


def dequantize_iq5_k_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Decode packed IQ5_K blocks using the ik_llama byte ordering."""
    blocks = _validate(W, m, n, IQ5_K_BLOCK_BYTES)
    codebook = _IQ5_CODEBOOK.to(W.device)
    d = blocks[:, :2].contiguous().view(torch.float16).squeeze(1).to(torch.float32)
    extra = blocks[:, 2].to(torch.int32) | (blocks[:, 3].to(torch.int32) << 8)
    # Promote to int32 before arithmetic: uint8 wraps on the -32 bias and on
    # left shifts, silently corrupting every scale and packed code.
    scales_h = blocks[:, 4:8].to(torch.int32)
    scales_l = blocks[:, 8:16].to(torch.int32)
    qs = blocks[:, 16:144].to(torch.int32)
    qh = blocks[:, 144:176].to(torch.int32)
    groups = []
    for ib32 in range(8):
        ib64, hi = ib32 // 2, ib32 & 1
        shift = 2 * ib64
        sh, sl = scales_h[:, ib64], scales_l[:, 2 * ib64 + hi]
        dl1 = d * (
            ((sl & 15) | ((sh >> (0 if hi else 0)) & 48 if hi else (sh << 4) & 48)) - 32
        )
        dl2 = d * (((sl >> 4) | ((sh >> 2) & 48 if hi else (sh << 2) & 48)) - 32)
        code = extra >> (4 * ib64 + 2 * hi)
        source = qs[:, 32 * ib64 : 32 * ib64 + 32]
        high = qh
        q1 = (source[:, :16] >> (4 if hi else 0)) & 15
        q2 = (source[:, 16:] >> (4 if hi else 0)) & 15
        q1 |= ((high[:, :16] >> shift) & (2 if hi else 1)) << (3 if hi else 4)
        q2 |= ((high[:, 16:] >> shift) & (2 if hi else 1)) << (3 if hi else 4)
        # code is (nblocks,) while q1/q2 are (nblocks, 16): the offset must be
        # unsqueezed or broadcasting only works for a single block.
        values1 = codebook[((code & 1) * 32)[:, None] + q1]
        values2 = codebook[((code & 2) * 16)[:, None] + q2]
        groups.append(
            torch.cat((dl1[:, None] * values1, dl2[:, None] * values2), dim=1)
        )
    return torch.cat(groups, dim=1).reshape(m, n).to(dtype or torch.float16)


def dequantize_iq6_k_reference(
    W: torch.Tensor, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Decode packed IQ6_K blocks, including its dual 6-bit codebook."""
    blocks = _validate(W, m, n, IQ6_K_BLOCK_BYTES)
    codebook = _IQ6_CODEBOOK.to(W.device)
    d = blocks[:, :2].contiguous().view(torch.float16).squeeze(1).to(torch.float32)
    extra = blocks[:, 2].to(torch.int32) | (blocks[:, 3].to(torch.int32) << 8)
    # block_iq6_k.scales is int8_t: reinterpret before widening, otherwise
    # negative scales read back as 128..255.
    scales = blocks[:, 4:20].contiguous().view(torch.int8).to(torch.int32)
    qs = blocks[:, 20:148].to(torch.int32)
    qh = blocks[:, 148:212].to(torch.int32)
    groups = []
    for ib32 in range(8):
        ib64, hi = ib32 // 2, ib32 & 1
        shift = 4 * (ib64 & 1)
        source = qs[:, 32 * ib64 : 32 * ib64 + 32]
        high = qh[:, 32 * (ib64 // 2) : 32 * (ib64 // 2) + 32]
        q1 = (source[:, :16] >> (4 if hi else 0)) & 15
        q2 = (source[:, 16:] >> (4 if hi else 0)) & 15
        q1 |= ((high[:, :16] >> shift) & (12 if hi else 3)) << (2 if hi else 4)
        q2 |= ((high[:, 16:] >> shift) & (12 if hi else 3)) << (2 if hi else 4)
        code = extra >> (4 * ib64 + 2 * hi)
        # code is (nblocks,) while q1/q2 are (nblocks, 16): the offset must be
        # unsqueezed or broadcasting only works for a single block.
        values1 = codebook[((code & 1) * 64)[:, None] + q1]
        values2 = codebook[((code & 2) * 32)[:, None] + q2]
        groups.append(
            torch.cat(
                (
                    d[:, None] * scales[:, 4 * ib64 + 2 * hi, None] * values1,
                    d[:, None] * scales[:, 4 * ib64 + 2 * hi + 1, None] * values2,
                ),
                dim=1,
            )
        )
    return torch.cat(groups, dim=1).reshape(m, n).to(dtype or torch.float16)


REFERENCE_DECODERS = {
    GGML_TYPE_IQ5_K: dequantize_iq5_k_reference,
    GGML_TYPE_IQ6_K: dequantize_iq6_k_reference,
}
