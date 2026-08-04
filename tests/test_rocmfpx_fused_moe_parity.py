"""GPU parity coverage for the ROCmFPX fused MoE kernels.

Every ROCmFPX quant type with a fused MoE kernel is compared against an
independent dequantize-plus-dense reference. The block decoders below are
transcribed from the ABI in ``csrc/gguf/ggml-common.h`` and the per-format
kernel docstrings; they deliberately import nothing from the production
decoders so a shared assumption cannot hide a decode error.

Mixed-quant cases matter because real ROCmFPX exports do not use one type per
layer. Read directly from published GGUF headers:

    Qwen3.6-14B-A3B-ROCmFPX-STRIX_LEAN   gate/up/down all 101
    Qwen-AgentWorld-35B-A3B Q6_0_ROCMFPX gate/up 102, down {102, 103}
    Qwen-AgentWorld-35B-A3B Q4_0_ROCMFP4 up 100, gate {13, 100}, down {13, 14}

Those cases go through ``ggml_moe_a8_triton`` rather than a kernel directly, so
a per-tensor dispatch regression surfaces as a numerical mismatch instead of
passing unnoticed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch

from vllm_gguf_plugin.triton.fused_moe.interface import ggml_moe_a8_triton

BLOCK_SIZE = 32
BLOCK_M = 4

CODEBOOK10 = (0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10)
CODEBOOK_3BIT = (0.0, 1.0, 2.0, 4.0, 0.0, -1.0, -2.0, -4.0)
CODEBOOK_2BIT = (-4.0, -1.0, 1.0, 4.0)


def _cuda_or_rocm() -> bool:
    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(
    not _cuda_or_rocm(), reason="requires CUDA or ROCm Triton execution"
)


def _ue4m3_scale(scale_byte: int) -> float:
    """UE4M3 half-scale, exponent bias 8. Reserved 0x7F..0xFF decode to zero."""
    if not 0 <= scale_byte <= 0x7E:
        return 0.0
    exponent, mantissa = (scale_byte >> 3) & 0x0F, scale_byte & 0x07
    if exponent == 0:
        return mantissa / 1024.0
    return (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 8))


def _decode_100(block: Sequence[int]) -> list[float]:
    """32 x 4-bit codebook indices; contiguous halves, one scale per half."""
    low_scale, high_scale = _ue4m3_scale(block[16]), _ue4m3_scale(block[17])
    low = [CODEBOOK10[block[j] & 0x0F] * low_scale for j in range(16)]
    high = [CODEBOOK10[block[j] >> 4] * high_scale for j in range(16)]
    return low + high


def _decode_101(block: Sequence[int]) -> list[float]:
    """Same nibble layout as type 100, but a single shared scale byte."""
    scale = _ue4m3_scale(block[16])
    low = [CODEBOOK10[block[j] & 0x0F] * scale for j in range(16)]
    high = [CODEBOOK10[block[j] >> 4] * scale for j in range(16)]
    return low + high


def _decode_102(block: Sequence[int]) -> list[float]:
    """32 x 6-bit sign-magnitude codes over 24 bytes; two half-scales.

    Bit 5 is the sign and bits 0..4 the magnitude. A set sign bit with zero
    magnitude encodes -32 rather than negative zero.
    """
    bits = 0
    for index, byte in enumerate(block[:24]):
        bits |= byte << (8 * index)
    scales = (_ue4m3_scale(block[24]), _ue4m3_scale(block[25]))
    values = []
    for j in range(BLOCK_SIZE):
        code = (bits >> (6 * j)) & 0x3F
        magnitude = code & 31
        signed = -(magnitude if magnitude else 32) if code & 32 else magnitude
        values.append(signed * scales[0 if j < 16 else 1])
    return values


def _decode_103(block: Sequence[int]) -> list[float]:
    """32 signed int8 codes plus one shared scale byte."""
    scale = _ue4m3_scale(block[32])
    return [(byte - 256 if byte > 127 else byte) * scale for byte in block[:32]]


def _decode_104(block: Sequence[int]) -> list[float]:
    """32 x 3-bit codebook indices at bits [3j, 3j+3) over 12 bytes."""
    bits = 0
    for index, byte in enumerate(block[:12]):
        bits |= byte << (8 * index)
    scales = (_ue4m3_scale(block[12]), _ue4m3_scale(block[13]))
    return [
        CODEBOOK_3BIT[(bits >> (3 * j)) & 7] * scales[0 if j < 16 else 1]
        for j in range(BLOCK_SIZE)
    ]


def _decode_107(block: Sequence[int]) -> list[float]:
    """32 x 2-bit codebook indices, four per byte, sequential."""
    scales = (_ue4m3_scale(block[8]), _ue4m3_scale(block[9]))
    values = []
    for j in range(BLOCK_SIZE):
        code = (block[j // 4] >> (2 * (j % 4))) & 3
        values.append(CODEBOOK_2BIT[code] * scales[0 if j < 16 else 1])
    return values


# quant type -> (block bytes, trailing scale bytes, decoder, format name)
SPECS: dict[int, tuple[int, int, Callable[[Sequence[int]], list[float]], str]] = {
    100: (18, 2, _decode_100, "Q4_0_ROCMFP4"),
    101: (17, 1, _decode_101, "Q4_0_ROCMFP4_FAST"),
    102: (26, 2, _decode_102, "Q6_0_ROCMFPX"),
    103: (33, 1, _decode_103, "Q8_0_ROCMFPX"),
    104: (14, 2, _decode_104, "Q3_0_ROCMFPX"),
    107: (10, 2, _decode_107, "Q2_0_ROCMFPX"),
}


def _pack(quant_type: int, experts: int, rows: int, blocks: int) -> torch.Tensor:
    """Random packed weights whose scale bytes stay in the valid 0x00..0x7E range.

    Reserved scale handling is covered by the dequantize tests; here the goal is
    real magnitudes rather than the zero path, so every block gets a live scale.
    """
    block_bytes, scale_bytes, _, _ = SPECS[quant_type]
    weights = torch.randint(
        0,
        256,
        (experts, rows, blocks * block_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    payload = block_bytes - scale_bytes
    for block in range(blocks):
        base = block * block_bytes
        for offset in range(scale_bytes):
            weights[:, :, base + payload + offset] = torch.randint(
                0x30, 0x50, (experts, rows), dtype=torch.uint8, device="cuda"
            )
    return weights


def _dequantize(weights: torch.Tensor, quant_type: int) -> torch.Tensor:
    """[experts, rows, packed] uint8 -> [experts, rows, hidden] float32."""
    block_bytes, _, decoder, _ = SPECS[quant_type]
    experts, rows, packed = weights.shape
    blocks = packed // block_bytes
    out = torch.zeros((experts, rows, blocks * BLOCK_SIZE), dtype=torch.float32)
    raw = weights.cpu().tolist()
    for expert in range(experts):
        for row in range(rows):
            packed_row = raw[expert][row]
            values: list[float] = []
            for block in range(blocks):
                start = block * block_bytes
                values.extend(decoder(packed_row[start : start + block_bytes]))
            out[expert, row] = torch.tensor(values, dtype=torch.float32)
    return out.to(weights.device)


def _align_tokens(topk_ids: torch.Tensor, experts: int):
    """Group flat expert slots into per-expert blocks, padding each to BLOCK_M.

    Mirrors ``moe_align_block_size``: the kernel derives its token offsets from
    ``sorted_token_ids``, so those must be flat slot indices (token * top_k + k),
    and no block may straddle two experts.
    """
    flat = topk_ids.reshape(-1)
    num_valid = flat.numel()
    sorted_ids: list[int] = []
    expert_ids: list[int] = []
    for expert in range(experts):
        slots = (flat == expert).nonzero(as_tuple=True)[0].tolist()
        for start in range(0, len(slots), BLOCK_M):
            chunk = slots[start : start + BLOCK_M]
            sorted_ids.extend(chunk + [num_valid] * (BLOCK_M - len(chunk)))
            expert_ids.append(expert)
    device = topk_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=device),
        torch.tensor(expert_ids, dtype=torch.int32, device=device),
        torch.tensor(len(sorted_ids), dtype=torch.int32, device=device),
    )


def _reference_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inter: int,
) -> torch.Tensor:
    """Dequantize then dense-matmul per routed slot, one token at a time."""
    tokens, hidden = x.shape
    out = torch.zeros((tokens, hidden), dtype=torch.float32, device=x.device)
    for token in range(tokens):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            projected = x[token] @ w13[expert].T
            gate, up = projected[:inter], projected[inter:]
            out[token] += (gate * up @ w2[expert].T) * topk_weights[token, slot]
    return out


def _fused_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_type: int,
    w2_type: int,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inter: int,
) -> torch.Tensor:
    """Two fused matmuls through the public dispatcher, matching _fused_moe_gguf."""
    tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    experts = w13.shape[0]
    sorted_ids, expert_ids, padded = _align_tokens(topk_ids, experts)

    first = ggml_moe_a8_triton(
        x, w13, sorted_ids, expert_ids, padded, w13_type, 2 * inter, top_k, tokens
    )
    activated = first[:, :inter] * first[:, inter:]
    second = ggml_moe_a8_triton(
        activated,
        w2,
        sorted_ids,
        expert_ids,
        padded,
        w2_type,
        hidden,
        1,
        tokens * top_k,
    )
    weighted = second.reshape(tokens, top_k, hidden) * topk_weights.unsqueeze(-1)
    return weighted.sum(dim=1)


def _assert_moe_parity(w13_type: int, w2_type: int, seed: int = 0) -> None:
    torch.manual_seed(seed)
    experts, tokens, hidden, inter, top_k = 4, 8, 64, 64, 2

    w13 = _pack(w13_type, experts, 2 * inter, hidden // BLOCK_SIZE)
    w2 = _pack(w2_type, experts, hidden, inter // BLOCK_SIZE)
    x = torch.randn((tokens, hidden), dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(
        0, experts, (tokens, top_k), dtype=torch.int32, device="cuda"
    )
    topk_weights = torch.rand((tokens, top_k), dtype=torch.float32, device="cuda")

    expected = _reference_moe(
        x,
        _dequantize(w13, w13_type),
        _dequantize(w2, w2_type),
        topk_ids,
        topk_weights,
        inter,
    )
    actual = _fused_moe(x, w13, w2, w13_type, w2_type, topk_ids, topk_weights, inter)

    # Relative tolerance: the reference accumulates per token while the kernel
    # accumulates over K tiles, so FP32 ordering differs. Observed worst case is
    # 1.2e-3 on GB10 and exact on gfx1151.
    scale = expected.abs().max().item() + 1e-6
    assert (expected - actual).abs().max().item() / scale < 1e-2


@requires_gpu
@pytest.mark.parametrize("quant_type", sorted(SPECS))
def test_rocmfpx_fused_moe_matches_independent_reference(quant_type: int) -> None:
    """Each ROCmFPX MoE kernel must match a dequantize-plus-dense reference."""
    _assert_moe_parity(quant_type, quant_type)


@requires_gpu
@pytest.mark.parametrize(
    "w13_type,w2_type",
    [
        pytest.param(102, 103, id="agentworld_q6_gate_up_102_down_103"),
        pytest.param(100, 101, id="rocmfp4_pair_100_101"),
        pytest.param(104, 107, id="narrow_pair_104_107"),
        pytest.param(101, 102, id="mixed_width_101_102"),
    ],
)
def test_mixed_quant_moe_layers_dispatch_per_tensor(
    w13_type: int, w2_type: int
) -> None:
    """w13 and w2 carrying different quant types must each select their kernel.

    Published ROCmFPX exports mix types across the expert tensors, so the
    dispatcher has to choose per tensor. Selecting one kernel for both sides
    would decode one of them with the wrong ABI and diverge here.
    """
    _assert_moe_parity(w13_type, w2_type)
