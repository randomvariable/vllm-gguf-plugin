import pytest
import torch

from vllm_gguf_plugin.triton.dequantize import interface
from vllm_gguf_plugin.triton.dequantize.ik_k56_reference import (
    REFERENCE_DECODERS as K56_REFERENCE_DECODERS,
)
from vllm_gguf_plugin.triton.dequantize.ik_k234_reference import (
    REFERENCE_DECODERS as K234_REFERENCE_DECODERS,
)
from vllm_gguf_plugin.triton.dequantize.ik_ks_reference import (
    REFERENCE_DECODERS as KS_REFERENCE_DECODERS,
)
from vllm_gguf_plugin.triton.dequantize.ik_kss_kl_reference import (
    REFERENCE_DECODERS as KSS_KL_REFERENCE_DECODERS,
)


def test_reference_dispatch_merges_without_mutating_sources():
    k234_before = K234_REFERENCE_DECODERS.copy()
    k56_before = K56_REFERENCE_DECODERS.copy()
    ks_before = KS_REFERENCE_DECODERS.copy()
    kss_kl_before = KSS_KL_REFERENCE_DECODERS.copy()

    expected = {
        **K234_REFERENCE_DECODERS,
        **K56_REFERENCE_DECODERS,
        **KS_REFERENCE_DECODERS,
        **KSS_KL_REFERENCE_DECODERS,
    }

    assert {key: interface.REFERENCE_DECODERS[key] for key in expected} == expected
    assert k234_before == K234_REFERENCE_DECODERS
    assert k56_before == K56_REFERENCE_DECODERS
    assert ks_before == KS_REFERENCE_DECODERS
    assert kss_kl_before == KSS_KL_REFERENCE_DECODERS


@pytest.mark.parametrize(
    ("quant_type", "block_bytes"),
    [
        (quant_type, 76 if quant_type == 137 else 110 if quant_type == 138 else 144)
        for quant_type in (137, 138, 139)
    ]
    + [
        (140, 176),
        (141, 212),
        (144, 140),
        (145, 72),
        (146, 132),
        (152, 172),
        (157, 88),
    ],
)
def test_reference_dispatch_zero_block_output_shape(quant_type, block_bytes):
    output = interface.ggml_dequantize_triton(
        torch.zeros(block_bytes, dtype=torch.uint8),
        quant_type,
        1,
        256,
        dtype=torch.float32,
    )

    assert output.shape == (1, 256)
