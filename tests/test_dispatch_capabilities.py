import vllm_gguf_plugin.ops as ops


def test_rocmfpx_native_dispatch_capabilities_are_fail_closed():
    supported_rocmfpx = {100, 102, 103, 104, 107}
    assert supported_rocmfpx.isdisjoint(ops._CUDA_GEMV_QUANT_TYPES)
    assert supported_rocmfpx <= ops._CUDA_DEQUANT_ONLY_TYPES
    assert 101 not in ops._CUDA_GEMV_QUANT_TYPES
    assert 101 in ops._CUDA_DEQUANT_ONLY_TYPES


def test_custom_dispatch_capabilities_are_truthful():
    custom_ids = {
        136,
        142,
        143,
        147,
        148,
        149,
        150,
        151,
    }
    portable_dequant_ids = {
        36,
        41,
        42,
        100,
        101,
        102,
        103,
        104,
        107,
        133,
        134,
        135,
        137,
        138,
        139,
        140,
        141,
        144,
        145,
        146,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
    }

    assert custom_ids.isdisjoint(ops._CUDA_GEMV_QUANT_TYPES)
    assert custom_ids.isdisjoint(ops._CUDA_GEMM_QUANT_TYPES)
    assert custom_ids.isdisjoint(ops._CUDA_DEQUANT_ONLY_TYPES - {36, 41, 42})
    assert portable_dequant_ids <= ops._CUDA_DEQUANT_ONLY_TYPES
    assert {136, 142, 143, 147, 148, 149, 150, 151}.isdisjoint(
        ops._CUDA_DEQUANT_ONLY_TYPES
    )
