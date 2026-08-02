# vLLM GGUF Plugin — Homelabs Edition

A fork of the [vLLM GGUF quantization plugin](https://github.com/vllm-project/vllm-gguf-plugin).
It extends GGUF coverage with standard formats and selected `ik_llama.cpp` and
ROCmFPX formats. The goal is to serve more models from one stack, while keeping
format validation status explicit.

The default branch **`homelabs-main`** carries the extended quant support; the
upstream plugin documentation is preserved at [`UPSTREAM.md`](UPSTREAM.md).

## Why This Fork

The upstream plugin supports the standard llama.cpp quant types. This fork adds
the formats that are popular in the community but missing upstream. Software
fallback dequantization exists for these formats; type 101 also has native
CUDA/HIP dequantization, but no native GEMV/GEMM/MoE kernels. They therefore run
on consumer GPUs without FP8/FP4 tensor cores:

| Target | GPU | Notes |
|---|---|---|
| **AMD Strix Halo** | Radeon 8060S, `gfx1151` / RDNA3.5 | No FP8/FP4 tensor cores — software-dequant formats are the path to large models in 96 GB UMA |
| **NVIDIA DGX Spark** | Grace Blackwell, `sm_121a` | Software-dequant formats; `MXFP4`/`NVFP4` support is deferred |

## Quantization Coverage

Upstream plugin formats (unchanged): `Q4_0`–`Q8_1`, `Q2_K`–`Q6_K`.
| Status | Formats | Evidence and limits |
|---|---|---|
| **Verified** | Standard formats: `Q4_0`-`Q8_1`, `Q2_K`-`Q6_K`, `IQ1_S`-`IQ4_XS` | Implemented by upstream plugin and retained here |
| **Limited** | ROCmFPX: `Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST` (type 101), `Q2_0_ROCMFPX`, `Q3_0_ROCMFPX`, `Q6_0_ROCMFPX`, `Q8_0_ROCMFPX` | Dequantization and materialized dense linear path; type 101 has native CUDA/HIP and software fallback, but no native GEMV/GEMM/MoE |
| **Limited** | ik_llama K/KS/KT: `IQ*_K`, `IQ*_KS`, `IQ*_KT`; `TQ1_0`, `TQ2_0`, `Q2_0` | Wired and registered, but complete trusted-reference and end-to-end validation is still missing |
| **Limited** | Type-41 `Q1_0` / `Q1_0_G128` | Accepted as aliases only with geometry `(128, 18)`; incompatible type-41 geometry is rejected; native runtime validation remains pending |
| **Deferred** | `MXFP4`, `NVFP4` | Not implemented; do not advertise as supported |

This table describes repository implementation status, not universal model
compatibility. A model may still require compatible tensor names, shapes, and
runtime/backend coverage.
## Installation

The plugin is baked into the homelab vLLM runtime images
(`vllm-strix-runtime` / `vllm-spark-runtime`), so GGUF serving works out of the
box there. To install standalone:

```bash
git clone https://github.com/randomvariable/vllm-gguf-plugin
cd vllm-gguf-plugin
git checkout homelabs-main
uv pip install -e . --torch-backend=auto
```

## Usage

Serve a GGUF model — the plugin registers its quant types with vLLM at import:

```bash
vllm serve <repo>/<model>-GGUF:Q4_0_ROCMFP4 --tokenizer <repo>/<model>
```

## For Contributors

`homelabs-main` is the consolidated branch; per-format work lands on feature
branches (e.g. `rocmfpx-q4_0`, `ik-iqk`) and merges in once reviewed. Each new
quant type adds a block struct, a dequant kernel, a dispatch case, and a
type-registration entry — see the ROCmFPX commits for the pattern. Upstream
development workflow is documented in [`UPSTREAM.md`](UPSTREAM.md).

## License

Apache 2.0, as upstream.
