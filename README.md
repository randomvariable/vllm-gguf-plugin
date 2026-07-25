# vLLM GGUF Plugin — Homelabs Edition

A fork of the
[vLLM GGUF quantization plugin](https://github.com/vllm-project/vllm-gguf-plugin)
that extends GGUF coverage so you can serve **any GGUF model that llama.cpp,
ik_llama.cpp, or ROCmFPX can** — straight from vLLM. The goal is to let a
homelab retire its llama.cpp instances and serve everything from one stack.

The default branch **`homelabs-main`** carries the extended quant support; the
upstream plugin documentation is preserved at [`UPSTREAM.md`](UPSTREAM.md).

## Why This Fork

The upstream plugin supports the standard llama.cpp quant types. This fork adds
the formats that are popular in the community but missing upstream, all of which
dequantize in pure software (lookup-table + integer arithmetic) and therefore
run on consumer GPUs without FP8/FP4 tensor cores:

| Target | GPU | Notes |
|---|---|---|
| **AMD Strix Halo** | Radeon 8060S, `gfx1151` / RDNA3.5 | No FP8/FP4 tensor cores — software-dequant formats are the path to large models in 96 GB UMA |
| **NVIDIA DGX Spark** | Grace Blackwell, `sm_121a` | Software-dequant formats plus FP4/FP8 tensor-core types |

## Quantization Coverage

**Supported by the upstream plugin** (unchanged): `Q4_0`–`Q8_1`, `Q2_K`–`Q6_K`,
and the i-quants `IQ1_S`–`IQ4_XS`.

**Added on `homelabs-main`:**

- **ROCmFPX family** — Strix-Halo-optimised codebook formats, all six model-weight
  types: `Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`, `Q2_0_ROCMFPX` (the format used by
  Hy3-iFP2 models), `Q3_0_ROCMFPX`, `Q6_0_ROCMFPX`, `Q8_0_ROCMFPX`.
- **ik_llama.cpp i-quants** *(in progress)* — the SOTA low-bit K-variant i-quants
  (`IQ4_KS`, `IQ4_K`, `IQ2_K`, `IQ3_K` first, then the wider `IQ*_K`/`KS`/`KT`
  family), which unlock the most community-quantized models.

**Planned:** the remaining ik_llama formats, the llama.cpp upstream gap
(`TQ1_0`/`TQ2_0` ternary, `Q1_0`/`Q2_0`), and the DGX Spark tensor-core types
(`MXFP4`/`NVFP4`).

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
