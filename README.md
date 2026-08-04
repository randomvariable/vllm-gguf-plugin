# vLLM GGUF Plugin - Homelabs Edition

Fork of the [vLLM GGUF quantization plugin](https://github.com/vllm-project/vllm-gguf-plugin).
Extends GGUF coverage with standard formats and selected `ik_llama.cpp` and
ROCmFPX formats. The goal is to serve more models from one stack while keeping
format validation status explicit.

Default branch **`homelabs-main`** carries extended quantization support. The
upstream plugin documentation is preserved in [UPSTREAM.md](UPSTREAM.md).

## Source Lineage and Authority

Format names and type IDs are not enough to establish compatibility. The
authoritative implementation determines packed layout, block geometry, lookup
tables, scale encoding, and expected decode behavior.

| Authority | Scope | Role in this fork |
|---|---|---|
| [upstream `llama.cpp`](https://github.com/ggml-org/llama.cpp) | Standard GGML/GGUF formats | Reference for standard block ABI and behavior |
| [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) | `IQ*_K`, `IQ*_KS`, `IQ*_KSS`, `IQ*_KL`, `IQ*_KT`, and trellis formats | Reference for extended i-quant and trellis layouts; this fork ports selected decoders and fallback paths |
| [upstream vLLM GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin) | vLLM integration baseline and upstream-supported GGUF formats | Baseline for weight loading, quantization registration, linear dispatch, and model integration |
| This fork | Python/Triton/native CUDA/HIP ports and fallback paths | Adds format registration, reference decoders, selected native dequantization, and explicit capability gates without silently changing an ABI |

When a format is extended or patched locally, its source lineage and geometry
must remain visible in code and documentation. This fork is not an independent
authority for a packed format merely because it accepts its type ID.

## Why This Fork

The upstream plugin supports standard `llama.cpp` quantization types. This fork
adds formats that are used in the community but are missing upstream. ROCmFPX
types 100 and 101 have native CUDA/HIP dequant-only decode plus software fallback
dequantization. ROCmFPX types 100, 102, 103, 104, and 107 have dedicated dense
Triton GEMM kernels gated on `gfx115*`, and type 101 has limited, conditional
native CUDA/HIP GEMV plus its own gated Triton GEMM. All six ROCmFPX types also
have fused Triton MoE kernels behind the same `gfx115*` gate, including layers
that mix quant types across the expert tensors. Outside that gate every ROCmFPX
format falls back to dequantize-plus-dense. No native C++/HIP extension MoE is
claimed, and only type 101 has any native GEMV. These paths therefore run on
consumer GPUs without FP8/FP4 tensor cores:

| Target | GPU | Notes |
|---|---|---|
| **AMD Strix Halo** | Radeon 8060S, `gfx1151` / RDNA3.5 | No FP8/FP4 tensor cores; software-dequant formats support large models in 96 GB UMA |
| **NVIDIA DGX Spark** | Grace Blackwell, `sm_121a` | Software-dequant formats; `MXFP4`/`NVFP4` support is deferred |

## Performance Model

The rows below describe expected work and traffic, not measured benchmark
results. A fused quantized kernel normally combines decode and dot-product
work. A dequant-only path decodes packed weights first and then uses a dense
matmul; that distinction matters more than the format name.

| Path | Arithmetic work | Temporary memory | Weight reads | Activation reads | Expected compute/bandwidth behavior |
|---|---|---|---|---|---|
| Upstream/native quantized GEMV/GEMM | Quantized dot product plus on-the-fly scale/codebook decode; no separate dense decode pass | No full dense weight temporary | Packed bytes read by the fused kernel; usually one main weight stream | Input vector/matrix read by the fused kernel | Lowest weight traffic for a matching ABI; often bandwidth-sensitive for small batches and compute/decode-sensitive for larger batches |
| This fork native dequant-only | Device-side decode followed by dense matmul; arithmetic includes decode and dense dot product | Decoded dense matrix for the active operation | Packed bytes are read, then decoded values are read again by matmul | Dense matmul reads activations once, subject to backend tiling | Avoids host round trips, but lacks fused quantized-kernel traffic; expected to use more bandwidth and memory than native GEMV/GEMM |
| This fork reference/dequantize-to-dense fallback | Reference decode followed by dense matmul; decode and matmul are separate operations | Full dense matrix for the active layer; fallback must bound peak temporary use to the active layer rather than the whole model | Packed bytes are read by the decoder, then dense values are read by matmul; CPU fallback also moves the decoded result to the target device | Dense matmul reads activations once | Highest decode overhead and usually highest bandwidth and temporary-memory cost; useful for correctness and unsupported native combinations |
| Unquantized dense | Dense dot product only | No quantization temporary | Dense FP16/BF16/F32 weights read directly | Dense matmul reads activations once | No decode overhead, but more weight bytes than packed quantization; behavior depends on dense GEMM efficiency and memory bandwidth |

For current added formats, `native dequant-only` is not a claim of fused
quantized GEMV/GEMM/MoE. The implementation must be described by its actual
capability table, not by whether its decoder runs on CUDA or HIP.

## Quantization Coverage and Performance Status

This matrix describes repository implementation status, not universal model
compatibility. A model may still require compatible tensor names, shapes,
runtime, and backend coverage. Each operation column is a separate capability;
presence in one column does not imply presence in another.

Legend: ✅ available · ⚠️ conditional or partial · ❌ not available.

There is no Triton GEMV path in this plugin. `ggml_mul_mat_a8_triton` is a GEMM
kernel, so small-batch work either uses a native GEMV kernel or falls back to
dequantize-plus-dense. The Triton GEMV column is therefore ❌ everywhere and is
kept only to make that explicit.

The dispatch tables in `ops.py`, `triton/gemm/interface.py`, and
`triton/fused_moe/interface.py` remain the authority; this table summarises them.

| Status | Format family | Native GEMV | Native GEMM | Native MoE | Triton GEMV | Triton GEMM | Triton MoE | Limits and evidence |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **Verified baseline** | Standard formats: `Q4_0`-`Q8_0`, `Q2_K`-`Q6_K` | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | Upstream-compatible packed weights. `Q8_1` is the exception in this family: it has Triton GEMM and Triton MoE but no native GEMV/GEMM/MoE |
| **Verified baseline** | Upstream IQ formats: `IQ1_M`, `IQ1_S`, `IQ2_*`, `IQ3_*`, `IQ4_NL`, `IQ4_XS` | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | Native support is GEMV-only; larger batches and MoE use the Triton kernels |
| **Limited** | ROCmFPX: type 100 `Q4_0_ROCMFP4`, type 101 `Q4_0_ROCMFP4_FAST`, type 102 `Q6_0_ROCMFPX`, type 103 `Q8_0_ROCMFPX`, type 104 `Q3_0_ROCMFPX`, type 107 `Q2_0_ROCMFPX` | ⚠️ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ | Native GEMV kernels exist for all six types (`ggml_mul_mat_vec_rocmfpx`, plus `ggml_mul_mat_vec_rocmfp4_fast` for type 101) and require the compiled extension. Triton GEMM and Triton MoE are gated on `gfx115*`; outside that gate every ROCmFPX format falls back to dequantize-plus-dense. Types 100 and 101 also have native CUDA/HIP dequantization. No native C++/HIP GEMM or MoE is claimed. Validated on Radeon 8060S (`gfx1151`, ROCm/HIP 7.14.60850, Torch `2.12.0+rocm7.14.0`, Triton 3.7.1, `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`) and NVIDIA GB10 (`sm_121`, CUDA 13.0). Independent per-format ABI oracles (Codebook10/UE4M3 bias-8 for 100/101; signed int8 for 103; LSB-first bit-offset extraction for the 2/3/6-bit formats, including the `0x20 => -32` sign-magnitude case) covered FP32/FP16/BF16, rank-2/rank-3 activations, tail output rows, multi-block rows, and reserved `0x7f..0xff` per-half scale zeroing. MoE coverage includes layers that mix quant types across the expert tensors. This is kernel and dispatch validation, not full vLLM end-to-end/model-load validation. |
| **Limited** | ik K: `IQ*_K` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Reference dequantization plus materialized-dense path. Registered and wired; focused decoder coverage exists, but complete trusted-reference and end-to-end model validation remains incomplete |
| **Limited** | ik KS: `IQ*_KS` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Reference dequantization plus materialized-dense path. Same limitation as ik K; packed geometry and decode tests are not a native GEMV/GEMM claim |
| **Limited** | ik KSS/KL: `IQ4_KSS`, `IQ2_KL` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Reference dequantization plus materialized-dense path. Registered reference paths; no accelerated capability is advertised |
| **Reference fallback** | ik KT: `IQ1_KT`, `IQ2_KT`, `IQ3_KT`, `IQ4_KT` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Software dequantization/materialized-dense reference fallback with one FP32 row-prefix per row. KT remains reference-fallback-only; no acceleration is claimed |
| **Limited** | Trellis and related extended types: `TQ1_0`, `TQ2_0`, `Q2_0` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Registered reference dequantization and dense fallback. Type registration and focused CPU/reference checks do not establish acceleration |
| **Limited** | Type-41 `Q1_0` / `Q1_0_G128` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Accepted as aliases only with geometry `(128, 18)`. Incompatible type-41 geometry is rejected; native runtime validation remains pending |
| **Deferred** | `MXFP4`, `NVFP4` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not implemented. Do not advertise as supported |

The standard baseline is retained from the upstream plugin. Added formats are
implemented as Python/Triton/native CUDA/HIP ports only where the corresponding
dispatch table says so; unsupported combinations fail or use an explicit dense
fallback rather than silently selecting a wrong kernel.

## Multi-Token Prediction (MTP)

vLLM's native MTP reuses the target model's quantization. The MTP head is built
from ordinary `LinearBase` and `RoutedExperts` layers, so it loads through the
same `GGUFLinearMethod` and `GGUFMoEMethod` as the main model. There is no
separate MTP quantization path, and no MTP-specific kernel work is required for
a format that already has linear and MoE support.

What MTP does require is tensor-name resolution. GGUF encodes an MTP head as
`blk.{n}.nextn.*` tensors, and `gguf.TensorNameMap` only emits those for
architectures listed in its `MODEL_TENSORS` table. The published `gguf` package
lags llama.cpp master here, so `vllm_gguf_plugin/mtp_types.py` registers the
missing entries at import time, in the same way `ik_types.py` registers
ik_llama quantization types.

| Architecture support | Count | Notes |
|---|--:|---|
| Declared by installed `gguf` | 5 | `glm4`, `glm4moe`, `glm-dsa`, `exaone-moe`, `bailingmoe2` |
| Added by this plugin | 7 | `deepseek2`, `qwen3next`, `qwen35`, `qwen35moe`, `exaone4`, `mimo2`, `step35` |
| Needs a newer `gguf` | 5 | `cohere2moe`, `deepseek32`, `deepseek4`, `gemma4-assistant`, `hy_v3` |

The last group has no `MODEL_ARCH` member in the installed release, so no
runtime patch can reach it. `mtp_types.unsupported_mtp_architectures()` reports
that set, so the gap is visible rather than appearing as an unsupported model.

This covers weight-name resolution only. Running an MTP checkpoint end to end
also depends on vLLM's speculative-decoding stack and on the target model's
quantization support, neither of which this section claims.

## How to Interpret Comparisons

- Compare packed weight bytes only when the format ABI matches: same type ID,
  block size, byte layout, alignment, scale representation, and lookup tables.
- Equal packed bytes do not imply equal runtime behavior. Fused kernels,
  dequant-only kernels, and reference fallbacks perform different decode and
  memory work.
- End-to-end performance depends on token count, matrix shape, shard layout,
  dtype, backend, kernel availability, and vLLM scheduling. GEMV can be the
  relevant path at low token counts; GEMM or MoE paths can dominate at higher
  counts.
- Do not infer a speedup from lower bits per weight alone. Measure packed-weight
  traffic, decode work, dense temporary allocation, matmul efficiency, and peak
  memory together.
- No measured benchmark numbers should be invented. Current evidence includes
  direct native `_C_gguf` type-101 GEMV validation on local AMD Radeon 8060S
  (`gfx1151`) with ROCm 7.14/Torch 2.12, FP32/FP16/BF16 launches, and an
  independent ROCmFP4 half-scale UE4M3 bias-8 oracle, plus dense Triton GEMM
  correctness validation for ROCmFPX types 100, 102, 103, 104, and 107 against
  independent per-format ABI oracles, including confirmation that public
  `ops.ggml_mul_mat_a8` dispatch selects the dedicated kernel on that device.
  All of this is direct-op and dispatch validation, not full vLLM end-to-end
  model validation, and no throughput or memory-traffic numbers are claimed.

## Definition of Done for a Format

A format is complete only when all gates below pass. Enum registration alone is
not support.

1. **ABI and source parity**: identify authoritative source, type ID, block
   size, byte layout, alignment, signedness, scales, lookup tables, and shape
   constraints; compare against the source implementation.
2. **Storage and geometry**: validate row storage, shard behavior, tails,
   non-contiguous inputs where supported, and malformed metadata. Reject
   incompatible geometry instead of decoding with a guessed layout.
3. **Decode correctness**: compare random and boundary blocks with a trusted
   reference, including zero/small dimensions, tails, dtype variants, and
   non-contiguous cases where relevant.
4. **Dispatch truthfulness**: register only operations implemented by the
   capability tables. Keep native GEMV, native GEMM, native MoE, Triton, and
   CPU/reference fallback claims separate.
5. **Native versus fallback behavior**: verify backend selection, explicit
   failure or fallback, and no silent wrong-kernel selection. A native
   quantized linear path must not allocate a full dense temporary. Current
   native dequant-only paths explicitly emit decoded dense values and therefore
   remain dequant-only until a fused path meets this gate. Fallback paths must
   bound their temporary to the active layer and document peak use.
6. **Compute and bandwidth validation**: benchmark representative token counts,
   matrix shapes, dtypes, and shard layouts against the authoritative or
   equivalent implementation. Report decode work, weight reads, temporary
   memory, throughput, and peak memory; do not substitute estimates for
   measurements.
7. **Backend matrix**: validate CUDA, ROCm, and CPU/reference fallback
   behavior separately, recording hardware, toolkit, Torch, vLLM, and plugin
   versions. A skipped backend is not evidence of support.
8. **End-to-end model validation**: load and run representative models through
   vLLM, including tensor mapping, sharding, standard linear layers, and MoE or
   diffusion paths when claimed.
9. **Documentation**: update source lineage, format status, capability limits,
   fallback memory behavior, validation evidence, and usage guidance.

Native KT GEMV is future work. Native KT GEMM and native KT MoE are separate
work items and must not be implied by completing KT GEMV.

## Installation

The plugin is baked into homelab vLLM runtime images
(`vllm-strix-runtime` / `vllm-spark-runtime`), so GGUF serving works out of the
box there. To install standalone:

```bash
git clone https://github.com/vllm-project/vllm-gguf-plugin
cd vllm-gguf-plugin
git checkout homelabs-main
uv pip install -e . --torch-backend=auto
```

### gfx1151 devtools image

This repository includes a local development image for AMD Strix Halo (`gfx1151`).
It extends the local, non-redistributable `vllm-strix-devtools:local` image and
requires matching ROCm and Torch packages in that base image.

```bash
docker build -f docker/Dockerfile.gfx1151 -t vllm-gguf-plugin:gfx1151 .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  vllm-gguf-plugin:gfx1151 \
  /opt/venv/bin/python -m pytest \
    tests/test_rocmfp4_fast_triton_gemm.py \
    tests/test_type101_public_dense_gemm_dispatch.py \
    tests/test_rocmfp4_fast_fused_gemv.py \
    tests/test_dispatch_capabilities
```

Override the local base image with `--build-arg BASE_IMAGE=<matching-image>`.
The image builds the plugin in editable mode and uses the base image's
compatible ROCm, Torch, and vLLM.

## Usage

Serve a GGUF model; the plugin registers quantization types when vLLM imports
the entry point:

```bash
vllm serve <repo>/<model>-GGUF:Q4_0_ROCMFP4 --tokenizer <repo>/<model>
```

## For Contributors

`homelabs-main` is the consolidated branch. Per-format work lands on feature
branches (for example `rocmfpx-q4_0` and `ik-iqk`) and merges after review.
Each new quant type adds a block struct, dequant kernel, dispatch case, and
type-registration entry; see the ROCmFPX commits for the pattern. The upstream
development workflow is documented in [UPSTREAM.md](UPSTREAM.md).

## License

Apache 2.0, as upstream.
