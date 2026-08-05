# Agent Instructions for vLLM GGUF Plugin

These instructions apply to all AI-assisted contributions in this repository.
Keep changes small, explicit, and consistent with the existing Python, Triton,
CUDA, and GGUF conventions.

## Mandatory workflow

- Read `README.md`, `UPSTREAM.md`, `pyproject.toml`, and the owning source/tests
  before editing. Follow upstream behavior unless this fork's documented
  extension requires otherwise.
- Use TDD for every behavior change: write or extend a focused failing test,
  implement the smallest fix, then refactor only after the test passes.
- Reuse nearby fixtures and helpers. Test observable behavior through public
  APIs; avoid tests that only assert implementation details.
- Use Hypothesis property tests for meaningful quantization, block-layout,
  shape, dtype, boundary, or round-trip state spaces. Add focused examples for
  known edge cases and malformed input. Mark large-model/GPU tests `slow`.
- Update owning documentation in the same change. Keep README/UPSTREAM
  coverage, supported quantization tables, usage, limitations, and test
  expectations accurate; never defer docs to a follow-up.
- After each edit, inspect diagnostics for touched files and fix introduced
  errors before continuing. Do not claim completion with known diagnostics.

## Environment and checks

- Use `uv` for all Python environment and package operations. Do not use bare
  `python`, system `python3`, `pip`, or `pip install`.
- Create/use `.venv` and install editable development dependencies with the
  project extras, for example:

  ```bash
  uv venv
  uv pip install -e '.[dev]' --torch-backend=auto
  source .venv/bin/activate
  pre-commit install
  ```

- Run focused tests with `.venv/bin/python -m pytest ...`; run relevant
  pre-commit hooks, Ruff, and type/diagnostic checks before review. Run the
  smallest affected package/test scope first, then broader checks when kernel,
  registration, loader, or dispatch behavior changes.
- Do not require CUDA/ROCm for CPU-only unit tests. State GPU hardware,
  toolkit, vLLM, Torch, and plugin versions when GPU tests are skipped or fail.

## Adding or changing GGUF quantization

Treat each quantization format as an end-to-end contract. Checklist:

- [ ] Define or verify GGUF enum/type and block size/type size, including
      byte layout, alignment, packing, signedness, scales, lookup tables, and
      shape constraints.
- [ ] Implement dequantization/reference behavior and Triton or CUDA kernels
      as applicable; keep fake implementations and CPU/reference paths aligned.
- [ ] Add dispatch coverage in `ops.py` and quantization methods for dequant,
      GEMV/GEMM, fused-MoE, and diffusion paths where supported. Unsupported
      combinations must fail clearly, not silently select a wrong kernel.
- [ ] Register the type everywhere required: GGUF enum patching, supported
      type sets, quantization config, weight loading/materialization, and the
      vLLM plugin entry point.
- [ ] Test block decoding against a trusted reference (llama.cpp,
      ik_llama.cpp, ROCmFPX, or a checked-in equivalent), including random
      blocks, tails, zero/small dimensions, non-contiguous inputs, dtype
      variants, and invalid metadata.
- [ ] Test numerical tolerances, output shapes, strides, device placement,
      empty inputs, and mixed/sharded weights. For GPU kernels, compare against
      a high-precision/reference implementation across representative sizes
      and architectures; check synchronization, bounds, alignment, and
      deterministic behavior.
- [ ] Verify dispatch on every supported backend and verify fallback behavior
      when an extension is unavailable or `VLLM_GGUF_USE_CUDA=0`.
- [ ] Update supported-format docs and add a regression test for the new path.

Never call a format supported merely because its enum exists. A format is
supported only when block decoding, dispatch, registration, loading, and tests
agree.

## Review rules

- Review every changed line as a human would: correctness first, then API
  compatibility, numerical accuracy, performance, portability, and security.
- Look specifically for wrong block geometry, byte-order/packing mistakes,
  integer overflow, dtype/device mismatches, silent fallback, missing fake
  kernels, stale registration sets, and loader/shard ordering bugs.
- Require focused tests and docs for behavior changes. Do not add one-off
  benchmarks to `tests/`; put performance work in an appropriate benchmark
  location while keeping correctness tests in the test suite.
- Do not hide failures with broad exception handling, skip GPU correctness
  checks without recording why, or change upstream-compatible behavior without
  documenting the fork-specific reason.
- Before submission, report commands and results, hardware/backend coverage,
  skipped checks with reasons, and any known limitations. A human must review
  and own the final change; AI assistance must be disclosed in the PR.
