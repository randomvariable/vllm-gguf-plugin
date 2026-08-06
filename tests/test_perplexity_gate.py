# SPDX-License-Identifier: Apache-2.0
"""End-to-end perplexity gate for quantized model execution.

Kernel parity tests compare a kernel against a reference oracle. They cannot
see a bug that lives *outside* the kernel -- in a converter transform, a weight
remap, or a layout assumption -- because both sides of that comparison are
already past it.

Two such bugs shipped in this repository while every kernel test passed:

* V heads were left in ggml's tiled order instead of being restored to the
  grouped order vLLM expects. A pure permutation, so magnitudes stayed healthy.
* Norm weights were applied with an off-by-one offset, because ggml folds the
  ``+1`` into the stored weight and vLLM adds it again at runtime. A constant
  offset, so again nothing looked anomalous.

Neither changed a magnitude, which is why residual-stream statistics, dequant
checks, GEMM parity, router behaviour, and embedding checks all came back
clean. Only a semantic measure separated them: perplexity moved 3.96e7 ->
3.21e5 -> 6.59 as each was fixed. Without that number the intermediate state
looked like gibberish either way, and the correct fix was nearly reverted.

This gate exists so that class of regression fails loudly. It follows
llama.cpp's ``ci/run.sh``, which quantizes a real model and rejects the build
if perplexity exceeds a fixed bound.

Running it
----------

Requires a GPU and a local GGUF checkpoint::

    VLLM_GGUF_PPL_MODEL=/path/to/model.gguf \\
    VLLM_GGUF_PPL_TOKENIZER=/path/to/tokenizer_dir \\
    pytest tests/test_perplexity_gate.py

The test skips when those are unset, so it stays out of the way of CPU runs.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

# vLLM's engine runs in a subprocess. The default fork start method inherits
# this process's CUDA context, which the runtime rejects on init; spawn gives
# the engine a clean one. Set before vLLM is imported.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Recorded on gfx1151 (AMD Radeon 8060S, ROCm 7.14) with
# Qwen3.6-14B-A3B-vibetuned-ROCmFPX-STRIX_LEAN, bfloat16, enforce_eager.
#
# This number is only meaningful for the exact CORPUS below, because perplexity
# is a mean over whichever tokens are scored. Editing the corpus invalidates it:
# the same model measures 6.59 over the first three entries (37 tokens) and 8.62
# over all six (78 tokens). Changing one without the other reads as a 31%
# regression that is not there.
#
# Update only alongside a deliberate, explained numerical change.
BASELINE_PERPLEXITY = 8.62

# Catastrophic ceiling, mirroring llama.cpp's ci/run.sh. A model this far off
# is broken rather than drifting; both historical bugs were orders of magnitude
# above it.
MAX_PERPLEXITY = 20.0

# Drift band around the recorded baseline. Generation is greedy over a fixed
# corpus with eager execution, so run-to-run variation should be nil; this
# leaves room for kernel scheduling and driver differences without admitting a
# real regression, which in every observed case moved perplexity by >1000x.
DRIFT_TOLERANCE = 0.05

# Fixed corpus. Deterministic and varied enough that a layout error cannot stay
# hidden in one domain: prose, code, factual statements, and dialogue each
# stress different parts of the vocabulary.
#
# BASELINE_PERPLEXITY is tied to this exact list. Re-record it if this changes.
CORPUS = (
    "The capital of France is Paris. The capital of Germany is Berlin.",
    "def add(a, b):\n    return a + b\n",
    "Water boils at 100 degrees Celsius at sea level.",
    "The mitochondrion is the powerhouse of the cell.",
    "Q: What is the largest planet? A: Jupiter is the largest planet.",
    "In 1969, humans first walked on the surface of the Moon.",
)


def _model_path() -> str | None:
    return os.environ.get("VLLM_GGUF_PPL_MODEL")


def _tokenizer_path() -> str | None:
    return os.environ.get("VLLM_GGUF_PPL_TOKENIZER")


requires_model = pytest.mark.skipif(
    not (_model_path() and _tokenizer_path() and torch.cuda.is_available()),
    reason=("requires a GPU plus VLLM_GGUF_PPL_MODEL and VLLM_GGUF_PPL_TOKENIZER"),
)


def _measure_perplexity() -> tuple[float, int]:
    """Return ``(perplexity, token_count)`` over :data:`CORPUS`.

    Perplexity is ``exp(-mean log P(token))`` across every scored token, which
    is what makes it sensitive to permutation and offset errors: the model still
    produces confident predictions, they are simply of the wrong tokens.
    """
    from vllm import LLM, SamplingParams

    import vllm_gguf_plugin  # noqa: F401  (registers the quantization method)

    llm = LLM(
        model=_model_path(),
        tokenizer=_tokenizer_path(),
        quantization="gguf",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
        # Eager execution keeps the measurement reproducible; graph capture can
        # reorder work in ways that perturb the last digits.
        enforce_eager=True,
    )

    params = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)
    outputs = llm.generate(list(CORPUS), params)

    total_logprob = 0.0
    total_tokens = 0
    for output in outputs:
        for entry in output.prompt_logprobs or []:
            if not entry:
                # The first token of each prompt has no predecessor to condition
                # on, so it carries no logprob and must not count toward the
                # mean.
                continue
            logprob = next(iter(entry.values()))
            total_logprob += getattr(logprob, "logprob", logprob)
            total_tokens += 1

    if total_tokens == 0:
        raise AssertionError("no tokens were scored; the corpus or API changed")

    return math.exp(-total_logprob / total_tokens), total_tokens


@requires_model
def test_quantized_model_perplexity_within_gate() -> None:
    """Quantized end-to-end execution must stay at its recorded quality.

    The absolute ceiling catches catastrophic breakage; the drift band catches
    the subtler regressions that motivated this gate.
    """
    perplexity, tokens = _measure_perplexity()

    assert tokens > 0, "no tokens scored"
    assert perplexity < MAX_PERPLEXITY, (
        f"perplexity {perplexity:.2f} exceeds the {MAX_PERPLEXITY} ceiling "
        f"over {tokens} tokens: the model is broken, not drifting"
    )

    drift = abs(perplexity - BASELINE_PERPLEXITY) / BASELINE_PERPLEXITY
    assert drift <= DRIFT_TOLERANCE, (
        f"perplexity {perplexity:.2f} differs from the recorded baseline "
        f"{BASELINE_PERPLEXITY} by {drift:.1%}, over the {DRIFT_TOLERANCE:.0%} "
        f"band. Either a regression landed or the baseline needs updating with "
        f"an explanation of the numerical change."
    )


@requires_model
def test_quantized_model_generates_coherent_text() -> None:
    """A factual continuation must be correct.

    Perplexity is averaged, so a localized failure can hide in it. This asserts
    on a specific completion the model is expected to get right, which is how
    the original gibberish was noticed in the first place.
    """
    from vllm import LLM, SamplingParams

    import vllm_gguf_plugin  # noqa: F401

    llm = LLM(
        model=_model_path(),
        tokenizer=_tokenizer_path(),
        quantization="gguf",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
        enforce_eager=True,
    )

    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=8, temperature=0),
    )
    text = outputs[0].outputs[0].text

    assert "Paris" in text, (
        f"expected 'Paris' in the continuation, got {text!r}. A permutation or "
        f"offset error yields confident but wrong tokens exactly like this."
    )
