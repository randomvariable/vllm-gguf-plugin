# SPDX-License-Identifier: Apache-2.0
"""Derived error bounds and out-of-bounds canaries for kernel parity tests.

Two problems this solves.

**Hand-picked tolerances.** Choosing a bound after watching a test fail fits the
bound to the result: a decode bug smaller than the fudge factor passes silently.
Every bound here is derived from the floating-point format actually in use, so
it can be checked against the model rather than against a remembered number.

**No out-of-bounds detection.** Kernels that compute their load offsets -- the
bit-offset and plane-major decoders especially -- can read or write past a
tensor and still produce plausible numbers. Sentinel canaries make that visible,
following the same approach as llama.cpp's ``test-backend-ops``.

Which dtype catches what
------------------------

float32 is the sensitivity test. Its bound is roughly three orders of magnitude
tighter than float16's, so it detects systematic drift far below what a
reduced-precision run can resolve.

float16 and bfloat16 catch *structural* failures: compile errors, dispatch
mistakes, dtype mismatches in ``tl.dot``, and gross decode errors such as a
half-scale or a swapped nibble order. They cannot catch subtle drift, and no
choice of tolerance would let them. Measured on gfx1151, legitimate float16
rounding through a two-stage MoE reaches 9.3e-4 relative while a 0.1%
systematic error produces 1.3e-3 -- 1.35x apart, which is not separable. A
bound tight enough to reject the bug would reject correct results.

So reduced-precision coverage is necessary (the type-103 fp16 crash was invisible
to a float32-only suite) but is not where numerical sensitivity comes from.
"""

from __future__ import annotations

import math
import os

import torch

# Mantissa bits including the implicit leading 1. Machine epsilon is 2^-p.
_MANTISSA_BITS = {
    torch.float64: 53,
    torch.float32: 24,
    torch.bfloat16: 8,
    torch.float16: 11,
}

# NVIDIA lowers float32 tl.dot to TF32 tensor cores unless TRITON_F32_DEFAULT=ieee
# is set. TF32 keeps float32's exponent but truncates to a 10-bit stored mantissa,
# so a float32 kernel is *less* precise than float16 there.
TF32_MANTISSA_BITS = 11

# Covers the gap between the random-data model below and a real distribution:
# operand magnitudes are not uniform, the peak-magnitude output is not
# necessarily the peak-error output, and reduction trees vary by backend.
#
# Sized against measurement, not guessed. Sweeping the ROCmFPX MoE parity
# harness on gfx1151 across six formats, three dtypes and four seeds gives
# observed/bound headroom of 8.5x (float32), 4.2x (float16) and 2.9x
# (bfloat16). Halving this would leave bfloat16 at 1.45x, close enough to the
# observed worst case to be flaky; doubling it would weaken every bound for no
# measured benefit.
_SAFETY = 4.0


def epsilon(dtype: torch.dtype, *, tf32: bool = False) -> float:
    """Machine epsilon (2^-p) for a floating-point dtype."""
    if tf32 and dtype is torch.float32:
        return 2.0**-TF32_MANTISSA_BITS
    try:
        return 2.0 ** -_MANTISSA_BITS[dtype]
    except KeyError:
        raise ValueError(f"no mantissa width known for {dtype}") from None


def max_relative_error(
    operand_dtype: torch.dtype,
    k: int,
    *,
    accumulate_dtype: torch.dtype = torch.float32,
    tf32: bool = False,
) -> float:
    """Error bound for a length-``k`` dot product against a dense reference.

    Both sides use the same quantized weights, so quantization error cancels and
    only arithmetic error remains. Two terms contribute:

    *Operand rounding.* Each operand carries at most ``eps/2``, so each product
    carries ``eps``. For random data both the accumulated error and the signal
    grow as ``sqrt(k)``, so after normalizing by output magnitude this term is
    independent of ``k``.

    *Reassociation.* The reference sums per output while the kernel sums over
    K-tiles. Both round at every partial sum in the accumulator's precision, and
    the difference between two orderings follows a random walk: ``eps_acc *
    sqrt(k)``. This dominates when operands are exact, which is why true float32
    is *less* accurate than the ``eps`` term alone suggests.

    Pass ``tf32=True`` on NVIDIA unless ``TRITON_F32_DEFAULT=ieee`` is set.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    operand = epsilon(operand_dtype, tf32=tf32)
    accumulate = epsilon(accumulate_dtype, tf32=tf32)
    return _SAFETY * (operand + accumulate * math.sqrt(k))


def uses_tf32(device: torch.device | str = "cuda") -> bool:
    """Whether float32 ``tl.dot`` lowers to TF32 tensor cores on this device.

    True on NVIDIA unless ``TRITON_F32_DEFAULT=ieee`` is set. AMD has no TF32
    path, so float32 there is true IEEE float32.
    """
    if os.environ.get("TRITON_F32_DEFAULT") == "ieee":
        return False
    if not torch.cuda.is_available():
        return False
    # torch.version.hip is set for ROCm builds, where "cuda" means HIP.
    return getattr(torch.version, "hip", None) is None


def max_relative_error_chained(
    operand_dtype: torch.dtype,
    reduction_lengths: tuple[int, ...],
    *,
    accumulate_dtype: torch.dtype = torch.float32,
    tf32: bool = False,
) -> float:
    """Error bound for chained matmuls, such as an MoE gate/up then down pass.

    Relative errors add through the chain: each stage's input already carries the
    upstream error, and the activation between them (SiLU, GELU) has a derivative
    bounded near 1, so it neither amplifies nor cancels it to first order.
    """
    return sum(
        max_relative_error(
            operand_dtype, k, accumulate_dtype=accumulate_dtype, tf32=tf32
        )
        for k in reduction_lengths
    )


def nmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Normalized mean squared error, ``mse(a, b) / mse(b, 0)``.

    Self-normalizing, so the bound does not drift with tensor magnitude. This is
    the aggregate companion to :func:`max_relative_error`: it catches a broadly
    wrong result that a max-error check might attribute to one unlucky element,
    while the max check catches a single blown element that an average hides.
    """
    a = actual.to(torch.float64)
    b = expected.to(torch.float64)
    denom = (b * b).mean().item()
    if denom == 0.0:
        # An all-zero reference has no scale to normalize against, so fall back
        # to absolute error rather than reporting a meaningless ratio.
        return ((a - b) ** 2).mean().item()
    return ((a - b) ** 2).mean().item() / denom


def max_nmse(
    operand_dtype: torch.dtype,
    k: int,
    *,
    accumulate_dtype: torch.dtype = torch.float32,
    tf32: bool = False,
) -> float:
    """NMSE bound for the same dot product :func:`max_relative_error` covers.

    NMSE is squared error over squared signal, so it bounds the square of the
    relative error.
    """
    return (
        max_relative_error(
            operand_dtype, k, accumulate_dtype=accumulate_dtype, tf32=tf32
        )
        ** 2
    )


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    operand_dtype: torch.dtype,
    k: int | tuple[int, ...],
    *,
    accumulate_dtype: torch.dtype = torch.float32,
    tf32: bool = False,
    label: str = "",
) -> None:
    """Assert kernel output matches a dense reference within derived bounds.

    Checks NaN and infinity first (including sign, since a flipped infinity is a
    real defect), then both the aggregate NMSE and the peak normalized error.

    ``k`` is the reduction length, or a tuple of them for a chained pipeline.
    """
    prefix = f"{label}: " if label else ""
    if actual.shape != expected.shape:
        raise AssertionError(f"{prefix}shape {actual.shape} != {expected.shape}")

    a = actual.to(torch.float64)
    b = expected.to(torch.float64)

    if torch.isnan(a).any() and not torch.isnan(b).any():
        index = int(torch.isnan(a).flatten().nonzero()[0].item())
        raise AssertionError(f"{prefix}NaN at flat index {index}")

    # An infinity must be matched by an infinity of the same sign; a finite
    # result where the reference overflowed is just as wrong as the reverse.
    a_inf, b_inf = torch.isinf(a), torch.isinf(b)
    if not torch.equal(a_inf, b_inf):
        index = int((a_inf != b_inf).flatten().nonzero()[0].item())
        raise AssertionError(f"{prefix}infinity mismatch at flat index {index}")
    if a_inf.any() and not torch.equal(
        torch.signbit(a[a_inf]), torch.signbit(b[b_inf])
    ):
        raise AssertionError(f"{prefix}infinity sign mismatch")

    lengths = (k,) if isinstance(k, int) else tuple(k)
    bound = max_relative_error_chained(
        operand_dtype, lengths, accumulate_dtype=accumulate_dtype, tf32=tf32
    )
    finite = ~a_inf
    scale = b[finite].abs().max().item() if finite.any() else 0.0
    if scale > 0.0:
        peak = (a[finite] - b[finite]).abs().max().item() / scale
        if peak > bound:
            raise AssertionError(
                f"{prefix}peak normalized error {peak:.3e} exceeds {bound:.3e} "
                f"(operand={operand_dtype}, k={k}, tf32={tf32})"
            )

    observed = nmse(actual, expected)
    limit = bound**2
    if observed > limit:
        raise AssertionError(
            f"{prefix}NMSE {observed:.3e} exceeds {limit:.3e} "
            f"(operand={operand_dtype}, k={k}, tf32={tf32})"
        )


def assert_finite_range(
    tensor: torch.Tensor, dtype: torch.dtype, *, label: str = ""
) -> None:
    """Assert values fit ``dtype``'s finite range.

    Synthetic quantized test data uses random codes with live scales, which give
    magnitudes far above a trained model's; a chained pipeline can then overflow
    float16's 65504 ceiling. That is a property of the test data, not a kernel
    defect, so it should be caught explicitly rather than surfacing as a
    mystery NaN inside a parity comparison.
    """
    prefix = f"{label}: " if label else ""
    peak = tensor.abs().max().item()
    limit = torch.finfo(dtype).max
    if peak > limit:
        raise AssertionError(
            f"{prefix}reference peak {peak:.3e} exceeds {dtype} max {limit:.3e}; "
            f"scale the test data down rather than widening the tolerance"
        )


# Distinct high-entropy byte patterns, so a canary hit reports which side was
# overrun and a partial overwrite is still visible. Applied at byte level, which
# works uniformly for every payload dtype -- a float16 buffer cannot represent an
# arbitrary 32-bit pattern, and a float dtype could quietly normalize it.
_CANARY_BEFORE = 0x5A
_CANARY_AFTER = 0xA5


class SentinelTensor:
    """A tensor flanked by canary regions that detect out-of-bounds access.

    Allocates one buffer holding ``[canary | payload | canary]`` and hands back a
    view of the payload. If a kernel writes outside its bounds, the canary
    changes and :meth:`check` reports it. This is the mechanism llama.cpp's
    ``test-backend-ops`` uses, and it catches offset bugs that produce plausible
    numbers -- the failure mode a value comparison alone cannot see.

    Contiguity matters: the canaries only detect an overrun if they are
    genuinely adjacent in memory, which is why the payload is a slice of one
    allocation rather than a separate tensor.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str,
        *,
        pad: int = 1024,
    ) -> None:
        self._shape = tuple(shape)
        self._pad = pad
        count = 1
        for dim in self._shape:
            count *= dim

        self._raw = torch.empty(count + 2 * pad, dtype=dtype, device=device)
        # Canaries are written and checked through a byte view, so the pattern
        # survives regardless of payload dtype.
        self._bytes = self._raw.view(torch.uint8)
        self._pad_bytes = pad * self._raw.element_size()

        self._fill_canaries()
        self._payload = self._raw[pad : pad + count].view(self._shape)

    def _fill_canaries(self) -> None:
        self._bytes[: self._pad_bytes] = _CANARY_BEFORE
        self._bytes[-self._pad_bytes :] = _CANARY_AFTER

    @property
    def tensor(self) -> torch.Tensor:
        """The payload view. Pass this to the kernel under test."""
        return self._payload

    def check(self, label: str = "") -> None:
        """Raise if either canary region was modified."""
        prefix = f"{label}: " if label else ""
        head = self._bytes[: self._pad_bytes]
        tail = self._bytes[-self._pad_bytes :]
        head_bad = int((head != _CANARY_BEFORE).sum().item())
        tail_bad = int((tail != _CANARY_AFTER).sum().item())

        if head_bad or tail_bad:
            raise AssertionError(
                f"{prefix}out-of-bounds write: {head_bad} byte(s) before the "
                f"tensor, {tail_bad} after (shape={self._shape})"
            )


def sentinel_zeros(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
    *,
    pad: int = 1024,
) -> SentinelTensor:
    """A zero-filled :class:`SentinelTensor`, for kernel output buffers."""
    guarded = SentinelTensor(shape, dtype, device, pad=pad)
    guarded.tensor.zero_()
    return guarded
