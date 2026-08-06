# SPDX-License-Identifier: Apache-2.0
"""Tests for the derived-tolerance and canary helpers.

A test helper that silently accepts wrong results is worse than no helper, so
these check both directions: real error must be caught, and legitimate
rounding must pass.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.numerics import (
    SentinelTensor,
    assert_close,
    assert_finite_range,
    dot_condition,
    epsilon,
    max_nmse,
    max_relative_error,
    max_relative_error_chained,
    nmse,
    sentinel_zeros,
    uses_tf32,
)


class TestEpsilon:
    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (torch.float32, 2.0**-24),
            (torch.float16, 2.0**-11),
            (torch.bfloat16, 2.0**-8),
            (torch.float64, 2.0**-53),
        ],
    )
    def test_matches_ieee_mantissa_width(
        self, dtype: torch.dtype, expected: float
    ) -> None:
        assert epsilon(dtype) == expected

    def test_epsilon_matches_torch_finfo(self) -> None:
        """Cross-check against torch's own constants rather than trusting the table."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            # finfo.eps is the gap above 1.0, which is 2^-(p-1) for p mantissa
            # bits including the implicit one; epsilon() reports 2^-p.
            assert epsilon(dtype) == pytest.approx(torch.finfo(dtype).eps / 2)

    def test_tf32_is_less_precise_than_float16(self) -> None:
        """The inversion that makes NVIDIA float32 results look wrong."""
        assert epsilon(torch.float32, tf32=True) > epsilon(torch.float32)
        assert epsilon(torch.float32, tf32=True) == epsilon(torch.float16)

    def test_tf32_only_affects_float32(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            assert epsilon(dtype, tf32=True) == epsilon(dtype)

    def test_unknown_dtype_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no mantissa width"):
            epsilon(torch.int32)


class TestMaxRelativeError:
    def test_ordering_follows_precision(self) -> None:
        k = 4096
        assert (
            max_relative_error(torch.float32, k)
            < max_relative_error(torch.float16, k)
            < max_relative_error(torch.bfloat16, k)
        )

    def test_grows_with_reduction_length(self) -> None:
        """Reassociation error follows a random walk, so it grows as sqrt(k).

        Checked in float32, where the operand and accumulator have equal
        precision so the sqrt(k) term is visible. With a reduced-precision
        operand and a float32 accumulator the constant operand term dominates
        instead -- covered separately below.
        """
        small = max_relative_error(torch.float32, 256)
        large = max_relative_error(torch.float32, 4096)
        assert large > small
        # 16x the terms is 4x the walk; the constant operand term dilutes this
        # slightly, so the observed ratio sits just under 4.
        assert 3.0 < large / small < 4.0

    def test_reduced_precision_operands_dominate_the_bound(self) -> None:
        """With a float32 accumulator, operand rounding sets the float16 bound.

        This is why a float16 kernel shows near-identical error at k=256 and
        k=4096: the accumulator's contribution is ~3 orders of magnitude smaller.
        """
        small = max_relative_error(torch.float16, 256)
        large = max_relative_error(torch.float16, 4096)
        assert large / small < 1.1

    def test_tf32_widens_the_bound(self) -> None:
        k = 1024
        assert max_relative_error(torch.float32, k, tf32=True) > max_relative_error(
            torch.float32, k
        )

    def test_rejects_non_positive_k(self) -> None:
        for k in (0, -1):
            with pytest.raises(ValueError, match="k must be positive"):
                max_relative_error(torch.float16, k)

    def test_nmse_bound_is_the_squared_relative_bound(self) -> None:
        k = 512
        assert max_nmse(torch.float16, k) == pytest.approx(
            max_relative_error(torch.float16, k) ** 2
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("k", [64, 256, 4096])
    def test_bound_holds_for_simulated_reduction(
        self, dtype: torch.dtype, k: int
    ) -> None:
        """The derived bound must actually cover real reduction error.

        Emulates the kernel/reference split: the reference accumulates in one
        pass, the kernel over K-tiles, both rounding to the operand dtype.
        """
        torch.manual_seed(k)
        worst = 0.0
        for _ in range(16):
            a = torch.randn(k, dtype=torch.float64)
            b = torch.randn(k, dtype=torch.float64)
            a_q = a.to(dtype).to(torch.float64)
            b_q = b.to(dtype).to(torch.float64)

            reference = (a_q * b_q).sum()
            tile = 32
            tiled = sum(
                (a_q[i : i + tile] * b_q[i : i + tile])
                .sum()
                .to(dtype)
                .to(torch.float64)
                for i in range(0, k, tile)
            )
            scale = max(abs(reference.item()), math.sqrt(k))
            worst = max(worst, abs(tiled.item() - reference.item()) / scale)

        assert worst < max_relative_error(dtype, k)


class TestNMSE:
    def test_identical_tensors_score_zero(self) -> None:
        x = torch.randn(64)
        assert nmse(x, x) == 0.0

    def test_is_scale_invariant(self) -> None:
        """The property that makes NMSE preferable to a raw max-abs check.

        Scaling the inputs is not bit-exact in float32: each scaled value carries
        ~eps of rounding, but ``actual - expected`` is much smaller than either
        operand, so cancellation amplifies that relative error by roughly
        ``|expected| / |actual - expected|``. The tolerance is derived from that
        ratio rather than picked, and it stays fixed while the magnitude sweeps
        nine orders of magnitude.
        """
        torch.manual_seed(0)
        expected = torch.randn(256)
        actual = expected + 0.01 * torch.randn(256)

        amplification = (
            expected.abs().mean() / (actual - expected).abs().mean()
        ).item()
        # NMSE is quadratic in the difference, hence the factor of two.
        tolerance = 2 * amplification * epsilon(torch.float32)

        base = nmse(actual, expected)
        for factor in (1e-3, 1e3, 1e6):
            assert nmse(actual * factor, expected * factor) == pytest.approx(
                base, rel=tolerance
            )

    def test_zero_reference_falls_back_to_absolute_error(self) -> None:
        expected = torch.zeros(32)
        actual = torch.full((32,), 0.5)
        assert nmse(actual, expected) == pytest.approx(0.25)

    def test_averages_over_elements(self) -> None:
        """One blown element in a large tensor is diluted -- hence the peak check."""
        expected = torch.ones(1024)
        actual = expected.clone()
        actual[0] = 100.0
        assert nmse(actual, expected) < 10.0


class TestAssertClose:
    def _dot(self, k: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(k)
        a = torch.randn(8, k, dtype=dtype)
        b = torch.randn(k, 8, dtype=dtype)
        expected = (a.to(torch.float32) @ b.to(torch.float32)).to(dtype)
        return expected, expected.clone()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_exact_match_passes(self, dtype: torch.dtype) -> None:
        expected, actual = self._dot(256, dtype)
        assert_close(actual, expected, dtype, 256)

    def test_catches_scale_error(self) -> None:
        """A wrong scale byte is the canonical decode bug -- must never pass."""
        expected, actual = self._dot(256, torch.float16)
        with pytest.raises(AssertionError, match="peak normalized error|NMSE"):
            assert_close(actual * 2.0, expected, torch.float16, 256)

    def test_catches_half_scale_error(self) -> None:
        """The 0.5x UE4M3 bias confusion seen on real ROCmFPX oracles."""
        expected, actual = self._dot(256, torch.float16)
        with pytest.raises(AssertionError):
            assert_close(actual * 0.5, expected, torch.float16, 256)

    def test_catches_permutation(self) -> None:
        """Reordered output is magnitude-preserving -- NMSE catches what norms miss."""
        expected, actual = self._dot(256, torch.float32)
        with pytest.raises(AssertionError):
            assert_close(actual.flip(-1), expected, torch.float32, 256)

    def test_catches_single_blown_element(self) -> None:
        expected, actual = self._dot(256, torch.float32)
        actual[0, 0] = expected.abs().max() * 50
        with pytest.raises(AssertionError, match="peak normalized error"):
            assert_close(actual, expected, torch.float32, 256)

    def test_catches_nan(self) -> None:
        expected, actual = self._dot(64, torch.float32)
        actual[0, 0] = float("nan")
        with pytest.raises(AssertionError, match="NaN"):
            assert_close(actual, expected, torch.float32, 64)

    def test_catches_unmatched_infinity(self) -> None:
        expected, actual = self._dot(64, torch.float32)
        actual[0, 0] = float("inf")
        with pytest.raises(AssertionError, match="infinity mismatch"):
            assert_close(actual, expected, torch.float32, 64)

    def test_catches_shape_mismatch(self) -> None:
        expected, _ = self._dot(64, torch.float32)
        with pytest.raises(AssertionError, match="shape"):
            assert_close(expected[:4], expected, torch.float32, 64)

    def test_accepts_legitimate_rounding(self) -> None:
        """Genuine reduced-precision rounding must not trip the bound."""
        k = 1024
        torch.manual_seed(0)
        a = torch.randn(8, k, dtype=torch.float64)
        b = torch.randn(k, 8, dtype=torch.float64)
        expected = a @ b
        actual = (a.to(torch.bfloat16).to(torch.float64)) @ (
            b.to(torch.bfloat16).to(torch.float64)
        )
        assert_close(actual, expected, torch.bfloat16, k)

    def test_label_appears_in_failure(self) -> None:
        expected, actual = self._dot(64, torch.float32)
        with pytest.raises(AssertionError, match="q4_0 rank-3"):
            assert_close(actual * 3.0, expected, torch.float32, 64, label="q4_0 rank-3")


class TestBoundSensitivity:
    """Mutation tests: the bounds must reject real bug classes, not just pass.

    A tolerance that accepts everything is worthless. Each mutation here is a
    defect class actually hit in this codebase.
    """

    K = (64, 64)

    def _reference(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(8, 64, dtype=torch.float32) * 10

    @pytest.mark.parametrize(
        "name,mutate",
        [
            ("half_scale", lambda t: t * 0.5),
            ("double_scale", lambda t: t * 2.0),
            ("nibble_order_swapped", lambda t: t.flip(-1)),
            ("sign_flip", lambda t: -t),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_gross_decode_errors_rejected_in_every_dtype(
        self, name: str, mutate, dtype: torch.dtype
    ) -> None:
        """Structural decode failures are O(1) and must fail at any precision."""
        expected = self._reference()
        actual = mutate(expected.to(dtype).to(torch.float32))
        with pytest.raises(AssertionError):
            assert_close(actual, expected, dtype, self.K, label=name)

    @pytest.mark.parametrize("drift", [1.001, 1.0001, 1.00001])
    def test_float32_detects_subtle_drift(self, drift: float) -> None:
        """float32 is the sensitivity test -- it resolves sub-0.01% error."""
        expected = self._reference()
        actual = expected * drift
        with pytest.raises(AssertionError):
            assert_close(actual, expected, torch.float32, self.K)

    def test_clean_result_passes_in_every_dtype(self) -> None:
        expected = self._reference()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            assert_close(expected.to(dtype).to(torch.float32), expected, dtype, self.K)

    def test_float32_bound_is_far_tighter_than_reduced_precision(self) -> None:
        """Guards the claim in the module docstring."""
        f32 = max_relative_error_chained(torch.float32, self.K)
        f16 = max_relative_error_chained(torch.float16, self.K)
        assert f16 / f32 > 100

    def test_subtle_drift_is_unresolvable_in_float16(self) -> None:
        """Documents why float16 cannot be held to a tighter bound.

        A 0.1% systematic error sits at roughly 2x float16's machine epsilon.
        Separating it from honest rounding would need a bound below ~2 eps,
        which real accumulation over a two-stage pipeline already exceeds --
        measured at 9.3e-4 on gfx1151, about 1.9 eps. So a bound tight enough
        to reject the bug also rejects correct results.
        """
        drift = 1e-3
        eps16 = epsilon(torch.float16)
        assert drift / eps16 < 3.0

        # The observed floor for a correct float16 MoE run on gfx1151, which a
        # drift-detecting bound would have to sit below.
        observed_floor = 9.31e-4
        assert observed_floor / eps16 > 1.5
        assert drift < 2 * observed_floor


class TestCancellation:
    """Rounding error scales with term magnitude, not output magnitude.

    Ignoring this flagged a numerically perfect kernel: measured error was
    5.7e-8 relative to the terms -- below float32 eps -- but 9.1e-6 relative to
    the heavily cancelled output.
    """

    def test_orthogonal_case_is_well_conditioned(self) -> None:
        torch.manual_seed(0)
        a = torch.randn(4, 128)
        b = torch.randn(128, 4)
        assert dot_condition(a, b) < 20.0

    def test_cancelling_case_is_ill_conditioned(self) -> None:
        """Large terms summing to nearly zero."""
        a = torch.tensor([[1e6, 1e6]])
        b = torch.tensor([[1.0], [-1.0 + 1e-6]])
        assert dot_condition(a, b) > 1e5

    def test_condition_is_never_below_one(self) -> None:
        a = torch.ones(2, 4)
        b = torch.ones(4, 2)
        assert dot_condition(a, b) >= 1.0

    def test_zero_terms_are_not_ill_conditioned(self) -> None:
        """An all-zero product has no scale; it is unconditioned, not infinite."""
        a = torch.zeros(2, 4)
        b = torch.zeros(4, 2)
        assert dot_condition(a, b) == 1.0

    def test_condition_widens_the_bound_proportionally(self) -> None:
        base = max_relative_error(torch.float32, 256)
        scaled = max_relative_error(torch.float32, 256, condition=100.0)
        assert scaled == pytest.approx(base * 100.0)

    def test_condition_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="condition must be at least"):
            max_relative_error(torch.float32, 256, condition=0.5)

    def test_cancellation_does_not_mask_a_real_bug(self) -> None:
        """Widening for cancellation must not let a decode error through.

        The condition number reached by the type-101 max-scale case was ~8e3;
        even at that width a half-scale error is still rejected.
        """
        torch.manual_seed(0)
        expected = torch.randn(8, 64) * 10
        with pytest.raises(AssertionError):
            assert_close(expected * 0.5, expected, torch.float32, 96, condition=8e3)


class TestChainedBound:
    def test_chain_exceeds_any_single_stage(self) -> None:
        stages = (64, 128)
        chained = max_relative_error_chained(torch.float16, stages)
        assert chained > max(max_relative_error(torch.float16, k) for k in stages)

    def test_single_stage_chain_matches_scalar_form(self) -> None:
        assert max_relative_error_chained(torch.float32, (256,)) == max_relative_error(
            torch.float32, 256
        )

    def test_assert_close_accepts_scalar_or_tuple_k(self) -> None:
        torch.manual_seed(0)
        expected = torch.randn(8, 32)
        actual = expected.clone()
        assert_close(actual, expected, torch.float32, 256)
        assert_close(actual, expected, torch.float32, (256, 128))


class TestFiniteRange:
    def test_in_range_values_pass(self) -> None:
        assert_finite_range(torch.full((8,), 1000.0), torch.float16)

    def test_float16_overflow_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="exceeds"):
            assert_finite_range(torch.full((8,), 1e9), torch.float16)

    def test_bfloat16_tolerates_float16_overflow(self) -> None:
        """bfloat16 keeps float32's exponent, so it has far more range."""
        assert_finite_range(torch.full((8,), 1e9), torch.bfloat16)

    def test_message_recommends_scaling_not_loosening(self) -> None:
        with pytest.raises(AssertionError, match="scale the test data down"):
            assert_finite_range(torch.full((8,), 1e9), torch.float16)


class TestUsesTF32:
    def test_ieee_override_disables_tf32(self, monkeypatch) -> None:
        monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
        assert uses_tf32() is False

    def test_false_without_cuda(self, monkeypatch) -> None:
        monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert uses_tf32() is False

    def test_false_on_rocm(self, monkeypatch) -> None:
        """AMD has no TF32 path, so float32 there is true IEEE float32."""
        monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.version, "hip", "7.14.0", raising=False)
        assert uses_tf32() is False


class TestSentinelTensor:
    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.uint8]
    )
    def test_clean_payload_write_passes(self, dtype: torch.dtype) -> None:
        guarded = SentinelTensor((4, 8), dtype, "cpu")
        guarded.tensor.fill_(1)
        guarded.check()

    def test_payload_has_requested_shape_and_dtype(self) -> None:
        guarded = SentinelTensor((3, 5, 7), torch.float16, "cpu")
        assert guarded.tensor.shape == (3, 5, 7)
        assert guarded.tensor.dtype is torch.float16

    def test_detects_write_before_payload(self) -> None:
        guarded = SentinelTensor((4, 8), torch.float32, "cpu", pad=64)
        flat = guarded._raw
        flat[63] = 1.0
        with pytest.raises(AssertionError, match="before the tensor"):
            guarded.check()

    def test_detects_write_after_payload(self) -> None:
        guarded = SentinelTensor((4, 8), torch.float32, "cpu", pad=64)
        flat = guarded._raw
        flat[64 + 32] = 1.0
        with pytest.raises(AssertionError, match="after"):
            guarded.check()

    def test_payload_is_contiguous_with_canaries(self) -> None:
        """Canaries only detect an overrun if they are genuinely adjacent."""
        guarded = SentinelTensor((4, 8), torch.float32, "cpu", pad=16)
        payload_start = guarded.tensor.data_ptr()
        raw_start = guarded._raw.data_ptr()
        assert payload_start == raw_start + 16 * guarded._raw.element_size()

    def test_label_appears_in_failure(self) -> None:
        guarded = SentinelTensor((4,), torch.float32, "cpu", pad=8)
        guarded._raw[0] = 1.0
        with pytest.raises(AssertionError, match="q8_0 output"):
            guarded.check(label="q8_0 output")

    def test_sentinel_zeros_starts_zeroed(self) -> None:
        guarded = sentinel_zeros((4, 8), torch.float32, "cpu")
        assert torch.equal(guarded.tensor, torch.zeros(4, 8))
        guarded.check()
