#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the kernel test suites across every available GPU backend.

Why this exists
---------------

A kernel that passes on one backend can fail on another, and the failures are
not exotic. Both of these shipped in this repository:

* ``tl.dot`` rejects mismatched operand dtypes, so a kernel leaving decoded
  weights in float32 while activations are bfloat16 fails to compile -- but only
  once a reduced-precision case actually runs.
* float32 ``tl.dot`` lowers to TF32 tensor cores on NVIDIA (10-bit mantissa) and
  to full precision on AMD, so the *same* kernel is bit-exact on gfx1151 and
  shows ~5e-4 relative error on sm_121. Reading that as a decode bug wastes a
  day; not reading it at all hides a real one behind a loosened tolerance.

llama.cpp covers this with 15+ backends in CI. We have two, so the matrix is
small -- but running it by hand means it gets skipped, and a per-backend
regression then lands unnoticed.

Backends
--------

``gfx1151``
    AMD Radeon 8060S (Strix Halo), via a local Docker container.

``sm_121``
    NVIDIA GB10, via a Kubernetes pod on a DGX node.

Both are detected before use and reported as unavailable rather than failing,
so a partial run still produces a usable matrix.

Memory
------

Suites run one at a time. These are unified-memory machines -- the gfx1151 host
shares 124 GB between CPU and GPU, and the DGX nodes have been observed over
100% memory commitment -- so concurrent pytest processes each holding a CUDA
context and Triton's compilation cache risk evicting the workloads that share
the node. Sequential execution costs wall time and nothing else.

The suites listed here are kernel-level and allocate small tensors. The
end-to-end perplexity gate is deliberately excluded: it loads a 14B model and
belongs in a separate, deliberate run.

Usage
-----

::

    scripts/backend_matrix.py                 # every available backend
    scripts/backend_matrix.py --backend gfx1151
    scripts/backend_matrix.py --list          # show configuration and exit
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence

# Suites that execute quantization kernels. Dispatch-contract suites that only
# assert on registration tables are excluded: they cannot vary by backend, so
# running them per backend adds time without adding information.
KERNEL_SUITES = (
    "tests/test_rocmfpx_fused_moe_parity.py",
    "tests/test_rocmfp4_fast_triton_gemm.py",
    "tests/test_rocmfp4_fast_fused_gemv.py",
    "tests/test_type100_rocmfp4_dense_gemm_contract.py",
    "tests/test_type102_rocmfpx_dense_gemm_contract.py",
    "tests/test_type103_rocmfpx_dense_gemm_contract.py",
    "tests/test_type104_rocmfpx_dense_gemm_contract.py",
    "tests/test_type107_rocmfpx_dense_gemm_contract.py",
    "tests/test_rocmfpx_moe_type103_contract.py",
    "tests/test_lowbit_gap_formats.py",
    "tests/test_fp4_formats.py",
    "tests/test_numerics.py",
)

# Triton 3.7 refuses to compile kernels referencing non-constexpr globals unless
# this is set. Required on both backends.
TRITON_ENV = "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1"


@dataclasses.dataclass(frozen=True)
class Backend:
    """A GPU target and how to run a pytest invocation on it."""

    name: str
    description: str

    def available(self) -> tuple[bool, str]:
        """Return ``(usable, reason)``."""
        raise NotImplementedError

    def sync(self) -> tuple[bool, str]:
        """Copy the working tree to the backend.

        Without this the matrix silently tests whatever source the backend last
        saw, which is worse than not running it: a stale pass reads as a real
        pass, and a stale failure sends you debugging code you already fixed.
        """
        raise NotImplementedError

    def run(self, suite: str) -> subprocess.CompletedProcess[str]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class DockerBackend(Backend):
    """A backend reached through a running Docker container."""

    container: str
    workdir: str
    python: str
    extra_pythonpath: str = ""

    def available(self) -> tuple[bool, str]:
        if not shutil.which("docker"):
            return False, "docker not on PATH"
        probe = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            return False, f"container {self.container} not found"
        if probe.stdout.strip() != "true":
            return False, f"container {self.container} is not running"
        return True, "ready"

    def sync(self) -> tuple[bool, str]:
        for tree in ("vllm_gguf_plugin", "tests"):
            copied = subprocess.run(
                [
                    "docker",
                    "cp",
                    "-q",
                    f"{tree}/.",
                    f"{self.container}:{self.workdir}/{tree}",
                ],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                return False, copied.stderr.strip()[:120] or "docker cp failed"
        return True, "synced"

    def run(self, suite: str) -> subprocess.CompletedProcess[str]:
        pythonpath = ":".join(p for p in (self.extra_pythonpath, self.workdir) if p)
        script = (
            f"cd {self.workdir} && "
            f"{TRITON_ENV} PYTHONPATH={pythonpath} "
            f"{self.python} -m pytest {suite} -q"
        )
        return subprocess.run(
            ["docker", "exec", self.container, "bash", "-c", script],
            capture_output=True,
            text=True,
        )


@dataclasses.dataclass(frozen=True)
class KubernetesBackend(Backend):
    """A backend reached through a Kubernetes pod."""

    pod: str
    namespace: str
    workdir: str
    python: str

    def available(self) -> tuple[bool, str]:
        if not shutil.which("kubectl"):
            return False, "kubectl not on PATH"
        probe = subprocess.run(
            [
                "kubectl",
                "get",
                "pod",
                self.pod,
                "-n",
                self.namespace,
                "-o",
                "jsonpath={.status.phase}",
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            return False, f"pod {self.pod} not found in {self.namespace}"
        if probe.stdout.strip() != "Running":
            return False, f"pod {self.pod} is {probe.stdout.strip() or 'absent'}"
        return True, "ready"

    def sync(self) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory() as tmp:
            archive = pathlib.Path(tmp) / "src.tgz"
            packed = subprocess.run(
                ["tar", "czf", str(archive), "vllm_gguf_plugin", "tests"],
                capture_output=True,
                text=True,
            )
            if packed.returncode != 0:
                return False, packed.stderr.strip()[:120] or "tar failed"

            copied = subprocess.run(
                [
                    "kubectl",
                    "cp",
                    str(archive),
                    f"{self.namespace}/{self.pod}:/tmp/src.tgz",
                ],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                return False, copied.stderr.strip()[:120] or "kubectl cp failed"

        extracted = subprocess.run(
            [
                "kubectl",
                "exec",
                "-n",
                self.namespace,
                self.pod,
                "--",
                "bash",
                "-c",
                f"cd {self.workdir} && tar xzf /tmp/src.tgz",
            ],
            capture_output=True,
            text=True,
        )
        if extracted.returncode != 0:
            return False, extracted.stderr.strip()[:120] or "extract failed"
        return True, "synced"

    def run(self, suite: str) -> subprocess.CompletedProcess[str]:
        script = (
            f"cd {self.workdir} && "
            f"{TRITON_ENV} PYTHONPATH={self.workdir} "
            f"{self.python} -m pytest {suite} -q"
        )
        return subprocess.run(
            [
                "kubectl",
                "exec",
                "-n",
                self.namespace,
                self.pod,
                "--",
                "bash",
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
        )


BACKENDS: tuple[Backend, ...] = (
    DockerBackend(
        name="gfx1151",
        description="AMD Radeon 8060S (Strix Halo), ROCm 7.14",
        container="vllm-strix-build",
        workdir="/workspace/lib",
        python="/opt/venv/bin/python",
        # amdsmi must come from the ROCm install rather than PyPI, or platform
        # detection falls back to UnspecifiedPlatform and no GPU is found.
        extra_pythonpath=(
            "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi"
        ),
    ),
    KubernetesBackend(
        name="sm_121",
        description="NVIDIA GB10 (DGX), CUDA 13.0",
        pod="gguf-backend-matrix",
        namespace="default",
        workdir="/workspace",
        python="/opt/venv/bin/python",
    ),
)

_SUMMARY = re.compile(r"(?:(\d+) failed)?,?\s*(?:(\d+) passed)?,?\s*(?:(\d+) skipped)?")


@dataclasses.dataclass
class Result:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.failed == 0

    def __str__(self) -> str:
        if self.error:
            return f"ERROR ({self.error})"
        parts = [f"{self.passed}P"]
        if self.failed:
            parts.append(f"{self.failed}F")
        if self.skipped:
            parts.append(f"{self.skipped}S")
        return " ".join(parts)


def parse_summary(output: str) -> Result:
    """Extract counts from pytest's summary line."""
    result = Result()
    for line in reversed(output.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            counts = re.findall(r"(\d+) (passed|failed|skipped|error)", line)
            for count, label in counts:
                if label == "passed":
                    result.passed = int(count)
                elif label in ("failed", "error"):
                    result.failed += int(count)
                elif label == "skipped":
                    result.skipped = int(count)
            return result
    result.error = "no summary line"
    return result


def run_matrix(backends: Sequence[Backend], suites: Sequence[str]) -> int:
    """Run every suite on every available backend. Returns an exit status."""
    usable: list[Backend] = []
    for backend in backends:
        ok, reason = backend.available()
        status = "available" if ok else f"unavailable -- {reason}"
        print(f"  {backend.name:<10} {status}")
        if ok:
            usable.append(backend)

    if not usable:
        print("\nNo backend is available; nothing to compare.")
        return 1

    # Sync before running. A backend testing stale source produces a result
    # that looks authoritative and is not.
    print()
    synced: list[Backend] = []
    for backend in usable:
        ok, reason = backend.sync()
        print(f"  {backend.name:<10} sync: {reason}")
        if ok:
            synced.append(backend)
        else:
            print(f"  {backend.name:<10} skipped -- would have tested stale source")
    usable = synced

    if not usable:
        print("\nNo backend could be synced; refusing to report stale results.")
        return 1

    print()
    results: dict[str, dict[str, Result]] = {b.name: {} for b in usable}
    for suite in suites:
        short = suite.removeprefix("tests/").removesuffix(".py")
        for backend in usable:
            print(f"  {backend.name:<10} {short} ... ", end="", flush=True)
            completed = backend.run(suite)
            result = parse_summary(completed.stdout + completed.stderr)
            results[backend.name][suite] = result
            print(result, flush=True)

    return report(usable, suites, results)


def report(
    backends: Sequence[Backend],
    suites: Sequence[str],
    results: dict[str, dict[str, Result]],
) -> int:
    """Print the matrix and return an exit status."""
    width = max(len(s.removeprefix("tests/").removesuffix(".py")) for s in suites)
    columns = [b.name for b in backends]

    print("\n" + "=" * (width + 2 + sum(len(c) + 3 for c in columns)))
    header = "suite".ljust(width) + "  " + "  ".join(c.ljust(12) for c in columns)
    print(header)
    print("-" * len(header))

    failures = 0
    for suite in suites:
        short = suite.removeprefix("tests/").removesuffix(".py")
        cells = []
        for name in columns:
            result = results[name][suite]
            cells.append(str(result).ljust(12))
            if not result.ok:
                failures += 1
        print(short.ljust(width) + "  " + "  ".join(cells))

    print("=" * len(header))

    # A suite that passes on one backend and is skipped on another is not a
    # cross-check: it means the second backend never ran the kernel. Call that
    # out, since it is the failure mode this script exists to prevent.
    if len(columns) > 1:
        for suite in suites:
            states = {n: results[n][suite] for n in columns}
            ran = {n for n, r in states.items() if r.passed and not r.skipped}
            skipped = {n for n, r in states.items() if r.skipped and not r.passed}
            if ran and skipped:
                short = suite.removeprefix("tests/").removesuffix(".py")
                print(
                    f"note: {short} ran on {', '.join(sorted(ran))} but was "
                    f"skipped on {', '.join(sorted(skipped))}"
                )

    if failures:
        print(f"\n{failures} suite/backend combination(s) failed")
        return 1
    print("\nall suites pass on all available backends")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        action="append",
        choices=[b.name for b in BACKENDS],
        help="restrict to one backend (repeatable)",
    )
    parser.add_argument(
        "--suite",
        action="append",
        help="restrict to one suite path (repeatable)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="show configured backends and suites, then exit",
    )
    args = parser.parse_args()

    backends = [b for b in BACKENDS if not args.backend or b.name in args.backend]
    suites = args.suite or list(KERNEL_SUITES)

    if args.list:
        print("backends:")
        for backend in backends:
            ok, reason = backend.available()
            mark = "+" if ok else "-"
            print(f"  {mark} {backend.name:<10} {backend.description}")
            if not ok:
                print(f"    {reason}")
        print("\nsuites:")
        for suite in suites:
            print(f"    {suite}")
        return 0

    return run_matrix(backends, suites)


if __name__ == "__main__":
    sys.exit(main())
