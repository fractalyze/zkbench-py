# Copyright 2026 The zkbench-py Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reusable base class for JAX benchmarks.

Provides ``JaxBenchmark``, a template-method ABC that handles warmup, timing,
memory measurement, statistics, verification, CLI parsing, and JSON output so
that concrete benchmarks only need to define ``get_config`` and ``get_ops``.
"""

from __future__ import annotations

import abc
import argparse
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from zkbench.schema import (
    BenchmarkReport,
    BenchmarkResult,
    Metadata,
    MetricValue,
    TestVectors,
)
from zkbench.statistics import calculate_confidence_interval, calculate_statistics


@dataclass
class BenchmarkConfig:
    """Static configuration for a benchmark suite."""

    implementation: str
    version: str
    default_iterations: int = 100
    default_warmup: int = 10


@dataclass
class BenchmarkOp:
    """A single benchmarkable operation.

    Attributes:
        name: Benchmark key in the output report (e.g. ``"dft_2p20"``).
        fn: Zero-arg callable that performs the JAX operation and returns its
            result.  The result is passed to ``jax.block_until_ready`` after
            each call unless it is ``None``.
        metadata: Extra key-value pairs written to ``BenchmarkResult.metadata``.
        throughput_unit: Unit string for the throughput metric.
        throughput_count: Logical operation count per invocation (e.g.
            ``num_leaves - 1`` for Merkle compressions).
        input_hash: Pre-computed hash of the input data.
        output_hash_fn: Optional callable that returns the output hash string.
        verify_fn: Optional callable that returns ``True`` if the result is
            correct.
        measure_memory: If ``False``, the memory metric is omitted from the
            result.
    """

    name: str
    fn: Callable[[], Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    throughput_unit: str = "ops/s"
    throughput_count: int = 1
    input_hash: str = ""
    output_hash_fn: Callable[[], str] | None = None
    verify_fn: Callable[[], bool] | None = None
    measure_memory: bool = True


class JaxBenchmark(abc.ABC):
    """Template-method base class for JAX benchmarks."""

    @abc.abstractmethod
    def get_config(self) -> BenchmarkConfig:
        """Return the benchmark configuration."""

    @abc.abstractmethod
    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        """Yield ``BenchmarkOp`` instances for the given CLI arguments."""

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        """Override to register benchmark-specific CLI flags."""

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, argv: list[str] | None = None) -> int:
        """Parse CLI args, run all ops, and emit the JSON report."""
        report = self.run_to_report(argv)
        json_output = report.to_json()

        args = self._last_args
        if args and args.output:
            with open(args.output, "w") as f:
                f.write(json_output)
            print(f"Results written to {args.output}")
        else:
            print(json_output)

        for name, bench in report.benchmarks.items():
            if bench.test_vectors and not bench.test_vectors.verified:
                print(
                    f"ERROR: Test vector verification failed for '{name}'!",
                    file=sys.stderr,
                )
                return 1
        return 0

    def run_to_report(self, argv: list[str] | None = None) -> BenchmarkReport:
        """Parse CLI args, run all ops, and return the report.

        Use this instead of ``run()`` when you need programmatic access to
        results (e.g. in tests).
        """
        config = self.get_config()

        parser = argparse.ArgumentParser()
        parser.add_argument("--output", "-o", type=str, default=None)
        parser.add_argument(
            "--iterations",
            type=int,
            default=config.default_iterations,
        )
        parser.add_argument(
            "--warmup",
            type=int,
            default=config.default_warmup,
        )
        self.add_custom_args(parser)
        args = parser.parse_args(argv)
        self._last_args = args

        benchmarks: dict[str, BenchmarkResult] = {}
        for op in self.get_ops(args):
            benchmarks[op.name] = self._run_single_op(op, args.iterations, args.warmup)

        metadata = Metadata.create(
            implementation=config.implementation,
            version=config.version,
        )
        return BenchmarkReport(metadata=metadata, benchmarks=benchmarks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_single_op(
        op: BenchmarkOp, iterations: int, warmup: int
    ) -> BenchmarkResult:
        """Warmup, time, measure memory, verify, and assemble the result."""
        import jax

        fn = op.fn

        # Warmup
        for _ in range(warmup):
            result = fn()
            if result is not None:
                jax.block_until_ready(result)

        # Timing
        latencies: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            result = fn()
            if result is not None:
                jax.block_until_ready(result)
            end = time.perf_counter_ns()
            latencies.append(end - start)

        mean, stdev = calculate_statistics(latencies)
        lower, upper = calculate_confidence_interval(mean, stdev)

        # Memory
        peak_memory: int | None = None
        if op.measure_memory:
            result = fn()
            if result is not None:
                jax.block_until_ready(result)
            tracemalloc.start()
            result = fn()
            if result is not None:
                jax.block_until_ready(result)
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        # Throughput
        throughput = op.throughput_count * 1e9 / mean if mean > 0 else 0

        # Test vectors
        output_hash = op.output_hash_fn() if op.output_hash_fn else ""
        verified = op.verify_fn() if op.verify_fn else True

        test_vectors = TestVectors(
            input_hash=op.input_hash,
            output_hash=output_hash,
            verified=verified,
        )

        result_obj = BenchmarkResult(
            latency=MetricValue(
                value=mean, unit="ns", lower_value=lower, upper_value=upper
            ),
            throughput=MetricValue(value=throughput, unit=op.throughput_unit),
            iterations=iterations,
            test_vectors=test_vectors,
            metadata=op.metadata,
        )
        if peak_memory is not None:
            result_obj.memory = MetricValue(value=peak_memory, unit="bytes")
        return result_obj
