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
"""zkbench - Reusable benchmarking library for zero-knowledge proofs."""

from importlib.metadata import PackageNotFoundError, version as _dist_version

# Read from the installed distribution rather than restating pyproject.toml's
# literal: the hardcoded copy this replaces sat at 0.5.0 while three releases
# shipped past it, so every install reported a version it was not.
try:
    __version__ = _dist_version("zkbench")
except PackageNotFoundError:
    # Running from a source tree that was never installed (no dist-info to
    # read). Only the metadata can answer this, so say so rather than guess a
    # number that would be wrong the moment pyproject moves.
    __version__ = "0.0.0+unknown"

from zkbench.benchmark import BenchmarkConfig, BenchmarkOp, FrxBenchmark
from zkbench.schema import (
    BenchmarkReport,
    BenchmarkResult,
    Metadata,
    MetricValue,
    Platform,
    TestVectors,
)
from zkbench.statistics import calculate_confidence_interval, calculate_statistics
from zkbench.utils import compute_array_hash, compute_hash, get_git_commit_sha

__all__ = [
    "__version__",
    "BenchmarkConfig",
    "BenchmarkOp",
    "BenchmarkReport",
    "BenchmarkResult",
    "FrxBenchmark",
    "Metadata",
    "MetricValue",
    "Platform",
    "TestVectors",
    "calculate_confidence_interval",
    "calculate_statistics",
    "compute_array_hash",
    "compute_hash",
    "get_git_commit_sha",
]
