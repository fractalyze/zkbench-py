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

__version__ = "0.1.0"

from zkbench.schema import (
    BenchmarkReport,
    BenchmarkResult,
    Metadata,
    MetricValue,
    Platform,
    TestVectors,
)
from zkbench.statistics import calculate_confidence_interval, calculate_statistics
from zkbench.utils import compute_array_hash, get_git_commit_sha

__all__ = [
    "__version__",
    "BenchmarkReport",
    "BenchmarkResult",
    "Metadata",
    "MetricValue",
    "Platform",
    "TestVectors",
    "calculate_confidence_interval",
    "calculate_statistics",
    "compute_array_hash",
    "get_git_commit_sha",
]
