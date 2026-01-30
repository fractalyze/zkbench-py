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
"""Common JSON schema for cross-implementation benchmark results."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from zkbench.platform_info import get_platform_info
from zkbench.utils import get_git_commit_sha


@dataclass
class MetricValue:
    """Represents a benchmark metric with optional confidence bounds."""

    value: float
    unit: str
    lower_value: float | None = None
    upper_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"value": self.value, "unit": self.unit}
        if self.lower_value is not None:
            result["lower_value"] = self.lower_value
        if self.upper_value is not None:
            result["upper_value"] = self.upper_value
        return result


@dataclass
class TestVectors:
    """Test vector verification information."""

    input_hash: str
    output_hash: str
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "verified": self.verified,
        }


@dataclass
class BenchmarkResult:
    """Represents results from a single benchmark."""

    latency: MetricValue | None = None
    memory: MetricValue | None = None
    throughput: MetricValue | None = None
    iterations: int = 0
    test_vectors: TestVectors | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.latency is not None:
            result["latency"] = self.latency.to_dict()
        if self.memory is not None:
            result["memory"] = self.memory.to_dict()
        if self.throughput is not None:
            result["throughput"] = self.throughput.to_dict()
        if self.iterations > 0:
            result["iterations"] = self.iterations
        if self.test_vectors is not None:
            result["test_vectors"] = self.test_vectors.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class Platform:
    """Platform information."""

    os: str
    arch: str
    cpu_count: int
    cpu_vendor: str | None = None

    @classmethod
    def current(cls) -> Platform:
        return get_platform_info()

    def to_dict(self) -> dict[str, str | int | None]:
        result: dict[str, str | int | None] = {
            "os": self.os,
            "arch": self.arch,
            "cpu_count": self.cpu_count,
        }
        if self.cpu_vendor is not None:
            result["cpu_vendor"] = self.cpu_vendor
        return result


@dataclass
class Metadata:
    """Benchmark metadata."""

    implementation: str
    version: str
    commit_sha: str
    timestamp: str
    platform: Platform

    @classmethod
    def create(
        cls,
        implementation: str,
        version: str,
    ) -> Metadata:
        commit_sha = get_git_commit_sha()
        timestamp = datetime.now(timezone.utc).isoformat()
        return cls(
            implementation=implementation,
            version=version,
            commit_sha=commit_sha,
            timestamp=timestamp,
            platform=Platform.current(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "version": self.version,
            "commit_sha": self.commit_sha,
            "timestamp": self.timestamp,
            "platform": self.platform.to_dict(),
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    metadata: Metadata
    benchmarks: dict[str, BenchmarkResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"metadata": self.metadata.to_dict()}
        result["benchmarks"] = {
            name: bench.to_dict() for name, bench in self.benchmarks.items()
        }
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
