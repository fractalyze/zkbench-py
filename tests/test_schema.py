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
"""Tests for zkbench.schema module."""
from __future__ import annotations

import json

from zkbench.schema import (
    BenchmarkReport,
    BenchmarkResult,
    Metadata,
    MetricValue,
    Platform,
    TestVectors,
)


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_to_dict_basic(self) -> None:
        """Should convert basic metric to dict."""
        metric = MetricValue(value=1.5, unit="ms")
        result = metric.to_dict()
        assert result == {"value": 1.5, "unit": "ms"}

    def test_to_dict_with_bounds(self) -> None:
        """Should include bounds when present."""
        metric = MetricValue(
            value=1.5, unit="ms", lower_value=1.0, upper_value=2.0
        )
        result = metric.to_dict()
        assert result == {
            "value": 1.5,
            "unit": "ms",
            "lower_value": 1.0,
            "upper_value": 2.0,
        }

    def test_to_dict_partial_bounds(self) -> None:
        """Should include only present bounds."""
        metric = MetricValue(value=1.5, unit="ms", lower_value=1.0)
        result = metric.to_dict()
        assert result == {"value": 1.5, "unit": "ms", "lower_value": 1.0}


class TestTestVectors:
    """Tests for TestVectors dataclass."""

    def test_to_dict(self) -> None:
        """Should convert to dict correctly."""
        vectors = TestVectors(
            input_hash="abc123", output_hash="def456", verified=True
        )
        result = vectors.to_dict()
        assert result == {
            "input_hash": "abc123",
            "output_hash": "def456",
            "verified": True,
        }


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_to_dict_empty(self) -> None:
        """Empty result should return empty dict."""
        result = BenchmarkResult()
        assert result.to_dict() == {}

    def test_to_dict_with_latency(self) -> None:
        """Should include latency when present."""
        result = BenchmarkResult(latency=MetricValue(value=1.5, unit="ms"))
        assert result.to_dict() == {"latency": {"value": 1.5, "unit": "ms"}}

    def test_to_dict_with_all_fields(self) -> None:
        """Should include all fields when present."""
        result = BenchmarkResult(
            latency=MetricValue(value=1.5, unit="ms"),
            memory=MetricValue(value=100.0, unit="MB"),
            throughput=MetricValue(value=1000.0, unit="ops/s"),
            iterations=100,
            test_vectors=TestVectors(
                input_hash="abc", output_hash="def", verified=True
            ),
            metadata={"custom": "value"},
        )
        d = result.to_dict()
        assert "latency" in d
        assert "memory" in d
        assert "throughput" in d
        assert d["iterations"] == 100
        assert "test_vectors" in d
        assert d["metadata"] == {"custom": "value"}

    def test_iterations_zero_excluded(self) -> None:
        """Zero iterations should be excluded."""
        result = BenchmarkResult(iterations=0)
        assert "iterations" not in result.to_dict()


class TestPlatform:
    """Tests for Platform dataclass."""

    def test_to_dict_basic(self) -> None:
        """Should convert basic platform to dict."""
        platform = Platform(os="linux", arch="x86_64", cpu_count=8)
        result = platform.to_dict()
        assert result == {"os": "linux", "arch": "x86_64", "cpu_count": 8}

    def test_to_dict_with_vendor(self) -> None:
        """Should include vendor when present."""
        platform = Platform(
            os="linux", arch="x86_64", cpu_count=8, cpu_vendor="Intel"
        )
        result = platform.to_dict()
        assert result["cpu_vendor"] == "Intel"

    def test_current(self) -> None:
        """Platform.current() should return valid platform."""
        platform = Platform.current()
        assert platform.os in ("linux", "darwin", "windows")
        assert platform.arch != ""
        assert platform.cpu_count >= 1


class TestMetadata:
    """Tests for Metadata dataclass."""

    def test_to_dict(self) -> None:
        """Should convert metadata to dict."""
        platform = Platform(os="linux", arch="x86_64", cpu_count=8)
        metadata = Metadata(
            implementation="test",
            version="1.0.0",
            commit_sha="abc123",
            timestamp="2026-01-01T00:00:00Z",
            platform=platform,
        )
        result = metadata.to_dict()
        assert result["implementation"] == "test"
        assert result["version"] == "1.0.0"
        assert result["commit_sha"] == "abc123"
        assert result["timestamp"] == "2026-01-01T00:00:00Z"
        assert "platform" in result

    def test_create(self) -> None:
        """Metadata.create() should populate fields automatically."""
        metadata = Metadata.create(implementation="test", version="1.0.0")
        assert metadata.implementation == "test"
        assert metadata.version == "1.0.0"
        assert metadata.commit_sha != ""
        assert metadata.timestamp != ""
        assert metadata.platform is not None


class TestBenchmarkReport:
    """Tests for BenchmarkReport dataclass."""

    def test_to_dict(self) -> None:
        """Should convert report to dict."""
        platform = Platform(os="linux", arch="x86_64", cpu_count=8)
        metadata = Metadata(
            implementation="test",
            version="1.0.0",
            commit_sha="abc123",
            timestamp="2026-01-01T00:00:00Z",
            platform=platform,
        )
        report = BenchmarkReport(
            metadata=metadata,
            benchmarks={
                "bench1": BenchmarkResult(
                    latency=MetricValue(value=1.0, unit="ms")
                )
            },
        )
        result = report.to_dict()
        assert "metadata" in result
        assert "benchmarks" in result
        assert "bench1" in result["benchmarks"]

    def test_to_json(self) -> None:
        """Should convert report to valid JSON."""
        platform = Platform(os="linux", arch="x86_64", cpu_count=8)
        metadata = Metadata(
            implementation="test",
            version="1.0.0",
            commit_sha="abc123",
            timestamp="2026-01-01T00:00:00Z",
            platform=platform,
        )
        report = BenchmarkReport(metadata=metadata)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["metadata"]["implementation"] == "test"

    def test_empty_benchmarks(self) -> None:
        """Empty benchmarks should be empty dict."""
        platform = Platform(os="linux", arch="x86_64", cpu_count=8)
        metadata = Metadata(
            implementation="test",
            version="1.0.0",
            commit_sha="abc123",
            timestamp="2026-01-01T00:00:00Z",
            platform=platform,
        )
        report = BenchmarkReport(metadata=metadata)
        assert report.to_dict()["benchmarks"] == {}
