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
"""Tests for zkbench.benchmark helpers."""

from __future__ import annotations

import sys
import types

import pytest

from zkbench.benchmark import BenchmarkOp, FrxBenchmark, _device_peak_bytes


class _FakeDevice:
    def __init__(self, stats: object) -> None:
        self._stats = stats

    def memory_stats(self) -> object:
        return self._stats


def test_reads_peak_from_device_stats() -> None:
    assert _device_peak_bytes(_FakeDevice({"peak_bytes_in_use": 4096})) == 4096


def test_none_when_peak_key_absent() -> None:
    # A backend that reports memory_stats but not the peak key (e.g. only
    # bytes_in_use) yields None, so the caller falls back to host tracemalloc.
    assert _device_peak_bytes(_FakeDevice({"bytes_in_use": 4096})) is None


def test_none_when_memory_stats_returns_none() -> None:
    assert _device_peak_bytes(_FakeDevice(None)) is None


def test_none_when_device_lacks_memory_stats() -> None:
    # The CPU backend's device has no memory_stats at all.
    assert _device_peak_bytes(object()) is None


# ---------------------------------------------------------------------------
# Phase selection in _run_single_op. zkbench imports the backend lazily so the
# library doesn't hard-depend on it; these inject a fake one so the phase
# branching is exercised without a real backend.
# ---------------------------------------------------------------------------


def _fake_backend() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        devices=lambda: [object()],  # no memory_stats -> device peak is None
        block_until_ready=lambda x: x,
    )


@pytest.fixture
def fake_frx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "frx", _fake_backend())


class _FakeLowered:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def compile(self) -> object:
        self._log.append("compiled")
        return object()


@pytest.mark.usefixtures("fake_frx")
def test_compile_phase_only_times_the_compile() -> None:
    log: list[str] = []
    op = BenchmarkOp(name="op", fn=lambda: None, lower=lambda: _FakeLowered(log))
    res = FrxBenchmark._run_single_op(op, iterations=2, warmup=1, phase="compile")
    assert log == ["compiled"]  # lowered.compile() ran
    assert res.compile_time is not None
    assert res.latency is None  # runtime phase skipped
    assert res.iterations == 0


@pytest.mark.usefixtures("fake_frx")
def test_runtime_phase_skips_compile() -> None:
    op = BenchmarkOp(name="op", fn=lambda: None, measure_memory=False)
    res = FrxBenchmark._run_single_op(op, iterations=2, warmup=1, phase="runtime")
    assert res.latency is not None
    assert res.iterations == 2
    assert res.compile_time is None


@pytest.mark.usefixtures("fake_frx")
def test_both_phases_populate_compile_and_runtime() -> None:
    log: list[str] = []
    op = BenchmarkOp(
        name="op",
        fn=lambda: None,
        measure_memory=False,
        lower=lambda: _FakeLowered(log),
    )
    res = FrxBenchmark._run_single_op(op, iterations=2, warmup=1, phase="both")
    assert log == ["compiled"]
    assert res.compile_time is not None
    assert res.latency is not None


@pytest.mark.usefixtures("fake_frx")
def test_compile_phase_no_op_without_lower() -> None:
    # An op without a lower thunk has no compile phase to measure.
    op = BenchmarkOp(name="op", fn=lambda: None, measure_memory=False)
    res = FrxBenchmark._run_single_op(op, iterations=1, warmup=0, phase="compile")
    assert res.compile_time is None
    assert res.latency is None
