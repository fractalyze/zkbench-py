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

from zkbench.benchmark import _device_peak_bytes


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
