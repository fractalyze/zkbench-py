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
"""Tests for zkbench.statistics module."""

from __future__ import annotations

import math

import pytest

from zkbench.statistics import calculate_confidence_interval, calculate_statistics


class TestCalculateStatistics:
    """Tests for calculate_statistics function."""

    def test_single_value(self) -> None:
        """Single value should return mean with zero stdev."""
        mean, stdev = calculate_statistics([5.0])
        assert mean == 5.0
        assert stdev == 0.0

    def test_multiple_values(self) -> None:
        """Multiple values should return correct mean and stdev."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        mean, stdev = calculate_statistics(values)
        assert mean == 5.0
        # Sample stdev: sqrt(32/7) ≈ 2.138
        assert math.isclose(stdev, math.sqrt(32 / 7), rel_tol=1e-9)

    def test_identical_values(self) -> None:
        """Identical values should return zero stdev."""
        mean, stdev = calculate_statistics([3.0, 3.0, 3.0, 3.0])
        assert mean == 3.0
        assert stdev == 0.0

    def test_empty_sequence_raises(self) -> None:
        """Empty sequence should raise ValueError."""
        with pytest.raises(ValueError, match="empty sequence"):
            calculate_statistics([])

    def test_tuple_input(self) -> None:
        """Should accept tuple as input."""
        mean, stdev = calculate_statistics((1.0, 2.0, 3.0))
        assert mean == 2.0


class TestCalculateConfidenceInterval:
    """Tests for calculate_confidence_interval function."""

    def test_95_confidence(self) -> None:
        """95% confidence: mean=100, stdev=10, n=25 → se=2, margin=3.92."""
        lower, upper = calculate_confidence_interval(100.0, 10.0, 25, confidence=0.95)
        assert math.isclose(lower, 96.08, rel_tol=1e-9)
        assert math.isclose(upper, 103.92, rel_tol=1e-9)

    def test_99_confidence(self) -> None:
        """99% confidence: mean=100, stdev=10, n=25 → se=2, margin=5.152."""
        lower, upper = calculate_confidence_interval(100.0, 10.0, 25, confidence=0.99)
        assert math.isclose(lower, 100.0 - 5.152, rel_tol=1e-9)
        assert math.isclose(upper, 100.0 + 5.152, rel_tol=1e-9)

    def test_default_confidence(self) -> None:
        """Default confidence should be 95%."""
        lower, upper = calculate_confidence_interval(100.0, 10.0, 25)
        assert math.isclose(lower, 96.08, rel_tol=1e-9)
        assert math.isclose(upper, 103.92, rel_tol=1e-9)

    def test_zero_stdev(self) -> None:
        """Zero stdev should return mean as both bounds."""
        lower, upper = calculate_confidence_interval(5.0, 0.0, 10)
        assert lower == 5.0
        assert upper == 5.0

    def test_single_sample(self) -> None:
        """n=1: se = stdev, margin = 1.96 × stdev."""
        lower, upper = calculate_confidence_interval(50.0, 5.0, 1, confidence=0.95)
        assert math.isclose(lower, 40.2, rel_tol=1e-9)
        assert math.isclose(upper, 59.8, rel_tol=1e-9)

    def test_zero_n_raises(self) -> None:
        """n=0 should raise ValueError."""
        with pytest.raises(ValueError, match="greater than zero"):
            calculate_confidence_interval(100.0, 10.0, 0)

    def test_negative_n_raises(self) -> None:
        """Negative n should raise ValueError."""
        with pytest.raises(ValueError, match="greater than zero"):
            calculate_confidence_interval(100.0, 10.0, -5)

    def test_unknown_confidence_defaults_to_95(self) -> None:
        """Unknown confidence level should default to z=1.96."""
        lower, upper = calculate_confidence_interval(100.0, 10.0, 25, confidence=0.90)
        assert math.isclose(lower, 96.08, rel_tol=1e-9)
        assert math.isclose(upper, 103.92, rel_tol=1e-9)
