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
"""Statistical calculations for benchmark data."""

from __future__ import annotations

import math
from typing import Sequence


def calculate_statistics(values: Sequence[float]) -> tuple[float, float]:
    """Calculate mean and standard deviation.

    Args:
        values: Sequence of numeric values.

    Returns:
        Tuple of (mean, standard_deviation).

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("Cannot calculate statistics on empty sequence")

    n = len(values)
    mean = sum(values) / n

    if n < 2:
        return mean, 0.0

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    stdev = math.sqrt(variance)

    return mean, stdev


def calculate_confidence_interval(
    mean: float,
    stdev: float,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate confidence interval bounds for the sample mean.

    Uses the formula: mean ± z × (stdev / √n), where stdev / √n is the
    standard error of the mean.

    z-score approximation:
    - 95% confidence: z = 1.96
    - 99% confidence: z = 2.576

    Args:
        mean: Sample mean.
        stdev: Sample standard deviation.
        n: Sample size.
        confidence: Confidence level (default 0.95 for 95%).

    Returns:
        Tuple of (lower_bound, upper_bound).

    Raises:
        ValueError: If n is not positive.
    """
    if n <= 0:
        raise ValueError("Sample size n must be greater than zero")

    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    else:
        z = 1.96

    se = stdev / math.sqrt(n)
    margin = z * se
    return mean - margin, mean + margin
