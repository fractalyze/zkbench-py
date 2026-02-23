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
"""Utility functions for benchmarking."""

from __future__ import annotations

import hashlib
import subprocess
from typing import Any


def get_git_commit_sha() -> str:
    """Get current git commit SHA (first 12 characters).

    Returns:
        Truncated git commit SHA, or "unknown" if not in a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def compute_array_hash(arr: Any) -> str:
    """Compute SHA256 hash of array data for verification.

    Supports numpy arrays, JAX arrays, and Python lists.

    Args:
        arr: Array-like object with tolist() method or Python list.

    Returns:
        Hexadecimal SHA256 hash string.
    """
    import numpy as np

    if hasattr(arr, "tolist"):
        data = np.array(arr.tolist(), dtype=np.uint32)
    else:
        data = np.array(arr, dtype=np.uint32)
    return hashlib.sha256(data.tobytes()).hexdigest()
