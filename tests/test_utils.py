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
"""Tests for zkbench.utils module."""
from __future__ import annotations

from unittest import mock

import numpy as np

from zkbench.utils import compute_array_hash, compute_hash, get_git_commit_sha


class TestGetGitCommitSha:
    """Tests for get_git_commit_sha function."""

    def test_returns_truncated_sha(self) -> None:
        """Should return first 12 characters of SHA."""
        full_sha = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(stdout=full_sha + "\n")
            result = get_git_commit_sha()
        assert result == "a1b2c3d4e5f6"
        assert len(result) == 12

    def test_returns_unknown_when_not_git_repo(self) -> None:
        """Should return 'unknown' when not in git repo."""
        import subprocess

        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(128, "git")
            result = get_git_commit_sha()
        assert result == "unknown"

    def test_returns_unknown_when_git_not_found(self) -> None:
        """Should return 'unknown' when git is not installed."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = get_git_commit_sha()
        assert result == "unknown"


class TestComputeHash:
    """Tests for compute_hash function."""

    def test_empty_input(self) -> None:
        """SHA-256 of empty bytes."""
        assert (
            compute_hash(b"")
            == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_abc(self) -> None:
        """SHA-256 of b'abc'."""
        assert (
            compute_hash(b"abc")
            == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_uint32_array_le(self) -> None:
        """SHA-256 of [1, 2, 3] as uint32 little-endian bytes."""
        import struct

        data = struct.pack("<III", 1, 2, 3)
        assert (
            compute_hash(data)
            == "4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d"
        )


class TestComputeArrayHash:
    """Tests for compute_array_hash function."""

    def test_numpy_array(self) -> None:
        """Should hash numpy array."""
        arr = np.array([1, 2, 3, 4], dtype=np.uint32)
        result = compute_array_hash(arr)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex length

    def test_python_list(self) -> None:
        """Should hash Python list."""
        result = compute_array_hash([1, 2, 3, 4])
        assert isinstance(result, str)
        assert len(result) == 64

    def test_same_data_same_hash(self) -> None:
        """Same data should produce same hash."""
        arr1 = np.array([1, 2, 3], dtype=np.uint32)
        arr2 = np.array([1, 2, 3], dtype=np.uint32)
        assert compute_array_hash(arr1) == compute_array_hash(arr2)

    def test_different_data_different_hash(self) -> None:
        """Different data should produce different hash."""
        arr1 = np.array([1, 2, 3], dtype=np.uint32)
        arr2 = np.array([1, 2, 4], dtype=np.uint32)
        assert compute_array_hash(arr1) != compute_array_hash(arr2)

    def test_correct_hash_value(self) -> None:
        """Should produce the known hash for [1, 2, 3] as little-endian uint32."""
        expected = "4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d"
        assert compute_array_hash([1, 2, 3]) == expected

    def test_object_with_tolist_method(self) -> None:
        """Should handle objects with tolist() method."""

        class ArrayLike:
            def tolist(self) -> list[int]:
                return [1, 2, 3]

        result = compute_array_hash(ArrayLike())
        expected = compute_array_hash([1, 2, 3])
        assert result == expected
