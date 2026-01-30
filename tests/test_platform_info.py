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
"""Tests for zkbench.platform_info module."""
from __future__ import annotations

from unittest import mock

from zkbench.platform_info import get_cpu_vendor, get_platform_info


class TestGetCpuVendor:
    """Tests for get_cpu_vendor function."""

    def test_linux_detection(self) -> None:
        """Should detect CPU on Linux."""
        cpuinfo = "model name\t: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz\n"
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("builtins.open", mock.mock_open(read_data=cpuinfo)),
        ):
            result = get_cpu_vendor()
        assert result == "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"

    def test_linux_cpuinfo_not_found(self) -> None:
        """Should return None if /proc/cpuinfo not accessible."""
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("builtins.open", side_effect=OSError()),
        ):
            result = get_cpu_vendor()
        assert result is None

    def test_macos_detection(self) -> None:
        """Should detect CPU on macOS."""
        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = mock.Mock(
                stdout="Apple M1 Pro\n", returncode=0
            )
            result = get_cpu_vendor()
        assert result == "Apple M1 Pro"

    def test_macos_sysctl_fails(self) -> None:
        """Should return None if sysctl fails on macOS."""
        import subprocess

        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = subprocess.CalledProcessError(1, "sysctl")
            result = get_cpu_vendor()
        assert result is None

    def test_windows_detection(self) -> None:
        """Should detect CPU on Windows from environment."""
        with (
            mock.patch("platform.system", return_value="Windows"),
            mock.patch.dict("os.environ", {"PROCESSOR_IDENTIFIER": "Intel64"}),
        ):
            result = get_cpu_vendor()
        assert result == "Intel64"

    def test_unknown_platform(self) -> None:
        """Should return None for unknown platform."""
        with mock.patch("platform.system", return_value="FreeBSD"):
            result = get_cpu_vendor()
        assert result is None


class TestGetPlatformInfo:
    """Tests for get_platform_info function."""

    def test_returns_platform_object(self) -> None:
        """Should return Platform instance."""
        from zkbench.schema import Platform

        result = get_platform_info()
        assert isinstance(result, Platform)

    def test_platform_fields_populated(self) -> None:
        """Should populate all required fields."""
        result = get_platform_info()
        assert result.os in ("linux", "darwin", "windows")
        assert result.arch != ""
        assert result.cpu_count >= 1

    def test_cpu_count_fallback(self) -> None:
        """Should default to 1 if cpu_count returns None."""
        with mock.patch("os.cpu_count", return_value=None):
            result = get_platform_info()
        assert result.cpu_count == 1
