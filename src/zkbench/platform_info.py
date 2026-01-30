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
"""Platform information detection utilities."""
from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zkbench.schema import Platform


def get_cpu_vendor() -> str | None:
    """Detect CPU vendor/model string.

    Returns CPU vendor information from:
    - Linux: /proc/cpuinfo
    - macOS: sysctl -n machdep.cpu.brand_string
    - Windows: PROCESSOR_IDENTIFIER environment variable

    Returns:
        CPU vendor string, or None if detection fails.
    """
    system = platform.system().lower()

    if system == "linux":
        return _get_cpu_vendor_linux()
    elif system == "darwin":
        return _get_cpu_vendor_macos()
    elif system == "windows":
        return _get_cpu_vendor_windows()
    return None


def _get_cpu_vendor_linux() -> str | None:
    """Get CPU vendor from /proc/cpuinfo on Linux."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    match = re.search(r":\s*(.+)", line)
                    if match:
                        return match.group(1).strip()
    except (OSError, IOError):
        pass
    return None


def _get_cpu_vendor_macos() -> str | None:
    """Get CPU vendor from sysctl on macOS."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def _get_cpu_vendor_windows() -> str | None:
    """Get CPU vendor from environment on Windows."""
    return os.environ.get("PROCESSOR_IDENTIFIER")


def get_platform_info() -> "Platform":
    """Get current platform information.

    Returns:
        Platform dataclass with os, arch, cpu_count, and cpu_vendor.
    """
    from zkbench.schema import Platform

    return Platform(
        os=platform.system().lower(),
        arch=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        cpu_vendor=get_cpu_vendor(),
    )
