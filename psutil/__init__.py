"""Minimal `psutil` shim for test environments.

This repository's test suite optionally depends on `psutil` for RSS memory
measurements. The upstream dependency is not always available in constrained
environments, so we provide a tiny subset of the API needed by the tests:

- `psutil.Process(pid).memory_info().rss`
"""

from __future__ import annotations

import os
import sys
from collections import namedtuple
from typing import Optional

_MemInfo = namedtuple("pmem", ["rss"])


def _get_rss_bytes(pid: int) -> int:
    # Windows: Query WorkingSetSize via GetProcessMemoryInfo.
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_INFORMATION = 0x0400
            PROCESS_VM_READ = 0x0010

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, int(pid)
            )
            if not handle:
                return 0
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            ctypes.windll.kernel32.CloseHandle(handle)
            if not ok:
                return 0
            return int(counters.WorkingSetSize)
        except Exception:
            return 0

    # POSIX: fall back to resource module.
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        if sys.platform == "darwin":
            return int(rss)
        return int(rss) * 1024
    except Exception:
        return 0


class Process:
    def __init__(self, pid: Optional[int] = None):
        self.pid = int(os.getpid() if pid is None else pid)

    def memory_info(self):
        return _MemInfo(rss=_get_rss_bytes(self.pid))


__all__ = ["Process"]

