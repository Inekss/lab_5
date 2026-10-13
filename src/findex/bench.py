from __future__ import annotations

import ctypes
import gc
import os
import sys
import threading
from ctypes import wintypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, process_time

from findex.documents import read_text
from findex.index import CORPUS, WITH_POSITIONS, build_all

RUNS = 3
EXECUTORS = ("serial", "threads", "processes")
COUNTS = sorted({1, 2, 4, 8, os.cpu_count() or 1})


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


# HANDLE is a pointer. Without argtypes the 64-bit pseudo-handle is truncated and RSS stays 0.
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_kernel = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
_kernel.GetCurrentProcess.restype = wintypes.HANDLE


def rss_bytes() -> int:
    # WorkingSetSize of this process during the cell, not the lifetime peak
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(pmc)
    ok = _psapi.GetProcessMemoryInfo(_kernel.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(pmc.WorkingSetSize)


def cpu_model() -> str:
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
    ) as key:
        name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    return str(name).strip()


def gil_enabled() -> bool:
    fn = getattr(sys, "_is_gil_enabled", None)
    return True if fn is None else bool(fn())


def median(xs: list[float]) -> float:
    ys = sorted(xs)
    return ys[len(ys) // 2]


def peak_rss(fn):
    box = [0]
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            box[0] = max(box[0], rss_bytes())
            stop.wait(0.05)

    sampler = threading.Thread(target=watch, daemon=True)
    sampler.start()
    try:
        out = fn()
    finally:
        stop.set()
        sampler.join()
    return out, box[0]


def once(executor: str, workers: int) -> tuple[float, float, int, float]:
    gc.collect()
    paths = sorted(CORPUS.glob("*.txt"))

    def run() -> tuple[float, float, float]:
        c0 = process_time()
        t0 = perf_counter()
        _index, merge_s, worker_cpu = build_all(paths, workers, executor, WITH_POSITIONS)
        wall = perf_counter() - t0
        # processes: parent clock plus each worker. Threads share one process clock.
        cpu = process_time() - c0
        if executor == "processes":
            cpu += worker_cpu
        return wall, cpu, merge_s

    (wall, cpu, merge_s), rss = peak_rss(run)
    return wall, cpu, rss, merge_s


def plot(rows: list[tuple]) -> None:
    try:
        import matplotlib
    except ImportError:
        # this 3.13t install has no matplotlib wheel
        print("plot skipped", flush=True)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    names: list[str] = []
    xs_all: list[int] = []
    for row in rows:
        if row[0] not in names:
            names.append(row[0])
    for name in names:
        pts = [row for row in rows if row[0] == name]
        xs = [row[1] for row in pts]
        ys = [row[6] for row in pts]
        xs_all.extend(xs)
        ax.plot(xs, ys, marker="o", label=name)
    top = max(xs_all) if xs_all else 1
    ax.plot([1, top], [1, top], linestyle="--", color="gray", label="ideal")
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup")
    ax.legend()
    filename = "speedup_free.png" if not gil_enabled() else "speedup.png"
    path = Path(__file__).resolve().parents[2] / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print("plot", path, flush=True)


shared = 0


def add() -> None:
    global shared
    for _ in range(10**6):
        shared += 1


def race() -> None:
    # each build_partial owns its dicts; this shared += is the one that needs the lock
    global shared
    shared = 0
    threads = [threading.Thread(target=add) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("race", shared, flush=True)  # 2_000_000 only if no update was lost
    shared = 0
    lock = threading.Lock()

    def add_locked() -> None:
        global shared
        for _ in range(10**6):
            with lock:
                shared += 1

    threads = [threading.Thread(target=add_locked) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("locked", shared, flush=True)


def read_one(path: Path) -> int:
    return len(read_text(path))


def io_bench() -> None:
    paths = sorted(CORPUS.glob("*.txt"))
    reps: list[Path] = []
    while len(reps) < 60:
        reps.extend(paths)
    reps = reps[:60]
    t0 = perf_counter()
    for path in reps:
        read_one(path)
    serial_s = perf_counter() - t0
    t0 = perf_counter()
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(read_one, reps))
    thread_s = perf_counter() - t0
    print(f"io serial={serial_s:.4f}s threads={thread_s:.4f}s", flush=True)


def main() -> None:
    if hasattr(sys, "_is_gil_enabled"):
        print("sys._is_gil_enabled()", sys._is_gil_enabled(), flush=True)
    else:
        print("sys._is_gil_enabled() missing; GIL is on", flush=True)
    print("cpu", cpu_model(), "cores", os.cpu_count(), flush=True)
    rows: list[tuple] = []
    serial_wall = 0.0
    for executor in EXECUTORS:
        counts = (1,) if executor == "serial" else COUNTS
        for n in counts:
            once(executor, n)
            samples = [once(executor, n) for _ in range(RUNS)]
            wall = median([s[0] for s in samples])
            cpu = median([s[1] for s in samples])
            rss = median([float(s[2]) for s in samples])
            merge_s = median([s[3] for s in samples])
            if executor == "serial":
                serial_wall = wall
            speedup = serial_wall / wall
            rows.append((executor, n, wall, cpu, rss, merge_s, speedup))
            print(
                f"{executor:10} {n:2}  wall={wall:.3f}  cpu={cpu:.3f}  "
                f"rss={rss / 1048576:.1f}MiB  merge={merge_s:.3f}  speedup={speedup:.2f}",
                flush=True,
            )
    plot(rows)
    race()
    io_bench()


if __name__ == "__main__":
    main()
