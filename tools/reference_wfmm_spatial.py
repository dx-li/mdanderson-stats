"""Capture synthetic native spatial analysis before failing initialization.

Linux x86_64 only. Checksum-pinned WFMM1 is an external research dependency.
A breakpoint in our own child at ANOVA_MOM entry reads its transformed synthetic
identity input; the child is then terminated. No fitting, inverse, native report
or RNG compatibility follows. No original binary bytes enter this fixture.
"""

import argparse
import ctypes
import hashlib
import json
import os
import signal
import struct
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import savemat

lib = ctypes.CDLL(None, use_errno=True)
lib.ptrace.restype = ctypes.c_long


class Regs(ctypes.Structure):
    _fields_ = [
        (s, ctypes.c_ulonglong)
        for s in (
            "r15 r14 r13 r12 rbp rbx r11 r10 r9 r8 rax rcx rdx rsi rdi "
            "orig_rax rip cs eflags rsp ss fs_base gs_base ds es fs gs"
        ).split()
    ]


def ptrace(req, pid, addr=0, data=0):
    ctypes.set_errno(0)
    ans = lib.ptrace(
        req, pid, ctypes.c_void_p(addr), ctypes.c_void_p(data) if isinstance(data, int) else data
    )
    if ans == -1 and ctypes.get_errno():
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    return ans


def capture(path, log, native):
    pid = os.fork()
    if pid == 0:
        ptrace(0, 0)
        f = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(f, 1)
        os.dup2(f, 2)
        os.execve(
            native / "wfmm1",
            [str(native / "wfmm1"), str(path)],
            dict(os.environ, LD_LIBRARY_PATH=str(native), OMP_NUM_THREADS="1"),
        )

    def timeout(signum, frame):
        raise TimeoutError("native transform capture exceeded 30 seconds")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(30)
    try:
        os.waitpid(pid, 0)
        entry = 0x4870E0
        word = ptrace(1, pid, entry)
        ptrace(4, pid, entry, (word & ~255) | 0xCC)
        ptrace(7, pid)
        while True:
            _, status = os.waitpid(pid, 0)
            if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                raise RuntimeError("source exited before capture")
            regs = Regs()
            ptrace(12, pid, 0, ctypes.byref(regs))
            if regs.rip == entry + 1:
                break
            ptrace(7, pid, 0, 0 if os.WSTOPSIG(status) == signal.SIGTRAP else os.WSTOPSIG(status))
        with open(f"/proc/{pid}/mem", "rb", buffering=0) as memory:
            memory.seek(regs.rdi)
            head = memory.read(24)
            data = struct.unpack_from("<Q", head, 8)[0]
            rows, cols = struct.unpack_from("<ii", head, 16)
            assert 1 <= rows <= 512 and 1 <= cols <= 4096
            memory.seek(data)
            return (
                np.frombuffer(memory.read(rows * cols * 8), dtype="<f8").reshape(rows, cols).copy()
            )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = args.directory.resolve()
    sha = "0c18cac7225eca3db264b8a4e769a45960cde0bfbf68fc2c1554c83cf451cf8e"
    if hashlib.sha256((native / "wfmm1").read_bytes()).hexdigest() != sha:
        raise ValueError("original executable checksum mismatch")
    temporary = tempfile.TemporaryDirectory(prefix="wfmm-spatial-reference-")
    root = Path(temporary.name)
    records = []
    settings = []
    for dims, levels in [((8, 4), (1, 1)), ((7, 5), (2, 1)), ((8, 8), (2, 2))]:
        for pair in [("db1", "db1"), ("db2", "coif1")]:
            for boundary in ["periodic", "zero", "reflection"]:
                settings.append(
                    dict(
                        dims=dims,
                        levels=levels,
                        wavelets=pair,
                        boundaries=(boundary,) * 2,
                        rectangular=True,
                    )
                )
    for dims, j in [((4, 4), 1), ((7, 5), 2), ((8, 8), 2), ((4, 4, 4), 1), ((5, 3, 3), 1)]:
        for wavelet in ["db1", "db2"]:
            for boundary in ["periodic", "zero", "reflection"]:
                settings.append(
                    dict(
                        dims=dims,
                        levels=(j,) * len(dims),
                        wavelets=(wavelet,) * len(dims),
                        boundaries=(boundary,) * len(dims),
                        rectangular=False,
                    )
                )
    for boundary in ["periodic", "zero", "reflection"]:
        settings.append(
            dict(
                dims=(5, 7),
                levels=(1, 2),
                wavelets=("coif1", "db2"),
                boundaries=(boundary, "zero"),
                rectangular=True,
            )
        )
    settings.append(
        dict(
            dims=(8, 8, 8),
            levels=(2, 2, 2),
            wavelets=("db1",) * 3,
            boundaries=("periodic",) * 3,
            rectangular=False,
        )
    )
    for spec in settings:
        dims = spec["dims"]
        n = int(np.prod(dims))
        p = root / "capture.mat"
        b = dict(
            transformtype="wavelet",
            dims=np.array(dims, dtype=float),
            wavelet=spec["wavelets"][0],
            nlevels=float(spec["levels"][0]),
            boundary=spec["boundaries"][0],
            extended_mode=1.0,
            rectangular=float(spec["rectangular"]),
        )
        if spec["rectangular"]:
            b.update(
                wavelet2=spec["wavelets"][1],
                nlevels2=float(spec["levels"][1]),
                boundary2=spec["boundaries"][1],
            )
        savemat(
            p,
            dict(
                Y=np.eye(n),
                model=dict(X=np.ones((n, 1)), C=np.ones((n, 1))),
                basis_specs=b,
                MCMCspecs=dict(nj_nosmooth=0.0, B=128.0, burnin=0.0, thin=1.0, time_update=1.0),
            ),
        )
        a = capture(p, str(p) + ".log", native)
        records.append(dict(**spec, analysis=a.tolist()))
        print(spec, a.shape, flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                executable_sha256=sha,
                capture=(
                    "ANOVA_MOM entry before initialization; analysis only; "
                    "no end-to-end native success claimed"
                ),
                cases=records,
            ),
            separators=(",", ":"),
        )
        + "\n"
    )

    temporary.cleanup()


if __name__ == "__main__":
    main()
