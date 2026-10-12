"""Trace original WFMM initialization likelihoods on synthetic inputs (Linux x86-64).

Requires a locally extracted official Linux bundle. Checks the original binary,
uses child-only ptrace, and exports numerical observations, not executable bytes.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import signal
import struct
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

SHA = "0c18cac7225eca3db264b8a4e769a45960cde0bfbf68fc2c1554c83cf451cf8e"
ENTRY = 0x49D960
FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/wfmm-native-initialization.json"
Y = np.array(
    [
        [1, 4, 2, 3],
        [2, 2, 3, 1],
        [4, 3, 1, 5],
        [5, 6, 4, 2],
        [6, 3, 7, 4],
        [3, 1, 2, 6],
        [2, 5, 5, 3],
        [4, 2, 6, 2.0],
    ]
)


class Registers(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "r15 r14 r13 r12 rbp rbx r11 r10 r9 r8 rax rcx rdx rsi rdi orig_rax rip cs "
            "eflags rsp ss fs_base gs_base ds es fs gs"
        ).split()
    ]


def trace(
    binary: Path,
    root: Path,
    model: dict,
    label: str,
    *,
    mode: str = "likelihood",
    coefficients: np.ndarray | None = None,
    mcmc_settings: dict | None = None,
    basis_settings: dict | None = None,
) -> dict:
    y = Y if coefficients is None else coefficients
    entry = {"likelihood": ENTRY, "mom": 0x428360, "eb": 0x494010, "bounds": 0x48AB80}[mode]
    library = ctypes.CDLL(None, use_errno=True)
    library.ptrace.restype = ctypes.c_long

    def ptrace(request, pid, address=0, data=0):
        ctypes.set_errno(0)
        answer = library.ptrace(
            request,
            pid,
            ctypes.c_void_p(address),
            ctypes.c_void_p(data) if isinstance(data, int) else data,
        )
        if answer == -1 and ctypes.get_errno():
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        return answer

    path = root / f"{label}.mat"
    savemat(
        path,
        dict(
            Y=y,
            model=model,
            basis_specs=dict(transformtype="none", nlevels=1.0) | (basis_settings or {}),
            MCMCspecs=dict(
                B=8.0,
                burnin=4.0,
                thin=1.0,
                time_update=1.0,
                nj_nosmooth=0.0,
            )
            | (mcmc_settings or {}),
        ),
    )
    pid = os.fork()
    if pid == 0:
        fd = os.open(root / f"{label}.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        ptrace(0, 0)
        os.execve(
            binary,
            [str(binary), str(path)],
            dict(os.environ, LD_LIBRARY_PATH=str(binary.parent), OMP_NUM_THREADS="1"),
        )
    os.waitpid(pid, 0)

    def read(address, length):
        if not 0 <= length <= 10000:
            raise ValueError("unexpected native matrix size")
        return b"".join(
            struct.pack("<Q", ptrace(2, pid, a) & ((1 << 64) - 1))
            for a in range(address, address + length, 8)
        )[:length]

    def word(address):
        return int.from_bytes(read(address, 8), "little")

    def breakpoint(address):
        old = word(address)
        ptrace(5, pid, address, (old & ~255) | 0xCC)
        return old

    def matrix(address):
        rows, columns = struct.unpack("<ii", read(address + 16, 8))
        values = struct.unpack(
            "<" + "d" * (rows * columns), read(word(address + 8), 8 * rows * columns)
        )
        return np.array(values).reshape(rows, columns).tolist()

    original = breakpoint(entry)
    active = None
    records = []
    exited = False
    try:
        for _ in range(20000):
            ptrace(7, pid)
            _, status = os.waitpid(pid, 0)
            if os.WIFEXITED(status):
                exited = True
                if os.WEXITSTATUS(status):
                    raise RuntimeError((root / f"{label}.log").read_text())
                break
            if not os.WIFSTOPPED(status) or os.WSTOPSIG(status) != signal.SIGTRAP:
                raise RuntimeError(f"unexpected native stop: {status}")
            registers = Registers()
            ptrace(12, pid, 0, ctypes.byref(registers))
            address = registers.rip - 1
            if address == entry:
                if mode == "likelihood":
                    record = dict(
                        theta=matrix(registers.rdi),
                        beta=matrix(registers.rsi),
                        betans=matrix(registers.rdx),
                        coefficient=word(registers.rsp + 8) & 0xFFFFFFFF,
                    )
                elif mode == "bounds":
                    record = dict(theta_MOM=matrix(registers.rsi), beta=matrix(registers.rdx))
                    output_pointers = [word(registers.rsp + offset) for offset in (32, 40, 48)]
                elif mode == "eb":
                    record = dict(
                        Z=matrix(registers.rdi),
                        p=matrix(registers.rsi),
                        T=matrix(registers.rdx),
                        niter=registers.rcx,
                    )
                    output_pointers = [word(registers.rsp + offset) for offset in (8, 16, 24)]
                else:
                    record = dict(A=matrix(registers.rsi), b=matrix(registers.rdx))
                return_address = word(registers.rsp)
                active = (return_address, breakpoint(return_address), record, registers.rdi)
                ptrace(5, pid, entry, original)
                registers.rip = entry
                ptrace(13, pid, 0, ctypes.byref(registers))
                ptrace(9, pid)
                os.waitpid(pid, 0)
                breakpoint(entry)
            elif active and address == active[0]:
                fp = ctypes.create_string_buffer(512)
                ptrace(14, pid, 0, fp)
                record = active[2]
                if mode == "likelihood":
                    record.update(
                        loglik=struct.unpack_from("<d", fp.raw, 160)[0], caller=hex(address)
                    )
                    records.append(record)
                elif mode == "bounds":
                    record.update(
                        zip(("mle", "upper", "lower"), map(matrix, output_pointers), strict=True)
                    )
                    records.append(record)
                elif mode == "eb":
                    record.update(
                        zip(("pout", "Tout", "gam"), map(matrix, output_pointers), strict=True)
                    )
                    record["caller"] = hex(address)
                    records.append(record)
                elif address == 0x488A25:
                    record.update(solution=matrix(active[3]), caller=hex(address))
                    records.append(record)
                ptrace(5, pid, address, active[1])
                registers.rip = address
                ptrace(13, pid, 0, ctypes.byref(registers))
                ptrace(9, pid)
                os.waitpid(pid, 0)
                active = None
            else:
                raise RuntimeError(f"unexpected breakpoint: {address:x}")
        else:
            raise RuntimeError("native trace exceeded bounded instruction-stop count")
    finally:
        if not exited:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except ProcessLookupError:
                pass
    native = loadmat(root / f"{label}_Init.mat")
    return dict(
        case=label,
        y=y.tolist(),
        model={k: v.tolist() for k, v in model.items()},
        traces=records,
        output={
            k: native[k].tolist()
            for k in (
                "omega_MOM",
                "omega_MLE",
                "se_omega",
                "betastar_ns",
                "alpha_MOM",
                "tau_MOM",
                "pi_MOM",
            )
        },
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=FIXTURE)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("reference tracer requires Linux x86-64")
    binary = args.directory.resolve() / "wfmm1"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != SHA:
        raise ValueError("native executable checksum mismatch")
    z = np.repeat(np.eye(4), 2, axis=0)
    cases = [
        ("identity", dict(X=np.ones((8, 1)), C=np.ones((8, 1)))),
        (
            "strata",
            dict(
                X=np.ones((8, 1)),
                C=np.column_stack((np.r_[np.ones(3), np.zeros(5)], np.r_[np.zeros(3), np.ones(5)])),
            ),
        ),
        ("random_intercept", dict(X=np.ones((8, 1)), C=np.ones((8, 1)), Z=z, m=np.array([4.0]))),
        (
            "two_random_levels",
            dict(
                X=np.ones((8, 1)),
                C=np.ones((8, 1)),
                Z=np.column_stack((z, np.repeat(np.eye(2), 4, axis=0))),
                m=np.array([4.0, 2.0]),
            ),
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="wfmm-init-reference-") as directory:
        result = [trace(binary, Path(directory), model, label) for label, model in cases]
    args.output.write_text(
        json.dumps(
            dict(
                executable_sha256=SHA,
                native_version="WFMM1 3.1.0, September14,2015",
                source_entry=hex(ENTRY),
                cases=result,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(len(result), "designs", sum(len(c["traces"]) for c in result), "likelihood evaluations")


if __name__ == "__main__":
    main()
