"""Audit native beta/inverse-gamma conversion with controlled numeric inputs.

Only our own checksum-pinned synthetic WFMM2 child is traced. Center values are
supplied at the function entry; no original executable or source bytes are saved.
This records behavior without accepting its implied prior moments.
"""

import argparse
import ctypes
import hashlib
import json
import os
import signal
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from reference_wfmm_spatial import Regs, ptrace
from scipy.io import savemat
from scipy.optimize import brentq

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
native = args.directory.resolve()
hashes = {
    "wfmm1": "0c18cac7225eca3db264b8a4e769a45960cde0bfbf68fc2c1554c83cf451cf8e",
    "wfmm2": "4db3d8724fe80e5450b6b666b67de2967d625f421c5e05a9cc0d31049e3a3f0e",
}
for name, sha in hashes.items():
    if hashlib.sha256((native / name).read_bytes()).hexdigest() != sha:
        raise ValueError("original executable checksum mismatch")
temporary = tempfile.TemporaryDirectory(prefix="wfmm-hyperprior-audit-")
root = Path(temporary.name)


def capture_prior(path, forced_mean, entry):
    pid = os.fork()
    if pid == 0:
        ptrace(0, 0)
        f = os.open(root / "trace.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(f, 1)
        os.dup2(f, 2)
        os.execve(
            native / "wfmm2",
            [str(native / "wfmm2"), str(path), str(root / "draws")],
            dict(os.environ, LD_LIBRARY_PATH=str(native), OMP_NUM_THREADS="1"),
        )

    def timeout(signum, frame):
        raise TimeoutError("native prior capture exceeded 30 seconds")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(30)
    try:
        os.waitpid(pid, 0)
        word = ptrace(1, pid, entry)
        ptrace(4, pid, entry, (word & ~255) | 0xCC)
        ptrace(7, pid)

        def wait(pc):
            while True:
                _, status = os.waitpid(pid, 0)
                if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                    raise RuntimeError("source exited before capture")
                regs = Regs()
                ptrace(12, pid, 0, ctypes.byref(regs))
                if regs.rip == pc + 1:
                    return regs
                ptrace(
                    7, pid, 0, 0 if os.WSTOPSIG(status) == signal.SIGTRAP else os.WSTOPSIG(status)
                )

        regs = wait(entry)
        mu, ap, bp = regs.rdi, regs.rsi, regs.rdx
        with open(f"/proc/{pid}/mem", "rb", buffering=0) as memory:

            def read(addr, n):
                memory.seek(addr)
                return memory.read(n)

            def matrix(addr):
                head = read(addr, 24)
                data = struct.unpack_from("<Q", head, 8)[0]
                rows, cols = struct.unpack_from("<ii", head, 16)
                assert 0 < rows * cols < 10000
                return (
                    np.frombuffer(read(data, rows * cols * 8), "<f8").reshape(rows, cols).tolist()
                )

            head = read(mu, 24)
            data = struct.unpack_from("<Q", head, 8)[0]
            rows, cols = struct.unpack_from("<ii", head, 16)
            bits = struct.unpack("<Q", struct.pack("<d", forced_mean))[0]
            for index in range(rows * cols):
                ptrace(4, pid, data + 8 * index, bits)
            initial = matrix(mu)
            fp = ctypes.create_string_buffer(512)
            ptrace(14, pid, 0, ctypes.byref(fp))
            v = struct.unpack_from("<d", fp.raw, 160)[0]
            ret = struct.unpack("<Q", read(regs.rsp, 8))[0]
            ptrace(4, pid, entry, word)
            regs.rip = entry
            ptrace(13, pid, 0, ctypes.byref(regs))
            rw = ptrace(1, pid, ret)
            ptrace(4, pid, ret, (rw & ~255) | 0xCC)
            ptrace(7, pid)
            wait(ret)
            return dict(
                initial=initial, variance=v, mean_after=matrix(mu), a=matrix(ap), b=matrix(bp)
            )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except ProcessLookupError:
            pass


records = []
for family, centers, variances, entry in [
    ("beta", [0.01, 0.2, 0.5, 0.8, 0.99], [0.01, 0.06], 0x42B6E0),
    ("inverse_gamma", [0.1, 1.0, 10.0], [1.0, 1000.0], 0x42BE30),
]:
    for center in centers:
        for variance in variances:
            input_path = root / "input.mat"
            savemat(
                input_path,
                dict(
                    Y=np.random.default_rng(117).normal(2, 1, size=(8, 16)),
                    model=dict(X=np.ones((8, 1)), C=np.ones((8, 1))),
                    basis_specs=dict(
                        transformtype="wavelet",
                        wavelet="db1",
                        nlevels=2.0,
                        boundary="periodic",
                        extended_mode=1.0,
                    ),
                    MCMCspecs=dict(
                        nj_nosmooth=0.0,
                        update_pi_tau=1.0,
                        pi_prior_var=variance if family == "beta" else 0.01,
                        tau_prior_var=variance if family == "inverse_gamma" else 1000.0,
                        B=8.0,
                        burnin=0.0,
                        thin=1.0,
                        time_update=1.0,
                    ),
                ),
            )
            proc = subprocess.run(
                [str(native / "wfmm1"), str(input_path)],
                env=dict(os.environ, LD_LIBRARY_PATH=str(native), OMP_NUM_THREADS="1"),
                capture_output=True,
                timeout=30,
            )
            if proc.returncode:
                raise RuntimeError((proc.stdout + proc.stderr).decode()[-500:])
            r = capture_prior(root / "input_Init.mat", center, entry)
            a, b = r["a"][0][0], r["b"][0][0]
            if family == "beta":
                r["implied_moments"] = dict(
                    mean=a / (a + b),
                    mode=(a - 1) / (a + b - 2),
                    variance=a * b / (a + b) ** 2 / (a + b + 1),
                )
                concentration = brentq(
                    lambda c: (
                        (1 + center * c) * (1 + (1 - center) * c) / (2 + c) ** 2 / (3 + c)
                        - variance
                    ),
                    0,
                    1000,
                    xtol=5e-15,
                    rtol=1e-15,
                )
                proper_a, proper_b = 1 + center * concentration, 1 + (1 - center) * concentration
            else:
                r["implied_moments"] = dict(
                    mean=b / (a - 1) if a > 1 else None,
                    mode=b / (a + 1),
                    variance=b * b / (a - 1) ** 2 / (a - 2) if a > 2 else None,
                    finite_variance=a > 2,
                )
                proper_a = brentq(
                    lambda a: center**2 * (a + 1) ** 2 / (a - 1) ** 2 / (a - 2) - variance,
                    2 + 1e-12,
                    1e6,
                    xtol=5e-15,
                    rtol=1e-15,
                )
                proper_b = center * (proper_a + 1)
            r["independent_mode_centered_reference"] = dict(
                a=proper_a,
                b=proper_b,
                interpretation=(
                    "Mathematical mode/variance solution, not a claim of native "
                    "implementation or intended centering"
                ),
            )
            records.append(
                dict(family=family, requested_center=center, requested_variance=variance, **r)
            )
args.output.write_text(
    json.dumps(
        dict(
            executable_sha256=hashes,
            method=(
                "Own child stopped at native prior conversion entry; numerical centering matrix "
                "supplied in memory, outputs captured at return; no posterior/RNG acceptance claim"
            ),
            cases=records,
        ),
        indent=2,
    )
    + "\n"
)
temporary.cleanup()
print(len(records), "native prior conversion probes")
