"""Read actual WFMM2 proposal SDs before its first variance update.

Only synthetic inputs and numeric references are exported. The native children
are stopped after reading initialized proposal arrays; no random-stream claim.
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
from reference_wfmm_initialization import SHA, Registers, Y
from scipy.io import loadmat, savemat

SHA2 = "4db3d8724fe80e5450b6b666b67de2967d625f421c5e05a9cc0d31049e3a3f0e"
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "tests/fixtures/wfmm-native-proposal.json",
)
args = parser.parse_args()
source = args.directory.resolve()
for name, digest in (("wfmm1", SHA), ("wfmm2", SHA2)):
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
        raise ValueError("native checksum mismatch")
library = ctypes.CDLL(None, use_errno=True)
library.ptrace.restype = ctypes.c_long


def ptrace(request, pid, address=0, data=0):
    ctypes.set_errno(0)
    result = library.ptrace(
        request,
        pid,
        ctypes.c_void_p(address),
        ctypes.c_void_p(data) if isinstance(data, int) else data,
    )
    if result == -1 and ctypes.get_errno():
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    return result


cases = []
with tempfile.TemporaryDirectory(prefix="wfmm-proposal-reference-") as temporary:
    root = Path(temporary)
    environment = dict(os.environ, LD_LIBRARY_PATH=str(source), OMP_NUM_THREADS="1")
    for scale in (0.5, 1.0, 1.5, 4.0):
        path = root / "input.mat"
        savemat(
            path,
            dict(
                Y=np.tile(Y, (1, 4)),
                model=dict(
                    X=np.ones((8, 1)),
                    C=np.ones((8, 1)),
                    Z=np.repeat(np.eye(4), 2, axis=0),
                    m=np.array([4.0]),
                ),
                basis_specs=dict(
                    transformtype="wavelet",
                    wavelet="db1",
                    boundary="periodic",
                    nlevels=2.0,
                    extended_mode=1.0,
                ),
                MCMCspecs=dict(
                    B=8.0,
                    burnin=4.0,
                    thin=1.0,
                    time_update=1.0,
                    nj_nosmooth=0.0,
                    propvar_omega=scale,
                ),
            ),
        )
        subprocess.run(
            [str(source / "wfmm1"), str(path)],
            env=environment,
            capture_output=True,
            check=True,
            timeout=20,
        )
        path = root / "input_Init.mat"
        native = loadmat(path)
        pid = os.fork()
        if pid == 0:
            fd = os.open(root / "native.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            ptrace(0, 0)
            os.execve(
                source / "wfmm2", [str(source / "wfmm2"), str(path), str(root / "out")], environment
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

        try:
            address = 0x4936A0
            old = word(address)
            ptrace(5, pid, address, (old & ~255) | 0xCC)
            ptrace(7, pid)
            _, status = os.waitpid(pid, 0)
            if not os.WIFSTOPPED(status) or os.WSTOPSIG(status) != signal.SIGTRAP:
                raise RuntimeError((root / "native.log").read_text())
            registers = Registers()
            ptrace(12, pid, 0, ctypes.byref(registers))
            if registers.rip - 1 != address:
                raise RuntimeError("unexpected native stop")
            pointer = word(registers.rsp + 56)
            rows, columns = struct.unpack("<ii", read(pointer + 16, 8))
            sd = np.frombuffer(read(word(pointer + 8), 8 * rows * columns), dtype="<f8")
            cases.append(
                dict(
                    propvar_omega=scale,
                    se_omega=native["se_omega"].tolist(),
                    proposal_sd=sd.reshape(rows, columns).tolist(),
                )
            )
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except ProcessLookupError:
                pass
args.output.write_text(
    json.dumps(dict(wfmm1_sha256=SHA, wfmm2_sha256=SHA2, entry="0x4936a0", cases=cases), indent=2)
    + "\n"
)
print(len(cases), "original proposal matrices")
