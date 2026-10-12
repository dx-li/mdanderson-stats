"""Probe PRT1.0 conduct covariance/min-max transform and native lower safeguard.

Executes RVA36fb–434c, including native inversion961d, from supplied bounded
posterior-risk draws. No posterior sampling, native RNG or trial parity claimed.
Only synthetic inputs/results are redistributed; the hash-pinned executable
must be supplied locally. Requires pefile, unicorn and NumPy.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_EIP, UC_X86_REG_ESP, UC_X86_REG_FPCW

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != "2943f71ab27e300c1168b2d9f6a93815b75efe753b6af2b543c1370d077a1123":
    raise ValueError("original conduct executable hash mismatch")
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase


def run(x):
    n, D = x.shape
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x23F)

    def put(address, fmt, *values):
        u.mem_write(address, struct.pack("<" + fmt, *values))

    put(bp - 0x38, "i", D)
    put(0x77DC90, "i", D)
    put(0x77DC20, "d", n)
    put(0x77CA10, "d" * D, *x.mean(0))
    for j in range(D):
        put(0x46F4D0 + j * 0x27100, "d" * n, *x[:, j])
    u.emu_start(base + 0x36FB, base + 0x434C, count=100_000_000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x434C:
        raise RuntimeError(f"native projection instruction budget exhausted for {D} doses")
    return np.array(
        [struct.unpack("<" + "d" * n, u.mem_read(0x5F5ED0 + j * 0x27100, n * 8)) for j in range(D)]
    ).T


cases = []
for D in (2, 3, 4, 6, 10):
    for seed in range(4):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 1, (32, D))
        cases.append(dict(kind="independent", seed=seed, input=x.tolist(), result=run(x).tolist()))
for kind in ("negative", "above_one", "correlated"):
    for seed in range(4):
        rng = np.random.default_rng(seed)
        t = rng.uniform(0.4, 0.95, 32)
        if kind == "negative":
            x = np.column_stack((t, 0.5 * t - 0.19 + rng.normal(0, 0.002, 32)))
        elif kind == "above_one":
            x = np.column_stack((0.5 * t + 0.5 + rng.normal(0, 0.002, 32), t))
        else:
            x = np.column_stack(
                (
                    0.5 * t + 0.5 + rng.normal(0, 0.002, 32),
                    t,
                    0.5 * t - 0.19 + rng.normal(0, 0.002, 32),
                )
            )
        assert np.all((x >= 0) & (x <= 1))
        cases.append(dict(kind=kind, seed=seed, input=x.tolist(), result=run(x).tolist()))
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="native covariance projection from synthetic posterior draws",
            cases=cases,
        ),
        indent=2,
    )
)
