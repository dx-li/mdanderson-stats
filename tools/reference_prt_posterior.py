"""Native PRT1.0 latent-Gibbs posterior pilot from synthetic patient-file rows.

Runs213b–434c: native fixed seeds,10000iterations/5000burn-in and total-risk
retention, covariance projection and lower safeguard. Captures both raw moments
and projected validity diagnostics. This is one native chain;
Monte Carlo error is reported and exact Python RNG/sample identity is not claimed.
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


def run(rows, intervals, doses, prior_mean, prior_variance):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(address, fmt, *values):
        u.mem_write(address, struct.pack("<" + fmt, *values))

    def get(address, fmt):
        return struct.unpack("<" + fmt, u.mem_read(address, struct.calcsize("<" + fmt)))

    put(bp - 8, "i", len(rows))
    put(0x77DC8C, "i", intervals)
    put(0x77DC90, "i", doses)
    put(0x77DAA0, "d" * 10, *([prior_mean] * 10))
    put(0x77DAF0, "d" * 10, *([prior_variance] * 10))
    for i, (dose, interval, censored) in enumerate(rows):
        put(0x46DBD0 + 4 * i, "i", dose)
        put(0x46CF50 + 4 * i, "i", interval)
        put(0x46E850 + 4 * i, "i", censored)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x37F)
    u.emu_start(base + 0x213B, base + 0x35F0, count=500_000_000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x35F0 or get(0x77DC20, "d")[0] != 5000:
        raise RuntimeError("native posterior instruction/retention budget exhausted")
    samples = np.array([get(0x46F4D0 + j * 0x27100, "d" * 5000) for j in range(doses)]).T
    assert np.all(np.isfinite(samples)) and np.all((samples >= 0) & (samples <= 1))
    u.emu_start(base + 0x35F0, base + 0x434C, count=500_000_000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x434C:
        raise RuntimeError("native projection instruction budget exhausted")
    projected = np.array([get(0x5F5ED0 + j * 0x27100, "d" * 5000) for j in range(doses)]).T
    return dict(
        projected_minimum=float(projected.min()),
        projected_maximum=float(projected.max()),
        projected_above_one=int(np.count_nonzero(projected > 1)),
        projected_zero=int(np.count_nonzero(projected == 0)),
        rows=rows,
        intervals=intervals,
        doses=doses,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        draws=5000,
        burn_in=5000,
        mean=samples.mean(0).tolist(),
        variance=samples.var(0).tolist(),
        batch_mcse=(samples.reshape(50, 100, doses).mean(1).std(0, ddof=1) / np.sqrt(50)).tolist(),
        samples_sha256=hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
    )


cases = [
    run([[1, 1, 1], [1, 2, 0], [2, 2, 1]], 3, 2, -0.5, 1.2),
    run([[1, 0, 1], [1, 3, 1], [2, 1, 0], [2, 2, 1]], 3, 2, 0.2, 0.8),
    run([[2, 6, 1], [2, 4, 1], [2, 3, 1], [3, 3, 1], [3, 2, 1], [3, 1, 1]], 6, 6, -14.0, 28.0),
]
print(
    json.dumps(
        dict(executable_sha256=digest, scope="native pre-projection posterior pilots", cases=cases),
        indent=2,
        allow_nan=False,
    )
)
