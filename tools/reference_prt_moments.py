"""Native PRT conduct moment matching/fallback, independent of posterior simulation."""

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_EIP, UC_X86_REG_ESP, UC_X86_REG_FPCW

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == "2943f71ab27e300c1168b2d9f6a93815b75efe753b6af2b543c1370d077a1123"
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase


def run(mean, sd, current, draws=None):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(a, fmt, *v):
        u.mem_write(a, struct.pack("<" + fmt, *v))

    def get(a):
        v = struct.unpack("<d", u.mem_read(a, 8))[0]
        return v if math.isfinite(v) else "nan" if math.isnan(v) else "inf" if v > 0 else "-inf"

    put(bp - 8, "i", len(mean))
    put(bp - 0x34, "i", current)
    put(0x77C8D0, "d" * len(mean), *mean)
    put(0x77DA00, "d" * len(sd), *sd)
    # Next-dose scratch values are initialized by the immediately preceding block.
    for offset in (0x48, 0x50):
        put(bp - offset, "d", 0.0)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x37F)
    if draws is not None:
        N = len(draws)
        put(bp - 0x3C, "i", 0)
        put(bp - 0x40, "i", N)
        put(bp - 0x44, "i", 0)
        put(0x77DC20, "d", float(N))
        put(0x77C8D0, "d" * len(mean), *[sum(row[j] for row in draws) for j in range(len(mean))])
        put(
            0x77C970,
            "d" * len(mean),
            *[float(sum(row[j] > 0.3 for row in draws)) for j in range(len(mean))],
        )
        for j in range(len(mean)):
            put(0x5F5ED0 + j * 0x27100, "d" * N, *[row[j] for row in draws])
    u.emu_start(base + (0x434C if draws is not None else 0x4567), base + 0x4692, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x4692
    return dict(
        mean=[get(0x77C8D0 + 8 * j) for j in range(len(mean))],
        sd=[get(0x77DA00 + 8 * j) for j in range(len(mean))],
        current=current,
        alpha=get(bp - 0x58),
        beta=get(bp - 0x60),
        next_alpha=get(bp - 0x48),
        next_beta=get(bp - 0x50),
        **(
            {"draws": draws, "exceedance": [get(0x77C970 + 8 * j) for j in range(len(mean))]}
            if draws is not None
            else {}
        ),
    )


cases = []
for m in (0.0, 0.2, 0.5, 0.8, 1.0, 1.01):
    for sd in (0.0, 0.05, 0.2, 0.5, 0.7):
        for current in (1, 2):
            cases.append(run([m, 0.4], [sd, 0.1], current))

for D in (2, 3, 6, 10):
    for N in (8, 9, 32):
        draws = [[(1 + (i * 7 + j * 3) % 97) / 100 for j in range(D)] for i in range(N)]
        cases.append(run([0.0] * D, [0.0] * D, 1, draws))
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope=("supplied mean/SD and projected-draw summaries; posterior sampling excluded"),
            cases=cases,
        ),
        indent=2,
    )
)
