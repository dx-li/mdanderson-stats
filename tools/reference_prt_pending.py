"""Native PRT pending-risk accumulation before isotonic projection."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_EIP, UC_X86_REG_ESP, UC_X86_REG_FPCW

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != "2943f71ab27e300c1168b2d9f6a93815b75efe753b6af2b543c1370d077a1123":
    raise ValueError("original conduct executable hash mismatch")
native_image = pefile.PE(str(path))
base = native_image.OPTIONAL_HEADER.ImageBase


def run(coefficients, rows):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (native_image.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, native_image.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000
    C, D = len(coefficients[0]), len(coefficients[0][0])

    def put(a, fmt, *values):
        u.mem_write(a, struct.pack("<" + fmt, *values))

    for i, (dose, interval, censored) in enumerate(rows):
        put(0x46DBD0 + 4 * i, "i", dose)
        put(0x46CF50 + 4 * i, "i", interval)
        put(0x46E850 + 4 * i, "i", censored)
    for sample in coefficients:
        for h, coefs in enumerate(sample):
            put(0x77CD28 + 80 * (h + 1) + 8, "d" * D, *coefs)
        for offset, value in [(0x38, D), (0x3C, 0), (8, len(rows)), (0x40, 0), (0x44, C)]:
            put(bp - offset, "i", value)
        u.reg_write(UC_X86_REG_EBP, bp)
        u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
        u.reg_write(UC_X86_REG_FPCW, 0x23F)
        u.emu_start(base + 0x3493, base + 0x35D7, count=1000000)
        if u.reg_read(UC_X86_REG_EIP) != base + 0x35D7:
            raise RuntimeError("native pending-risk instruction budget exhausted")
    result = []
    for k in range(1, D + 1):
        count = sum(d == k and z == 1 and h < C for d, h, z in rows)
        values = (
            struct.unpack("<" + "d" * count, u.mem_read(0x77C490 + 6400 * k, 8 * count))
            if count
            else ()
        )
        result.append([v / len(coefficients) for v in values])
    return dict(coefficients=coefficients, rows=rows, mean_pending_risks=result)


cases = []
for C in (3, 6):
    for D in (2, 3, 6):
        rows = [[k, h, 1] for k in range(1, D + 1) for h in (0, 1, C - 1, C)]
        rows += [[k, 1, 0] for k in range(1, D + 1)]
        for shift in (-2.0, 0.0, 2.0):
            coefficients = [
                [[shift + 0.1 * h - 0.4 * k + 0.2 * j for k in range(D)] for h in range(C)]
                for j in range(2)
            ]
            cases.append(run(coefficients, rows))
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="raw pending-risk average before total-risk isotonic projection",
            cases=cases,
        ),
        indent=2,
    )
)
