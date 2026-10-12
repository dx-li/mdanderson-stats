"""Native PRT conduct decision-code boundaries from supplied posterior criteria."""

import hashlib
import itertools
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


def run(xi, current, assigned, assessed, pn, pe, next_pe):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (native_image.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, native_image.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(a, fmt, *values):
        u.mem_write(a, struct.pack("<" + fmt, *values))

    put(bp - 0x34, "i", current)
    put(bp - 0x40, "i", 3 if xi[current - 1] < 0.3 else 2)
    put(bp - 0x88, "d", xi[current - 1])
    put(bp - 0x80, "d", 6.4 if pn < 0.05 else 6.3 if next_pe > 0.05 else 6.2)
    put(bp - 0x68, "d", pn)
    put(bp - 0x70, "d", next_pe)
    put(bp - 0x78, "d", pe)
    put(0x77DB40, "d", pe)
    put(0x77DB60, "d", 0.9)
    put(0x77DB68, "d", 0.3)
    put(0x77DC48, "d", 0.05)
    put(0x77C970, "d" * len(xi), *xi)
    put(0x77DA50, "d" * len(assigned), *assigned)
    put(0x46CEB0, "d" * len(assessed), *assessed)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x23F)
    u.emu_start(base + 0x4FC8, base + 0x5552, count=10000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x5552:
        raise RuntimeError("native controller instruction budget exhausted")
    code = struct.unpack("<d", u.mem_read(bp - 0x80, 8))[0]
    return dict(
        exceedance=xi,
        current=current,
        assigned=assigned,
        assessed=assessed,
        pn=pn,
        pe=pe,
        next_pe=next_pe,
        code=code,
    )


cases = []
# Moderate/excessive categories: native predictive calculation is bypassed;
# supplied PE is the controller input, not a parity claim for its estimator.
for xi, current, pending, pe in itertools.product(
    ([0.1, 0.3, 0.5], [0.1, 0.5, 0.9], [0.1, 0.9, 0.95], [0.91, 0.95, 1.0]),
    (1, 2, 3),
    (0, 3),
    (0.0, 0.05, 0.051, 0.949, 0.95, 0.951, 1.0),
):
    if xi[current - 1] < 0.3:
        continue
    assigned = [6.0] * 3
    assessed = assigned.copy()
    assessed[current - 1] -= pending
    cases.append(run(xi, current, assigned, assessed, 0.5, pe, 0.02))
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="acceptable/excessive native conduct code from supplied criteria",
            cases=cases,
        ),
        indent=2,
    )
)
