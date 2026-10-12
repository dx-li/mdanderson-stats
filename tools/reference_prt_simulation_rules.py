"""Native PRT simulation late allocation from supplied posterior criteria."""

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
if digest != "52903a7af2e75998bf2a8cbda6e2049e7fa25e0981cf9806317841da13a91b54":
    raise ValueError("original simulation executable hash mismatch")
pe_image = pefile.PE(str(path))
base = pe_image.OPTIONAL_HEADER.ImageBase


def run(xi, current, pending, pn, pe, next_pe, distances=None):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe_image.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe_image.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000
    D = len(xi)

    def put(a, fmt, *values):
        u.mem_write(a, struct.pack("<" + fmt, *values))

    def get(a):
        return struct.unpack("<i", u.mem_read(a, 4))[0]

    assigned = [6.0] * D
    assessed = [6.0 - pending] * D
    for offset, value in [
        (0xB8, current),
        (0xB0, 1),
        (0xC0, 0),
        (0xD4, 0),
        (0xEC, D),
        (0xF0, int(current < D)),
        (0x100, 0x47BC88 + 8 * current),
        (0x114, 0x47C958 + 8 * current),
    ]:
        put(bp - offset, "i", value)
    for offset, value in [
        (0x110, 6.0 - pending),
        (0x120, 6.0),
        (0x130, xi[current - 1]),
        (0xF8, pe),
    ]:
        put(bp - offset, "d", value)
    put(0x78BF80, "i", 3 if xi[current - 1] < 0.3 else 2)
    put(0x78BF64, "i", current)
    put(0x78BD98, "d", 0.9)
    put(0x78BEE0, "d", 0.05)
    put(0x78BD78, "d", pe)
    put(0x78BD88, "d", pn)
    put(0x78BD80, "d", next_pe)
    put(0x47BC88, "d" * D, *assessed)
    put(0x47C958, "d" * D, *assigned)
    put(0x47CC28, "d" * D, *xi)
    distances = [0.25 - 0.01 * k for k in range(D)] if distances is None else distances
    put(0x47CB88, "d" * D, *distances)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x23F)
    u.emu_start(base + 0x8FF1, base + 0x97E1, count=100000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x97E1:
        raise RuntimeError("native simulation controller budget exhausted")
    return dict(
        exceedance=xi,
        current=current,
        pending=pending,
        pn=pn,
        pe=pe,
        next_pe=next_pe,
        distance=distances,
        selected=get(bp - 0xB8),
        suspend=get(bp - 0xD4),
        stopped=get(bp - 0xC0),
    )


cases = []
for D, current_choice, pending, pn, pe in itertools.product(
    (3, 10), (0, 1, 2), (0, 3), (0.04, 0.05, 0.94, 0.95), (0.04, 0.05, 0.94, 0.95, 0.96)
):
    current = (1, D - 1, D)[current_choice]
    for xi in ([0.1] * D, [0.5] * D, [0.1] * (D - 1) + [0.95], [0.95] * D):
        cases.append(run(xi, current, pending, pn, pe, pe))
for distance in ([0.05, 0.2, 0.15, 0.3], [0.1, 0.1, 0.1, 0.2], [0.2, 0.1, 0.15, 0.3]):
    cases.append(run([0.1, 0.2, 0.4, 0.95], 4, 0, 0.95, 0.01, 0.01, distance))

print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope=(
                "late simulation allocation only; pre-inference screening "
                "and predictive estimation excluded"
            ),
            cases=cases,
        ),
        indent=2,
    )
)
