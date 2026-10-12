"""Native PRT simulation final choice from supplied posterior summaries/counts."""

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
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase


def run(assessed, exceedance, distance, current):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(a, fmt, *values):
        u.mem_write(a, struct.pack("<" + fmt, *values))

    def get(a, fmt):
        return struct.unpack("<" + fmt, u.mem_read(a, struct.calcsize("<" + fmt)))[0]

    for offset, value in [(0x74, 36), (0xE0, 36), (0xC0, 0), (0xB8, current)]:
        put(bp - offset, "i", value)
    put(0x78BF4C, "i", len(assessed))
    put(0x78BD98, "d", 0.9)
    for address, values in [(0x47BC88, assessed), (0x47CC28, exceedance), (0x47CB88, distance)]:
        put(address, "d" * len(values), *values)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x23F)
    u.emu_start(base + 0x97E1, base + 0x98FE, count=10000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x98FE:
        raise RuntimeError("native final instruction budget exhausted")
    return dict(
        assessed=assessed,
        exceedance=exceedance,
        distance=distance,
        current=current,
        stopped=get(bp - 0xC0, "i"),
        selected=get(bp - 0xB8, "i"),
    )


cases = []
for a, xi, distances, current in itertools.product(
    ([0.0, 0.0, 0.0], [2.0, 3.0, 4.0], [3.0, 3.0, 3.0], [4.0, 2.0, 3.0]),
    ([0.1, 0.5, 0.9], [0.9, 0.9, 0.91], [0.91, 0.95, 1.0]),
    ([0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.2, 0.2], [0.1, 0.1, 0.2]),
    (1, 2, 3),
):
    cases.append(run(a, xi, distances, current))
print(
    json.dumps(
        dict(executable_sha256=digest, scope="final summary selector only", cases=cases), indent=2
    )
)
