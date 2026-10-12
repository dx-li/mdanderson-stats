"""Native x87 event-observation comparisons around binary64 clock boundaries."""

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
if digest != "52903a7af2e75998bf2a8cbda6e2049e7fa25e0981cf9806317841da13a91b54":
    raise ValueError("source hash mismatch")
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase


def run(arrival, delay, clock):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(address, fmt, value):
        u.mem_write(address, struct.pack("<" + fmt, value))

    put(0x78BF78, "i", 1)
    put(0x477040, "d", arrival)
    put(0x47A240, "d", delay)
    put(bp - 0x48, "d", 100.0)
    put(bp - 0xE8, "d", clock)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x23F)
    u.emu_start(base + 0x3CAE, base + 0x3D69, count=1000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x3D69:
        raise RuntimeError("instruction budget")
    return dict(
        arrival=arrival,
        delay=delay,
        clock=clock,
        censored=struct.unpack("<i", u.mem_read(0x473E40, 4))[0],
        observed_time=struct.unpack("<d", u.mem_read(0x478940, 8))[0],
    )


cases = []
for a in (0.0, 0.1, 1.0, 7.0, 1e6, 1e8):
    for t in (0.0, 2**-65, 2**-54, 2**-20, 0.1, 1.0, 10.0):
        for clock in (math.nextafter(a + t, -math.inf), a + t, math.nextafter(a + t, math.inf)):
            if clock >= a:
                cases.append(run(a, t, clock))
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="native x87 as-of event comparison; finite nonnegative inputs, FPCW0x23f",
            cases=cases,
        ),
        indent=2,
    )
)
