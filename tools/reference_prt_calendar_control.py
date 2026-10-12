"""Execute native PRT suspension-clock and final-partial-cohort bookkeeping."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_EIP, UC_X86_REG_ESP

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != "52903a7af2e75998bf2a8cbda6e2049e7fa25e0981cf9806317841da13a91b54":
    raise ValueError("original simulation executable hash mismatch")
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase


def run(maximum, enrolled, cohort, candidate_index):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(address, fmt, *values):
        u.mem_write(address, struct.pack("<" + fmt, *values))

    def get(address, fmt):
        return struct.unpack("<" + fmt, u.mem_read(address, struct.calcsize("<" + fmt)))[0]

    put(bp - 0x74, "i", maximum)
    put(bp - 0xD0, "i", enrolled)
    put(bp - 0xD8, "i", cohort)
    put(bp - 0xC8, "i", candidate_index)
    put(bp - 0xE8, "d", 0.125)
    arrivals = [float(i * i + 1) for i in range(20)]
    put(0xCCD648, "d" * len(arrivals), *arrivals)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.emu_start(base + 0x3ACB, base + 0x3B01, count=1000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x3B01:
        raise RuntimeError("suspension instruction budget exhausted")
    result = dict(index=get(bp - 0xC8, "i"), clock=get(bp - 0xE8, "d"))
    u.emu_start(base + 0x4000, base + 0x4045, count=1000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x4045:
        raise RuntimeError("cohort instruction budget exhausted")
    result["cohort"] = get(bp - 0xD8, "i")
    result["enrolled"] = get(bp - 0xD0, "i")
    return dict(
        maximum=maximum,
        enrolled=enrolled,
        cohort=cohort,
        index=candidate_index,
        arrivals=arrivals,
        result=result,
    )


cases = [
    run(n, n - remaining, cohort, index)
    for n in (8, 9, 36)
    for remaining in (0, 1, 2, 3, 4)
    for cohort in (1, 2, 3)
    for index in (0, 4)
]
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="suspension arrival advance and cohort remainder only",
            cases=cases,
        ),
        indent=2,
    )
)
