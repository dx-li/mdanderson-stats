"""Native PRT1.0 conduct counts from supplied patient-file triples."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_ESP, UC_X86_REG_FPCW

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != "2943f71ab27e300c1168b2d9f6a93815b75efe753b6af2b543c1370d077a1123":
    raise ValueError("original conduct executable hash mismatch")
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase


def run(rows, intervals, doses):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(address, fmt, *values):
        u.mem_write(address, struct.pack("<" + fmt, *values))

    put(bp - 8, "i", len(rows))
    put(0x77DC8C, "i", intervals)
    for i, (dose, observed, censored) in enumerate(rows):
        put(0x46DBD0 + 4 * i, "i", dose)
        put(0x46CF50 + 4 * i, "i", observed)
        put(0x46E850 + 4 * i, "i", censored)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x37F)
    u.emu_start(base + 0x223D, base + 0x2441, count=100000)
    return {
        name: struct.unpack("<" + "d" * doses, u.mem_read(address, doses * 8))
        for name, address in [("assigned", 0x77DA50), ("assessed", 0x46CEB0), ("events", 0x46CF00)]
    }


cases = []
for C, D in ((3, 2), (6, 6), (10, 10)):
    cases.append(dict(intervals=C, doses=D, rows=[], result=run([], C, D)))
    rows = []
    for dose in (1, D):
        for censored in (0, 1):
            for observed in range(1 - censored, C + 1):
                row = [dose, observed, censored]
                cases.append(dict(intervals=C, doses=D, rows=[row], result=run([row], C, D)))
                rows.append(row)
    cases.append(dict(intervals=C, doses=D, rows=rows, result=run(rows, C, D)))
print(
    json.dumps(
        dict(executable_sha256=digest, scope="native assigned/assessed/event counts", cases=cases),
        indent=2,
    )
)
