"""Native PRT simulation pre-inference screening from supplied count summaries."""

import hashlib
import itertools
import json
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_EIP, UC_X86_REG_ESP, UC_X86_REG_FPCW

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != "52903a7af2e75998bf2a8cbda6e2049e7fa25e0981cf9806317841da13a91b54":
    raise ValueError("original simulation executable hash mismatch")
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase


def run(assigned, assessed, observed, first_interval, current, final):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(a, fmt, *values):
        u.mem_write(a, struct.pack("<" + fmt, *values))

    counts = [3.0, 3.0]
    counts[current - 1] = assigned
    for offset, value in [
        (0xB8, current),
        (0xD4, 0),
        (0xE0, int(sum(counts))),
        (0x74, int(sum(counts)) + (0 if final else 3)),
    ]:
        put(bp - offset, "i", value)
    put(0x47C958, "dd", *counts)
    put(0x47BC88 + 8 * (current - 1), "d", assessed)
    put(0x47BC38 + 8 * (current - 1), "d", observed)
    put(0x450230, "i", first_interval)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x37F)
    u.hook_add(
        UC_HOOK_CODE,
        lambda machine, a, size, data: machine.emu_stop(),
        begin=base + 0x97E1,
        end=base + 0x97E1,
    )
    u.emu_start(base + 0x4674, base + 0x47FE, count=10000)
    end = u.reg_read(UC_X86_REG_EIP) - base
    if end not in (0x97E1, 0x47FE):
        raise RuntimeError("native screening instruction budget exhausted")
    suspend = struct.unpack("<i", u.mem_read(bp - 0xD4, 4))[0]
    return dict(
        assigned=counts,
        assessed=assessed,
        observed=observed,
        first_interval=first_interval,
        current=current,
        final=final,
        infer=end == 0x47FE,
        suspend=suspend,
    )


cases = []
for assigned, pending, observed, first_interval, current, final in itertools.product(
    (2.0, 3.0, 4.0, 12.0), (0.0, 1.0, 2.0, 10.0), (0.0, 1.0), (0, 1), (1, 2), (False, True)
):
    if pending <= assigned and observed <= assigned:
        cases.append(run(assigned, assigned - pending, observed, first_interval, current, final))
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="pre-inference simulation screening from supplied counts only",
            cases=cases,
        ),
        indent=2,
    )
)
