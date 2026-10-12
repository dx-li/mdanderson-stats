"""Native cohort enrollment/as-of clock with supplied delays; no posterior/RNG claim."""

import hashlib
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


def run(maximum, decisions, delays):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(address, fmt, *values):
        u.mem_write(address, struct.pack("<" + fmt, *values))

    def get(address, fmt):
        return struct.unpack("<" + fmt, u.mem_read(address, struct.calcsize("<" + fmt)))

    arrivals = [float(i + 1) for i in range(40)]
    endpoints = [3.0, 7.0, 10.0]
    put(0xCCD648, "d" * len(arrivals), *arrivals)
    put(0x78BF48, "i", 10)
    put(bp - 0x48, "d", 10.0)
    put(bp - 0x68, "i", 9)
    put(bp - 0x6C, "i", 0)
    put(bp - 0x88, "i", 4)
    put(0xCBB100, "d" * 10, *range(1, 11))
    terminal = struct.unpack("<d", pe.get_data(0x4B758, 8))[0]
    put(0xC7F960, "d" * 5, 0.0, *endpoints, terminal)
    put(bp - 0x74, "i", maximum)
    put(bp - 0xD8, "i", 3)
    put(bp - 0xE8, "d", 0.0)

    def skip_generation(uc, address, size, data):
        if address == base + 0x3779:
            index = get(bp - 0xC4, "i")[0] - 1
            put(get(bp - 0x100, "i")[0], "d", delays[index])
            put(get(bp - 0xFC, "i")[0], "i", get(bp - 0xBC, "i")[0])
            uc.reg_write(UC_X86_REG_EIP, base + 0x38C3)

    u.hook_add(UC_HOOK_CODE, skip_generation, begin=base + 0x3779, end=base + 0x3779)
    rows = []
    for suspended, dose in decisions:
        put(bp - 0xD4, "i", suspended)
        put(bp - 0xBC, "i", dose)
        u.reg_write(UC_X86_REG_EBP, bp)
        u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
        u.reg_write(UC_X86_REG_FPCW, 0x23F)
        u.emu_start(base + 0x36AB, base + 0x3F2F, count=100000)
        if u.reg_read(UC_X86_REG_EIP) != base + 0x3F2F:
            raise RuntimeError("instruction budget")
        n = get(bp - 0xD0, "i")[0]
        rows.append(
            dict(
                enrolled=n,
                candidate_index=get(bp - 0xC8, "i")[0],
                clock=get(bp - 0xE8, "d")[0],
                arrivals=get(0x477040, "d" * n),
                doses=get(0x47BCD8, "i" * n),
                observed_time=get(0x478940, "d" * n),
                censored=get(0x473E40, "i" * n),
                interval=get(0x450230, "i" * n),
            )
        )
        u.emu_start(base + 0x4000, base + 0x4045, count=1000)
        if n >= maximum:
            break
    return dict(
        maximum=maximum,
        decisions=decisions,
        delays=delays,
        arrivals=arrivals,
        endpoints=endpoints,
        result=rows,
    )


cases = []
for maximum in (7, 8, 9):
    for delays in (
        [11.0] * 9,
        [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [10.0, 10.0, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ):
        for decisions in ([(0, 1), (0, 2), (0, 3)], [(0, 1), (1, 1), (1, 1), (0, 1), (0, 2)]):
            cases.append(run(maximum, decisions, delays))
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="native supplied-delay enrollment/as-of trace; inference and RNG excluded",
            cases=cases,
        ),
        indent=2,
    )
)
