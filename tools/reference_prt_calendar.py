"""Native PRT simulation as-of conversion and last-three-patient final clock.

Executes3cae–3f2f and3b01–3cae on synthetic arrived patients. No posterior or
whole-trial parity is claimed. Original simulation executable required locally.
"""

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
if digest != "52903a7af2e75998bf2a8cbda6e2049e7fa25e0981cf9806317841da13a91b54":
    raise ValueError("original simulation executable hash mismatch")
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase


def run(rows, endpoints, clock, final=False):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(address, fmt, *values):
        u.mem_write(address, struct.pack("<" + fmt, *values))

    def get(address, fmt):
        return struct.unpack("<" + fmt, u.mem_read(address, struct.calcsize("<" + fmt)))

    window, n = int(endpoints[-1]), len(rows)
    put(0x78BF78, "i", n)
    put(0x78BF48, "i", window)
    put(bp - 0x48, "d", window)
    put(bp - 0xE8, "d", clock)
    put(bp - 0x68, "i", window - 1)
    put(bp - 0x6C, "i", 0)
    put(bp - 0x88, "i", len(endpoints) + 1)
    put(0xCBB100, "d" * window, *range(1, window + 1))
    terminal = struct.unpack("<d", pe.get_data(0x4B758, 8))[0]
    put(0xC7F960, "d" * (len(endpoints) + 2), 0.0, *endpoints, terminal)
    for i, (arrival, delay) in enumerate(rows):
        put(0x477040 + 8 * i, "d", arrival)
        put(0x47A240 + 8 * i, "d", delay)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x23F)
    if final:
        put(bp - 0x74, "i", n)
        put(bp - 0xD0, "i", n)
        start, end = (0x3919 if final == "after_accrual" else 0x3B01), 0x3CAE
    else:
        start, end = 0x3CAE, 0x3F2F
    u.emu_start(base + start, base + end, count=1000000)
    if u.reg_read(UC_X86_REG_EIP) != base + end:
        raise RuntimeError("native calendar instruction budget exhausted")
    return dict(
        clock=get(bp - 0xE8, "d")[0],
        observed_time=get(0x478940, "d" * n),
        censored=get(0x473E40, "i" * n),
        daily_index=get(0x474AC0, "i" * n),
        interval=get(0x450230, "i" * n),
    )


cases = []
for endpoints in ([3.0, 7.0, 10.0], [15.0, 30.0, 45.0, 60.0, 75.0, 90.0]):
    w = endpoints[-1]
    for delay in (0.0, 0.1, 1.0, 2.0, endpoints[0], w - 1e-6, w, w + 1):
        for clock in (0.0, endpoints[0], w, w + 1):
            rows = [[0.0, delay]]
            cases.append(
                dict(
                    rows=rows, endpoints=endpoints, clock=clock, result=run(rows, endpoints, clock)
                )
            )
final_cases = []
for rows in (
    [[0.0, 90.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
    [[0.0, 100.0], [1.0, 100.0], [2.0, 100.0]],
    [[0.0, 90.0], [2.0, 90.0], [4.0, 90.0], [6.0, 1.0], [8.0, 1.0], [10.0, 1.0]],
):
    final_cases.append(
        dict(
            rows=rows,
            endpoints=[30.0, 60.0, 90.0],
            result=run(rows, [30.0, 60.0, 90.0], 0.0, True),
            after_accrual_clock=run(rows, [30.0, 60.0, 90.0], 0.0, "after_accrual")["clock"],
        )
    )
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="native as-of conversion/final clock",
            cases=cases,
            final_cases=final_cases,
        ),
        indent=2,
    )
)
