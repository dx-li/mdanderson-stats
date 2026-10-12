"""Execute the advertised SimARAND replication loop, RVA0x18d4b5..0x18d51a.

Progress printing is a skipped sink; reset/run/accumulate are recorded services.
This verifies ordering and count, not RNG values or replacement posterior code.
"""

import json
import struct

from arand_desktop_report_reference import machine
from arand_x86_reference import base
from unicorn import UC_HOOK_CODE
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_EDI, UC_X86_REG_EIP, UC_X86_REG_ESP


def run(replications):
    if not isinstance(replications, int) or not 0 <= replications <= 10000:
        raise ValueError("replications out of bounds")
    u = machine()
    obj = 1052672
    director = 1056768
    table = 1060864
    director_table = 1064960
    services = {1050624: "director_reset", 1050688: "run_trial", 1050752: "accumulate"}
    u.mem_write(obj, struct.pack("<I", table))
    u.mem_write(obj + 28, struct.pack("<i", replications))
    u.mem_write(obj + 36, struct.pack("<I", director))
    u.mem_write(director, struct.pack("<I", director_table))
    u.mem_write(director_table + 32, struct.pack("<I", 1050624))
    u.mem_write(table + 24, struct.pack("<I", 1050688))
    u.mem_write(table + 32, struct.pack("<I", 1050752))
    u.reg_write(UC_X86_REG_EDI, obj)
    u.reg_write(UC_X86_REG_EBP, 1114368)
    calls = []
    progress = []

    def hook(u, address, size, data):
        if address == base + 0x18D4D7:
            progress.append(len(calls) // 3)
            u.reg_write(UC_X86_REG_EIP, base + 0x18D501)
        elif address in services:
            calls.append(services[address])
            sp = u.reg_read(UC_X86_REG_ESP)
            target = struct.unpack("<I", u.mem_read(sp, 4))[0]
            u.reg_write(UC_X86_REG_ESP, sp + 4)
            u.reg_write(UC_X86_REG_EIP, target)

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + 0x18D4B5, base + 0x18D51A, count=1000000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x18D51A
    return dict(replications=replications, calls=calls, progress=progress)


if __name__ == "__main__":
    for n in (0, 1, 2, 3, 100, 101):
        print(json.dumps(run(n), sort_keys=True))
