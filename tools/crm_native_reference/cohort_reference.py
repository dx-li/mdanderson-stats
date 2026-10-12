"""Execute original cohort-completion helper and DA within-cohort safety slice."""

import json
import struct

from patient_reference import legacy, p
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_EBP, UC_X86_REG_EBX, UC_X86_REG_ESP

BASE = p.OPTIONAL_HEADER.ImageBase


def machine():
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(BASE, 0x100000)
    u.mem_write(BASE, p.get_memory_mapped_image())
    u.mem_map(0x600000, 0x200000)
    u.reg_write(UC_X86_REG_ESP, 0x680000)
    return u


def put(u, address, *values):
    u.mem_write(address, struct.pack("<" + "I" * len(values), *values))


def cohort(cohorts, size):
    u = machine()
    put(u, 0x680000, 0x700000, 0x610000, size)
    put(u, 0x610000, 0x620000, 0x620000 + 72 * len(cohorts))
    for i, value in enumerate(cohorts):
        put(u, 0x620000 + 72 * i + 4, value)
    u.emu_start(BASE + 0x331F0 - (16 if legacy else 0), 0x700000, count=10000)
    return {"cohorts": cohorts, "size": size, "incomplete": bool(u.reg_read(UC_X86_REG_EAX) & 255)}


def safety(flag):
    u = machine()
    u.reg_write(UC_X86_REG_EBP, 0x650000)
    u.reg_write(UC_X86_REG_EBX, 0x610000)
    u.mem_write(0x650000 - 0xF8, bytes([flag]))
    u.emu_start(
        BASE + 0x3213B - (16 if legacy else 0), BASE + 0x3214A - (16 if legacy else 0), count=100
    )
    return {
        "da_within_cohort_safety": flag,
        "action": struct.unpack("<I", u.mem_read(0x610000, 4))[0],
    }


if __name__ == "__main__":
    for size in range(1, 5):
        for cohorts in (
            [],
            [1],
            [1, 1],
            [1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 2],
            [1, 1, 2, 2],
            [0, 0, 1],
        ):
            print(json.dumps(cohort(cohorts, size), sort_keys=True))
    for flag in (0, 1):
        print(json.dumps(safety(flag), sort_keys=True))
