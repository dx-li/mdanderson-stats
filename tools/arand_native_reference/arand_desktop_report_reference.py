"""Actual SimARAND native report slices: stopped-arm count and enrollment percentiles.

RVA0x18bf90 is reached through derived slot16, resolved via CLR fixups.
Services: per-status count lookup at0x138860 and scalar Ceiling at0x1a5de6.
No full report rendering or sorting execution is claimed.
"""

import json
import math
import struct

from arand_x86_reference import base, p
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EAX,
    UC_X86_REG_EBX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESP,
    UC_X86_REG_FPCW,
)


def machine():
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    u.mem_write(base, p.get_memory_mapped_image())
    u.mem_map(1048576, 131072)
    u.reg_write(UC_X86_REG_ESP, 1114112)
    u.reg_write(UC_X86_REG_FPCW, 895)
    return u


def stopped(counts):
    u = machine()
    stack = 1114112
    obj = 1052672
    arm = 1056768
    u.mem_write(stack + 0x44, struct.pack("<I", obj))
    u.mem_write(obj + 0x70, struct.pack("<I", 1060864))
    u.reg_write(UC_X86_REG_EBX, arm)
    u.mem_write(arm + 0x18, struct.pack("<I", 1064960))
    reads = []

    def hook(u, address, size, data):
        if address == base + 0x18CAD5:
            u.emu_stop()
        elif address == base + 0x138860:
            sp = u.reg_read(UC_X86_REG_ESP)
            ret, group, status = struct.unpack("<III", u.mem_read(sp, 12))
            reads.append(status)
            u.reg_write(UC_X86_REG_EAX, counts[status])
            u.reg_write(UC_X86_REG_ESP, sp + 12)
            u.reg_write(UC_X86_REG_EIP, ret)

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + 0x18CA87, base + 0x18CAD5, count=10000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x18CAD5
    return dict(
        counts=counts,
        queried_statuses=reads,
        stopped=struct.unpack("<i", u.mem_read(stack + 0x68, 4))[0],
    )


def percentiles(n):
    if not isinstance(n, int) or not 0 <= n <= 10000:
        raise ValueError("size out of bounds")
    u = machine()
    vector = 1052672
    values = 1060864
    scratch = 1050624
    result = 1050640
    u.mem_write(vector, struct.pack("<II", values, values + 4 * n))
    if n:
        u.mem_write(values, struct.pack("<" + "i" * n, *range(n)))
    u.reg_write(UC_X86_REG_EBX, vector)
    # A service trampoline returns the supplied mathematical ceil in x87 ST0.
    u.mem_write(scratch, b"\xdd\x05" + struct.pack("<I", result) + b"\xc3")
    reads = []
    suppressed = False

    def hook(u, address, size, data):
        nonlocal suppressed
        if address == base + 0x18D061:
            suppressed = True
            u.emu_stop()
        elif address == base + 0x18D039:
            u.emu_stop()
        elif address == base + 0x1A5DE6:
            sp = u.reg_read(UC_X86_REG_ESP)
            value = struct.unpack("<d", u.mem_read(sp + 4, 8))[0]
            u.mem_write(result, struct.pack("<d", math.ceil(value)))
            u.reg_write(UC_X86_REG_EIP, scratch)
        elif address in (base + 0x18D010, base + 0x18D01E):
            index = u.reg_read(UC_X86_REG_EAX if address == base + 0x18D010 else UC_X86_REG_EBX)
            assert 0 <= index < n
            reads.append(index)

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + 0x18CF36, base + 0x18D039, count=10000)
    assert u.reg_read(UC_X86_REG_EIP) in (base + 0x18D039, base + 0x18D061)
    return dict(size=n, suppressed=suppressed, indices=reads)


if __name__ == "__main__":
    for counts in ([0] * 10, list(range(10)), [0, 2, 3, 5, 7, 11, 0, 0, 0, 13]):
        print(json.dumps(dict(kind="stopped", native=stopped(counts)), sort_keys=True))
    for n in (0, 1, 20, 99, 100, 101, 120, 199, 200, 1000):
        print(json.dumps(dict(kind="percentiles", native=percentiles(n)), sort_keys=True))
