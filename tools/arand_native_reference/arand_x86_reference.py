import json
import math
import struct

import pefile
from arand_common import library_path
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EAX,
    UC_X86_REG_ECX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESP,
    UC_X86_REG_XMM0,
    UC_X86_REG_XMM1,
)

p = pefile.PE(library_path())
base = p.OPTIONAL_HEADER.ImageBase


def run(values, power, floor):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, p.OPTIONAL_HEADER.SizeOfImage + 4095 & ~4095)
    u.mem_write(base, p.get_memory_mapped_image())
    u.mem_map(1048576, 131072)
    stack = 1114112
    this = 1052672
    vec = 1056768
    data = 1060864
    stop = 1050624
    u.mem_write(this, struct.pack("<Idd", 0, power, floor)[:4])
    u.mem_write(this + 8, struct.pack("<dd", power, floor))
    u.mem_write(vec + 8, struct.pack("<i", len(values)))
    u.mem_write(data, struct.pack("<" + "d" * len(values), *values))
    u.mem_write(stack, struct.pack("<II", stop, vec))
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, this)

    def ret(n):
        sp = u.reg_read(UC_X86_REG_ESP)
        addr = struct.unpack("<I", u.mem_read(sp, 4))[0]
        u.reg_write(UC_X86_REG_ESP, sp + 4 + n)
        u.reg_write(UC_X86_REG_EIP, addr)

    def hook(u, addr, size, _):
        if addr == stop:
            u.emu_stop()
        elif addr == base + 1044464:
            sp = u.reg_read(UC_X86_REG_ESP)
            i = struct.unpack("<i", u.mem_read(sp + 4, 4))[0]
            assert 0 <= i < len(values), i
            u.reg_write(UC_X86_REG_EAX, data + 8 * i)
            ret(4)
        elif addr == base + 1727920:
            a = struct.unpack(
                "<d", (u.reg_read(UC_X86_REG_XMM0) & (1 << 64) - 1).to_bytes(8, "little")
            )[0]
            b = struct.unpack(
                "<d", (u.reg_read(UC_X86_REG_XMM1) & (1 << 64) - 1).to_bytes(8, "little")
            )[0]
            u.reg_write(
                UC_X86_REG_XMM0, int.from_bytes(struct.pack("<d", math.pow(a, b)), "little")
            )
            ret(0)

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + 1237408, stop, count=1000000)
    assert u.reg_read(UC_X86_REG_EIP) == stop, "instruction budget exceeded"
    return struct.unpack("<" + "d" * len(values), u.mem_read(data, len(values) * 8))


if __name__ == "__main__":
    for v, t, f in [
        ([0.05, 0.15, 0.8], 1, 0.2),
        ([0.01, 0.09, 0.9], 1, 0.2),
        ([0.01, 0.09, 0.9], 2, 0.2),
        ([0.1, 0.2, 0.7], 0, 0.2),
        ([0.2, 0.3, 0.5], 1, 0),
        ([0.01, 0.19, 0.8], 1, 0.2),
        ([1], 1, 0),
    ]:
        print(json.dumps({"probabilities": v, "power": t, "floor": f, "native": run(v, t, f)}))
