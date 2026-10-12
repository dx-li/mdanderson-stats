import json
import struct

from arand_x86_reference import base, p
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EAX,
    UC_X86_REG_ECX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESP,
    UC_X86_REG_FPCW,
)


def run(values, mode="pending"):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, p.OPTIONAL_HEADER.SizeOfImage + 4095 & ~4095)
    u.mem_write(base, p.get_memory_mapped_image())
    u.mem_map(0, 4096)
    u.mem_map(1048576, 131072)
    trial = 1052672
    ps = 1056768
    os = 1060864
    stop = 1050624
    stub = 1050880
    valueptr = 1069056
    stack = 1114112

    def wi(a, v):
        u.mem_write(a, struct.pack("<I", v))

    def tree(head, n, first):
        wi(head, first if n else head)
        u.mem_write(head + 13, b"\x01")
        for i in range(n):
            node = first + i * 128
            wi(node, head)
            wi(node + 4, head if i == 0 else node - 128)
            wi(node + 8, node + 128 if i + 1 < n else head)

    no = len(values[0]) if values else 1
    tree(ps, len(values), 1073152)
    tree(os, no, 1081344)
    for i in range(len(values)):
        wi(1073152 + i * 128 + 40, 1085440 + i * 128)
    wi(trial + 92, os)
    wi(trial + 100, ps)
    wi(trial + 116, os)
    wi(trial + 120, no)
    u.mem_write(stub, b"\xdd\x05" + struct.pack("<I", valueptr) + b"\xc2\x0c\x00")
    u.mem_write(stack, struct.pack("<Id", stop, 5.0))
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, trial)
    u.reg_write(UC_X86_REG_FPCW, 895)

    def ret(n):
        sp = u.reg_read(UC_X86_REG_ESP)
        addr = struct.unpack("<I", u.mem_read(sp, 4))[0]
        u.reg_write(UC_X86_REG_ESP, sp + 4 + n)
        u.reg_write(UC_X86_REG_EIP, addr)

    def hook(u, addr, size, _):
        if addr == stop:
            u.emu_stop()
        elif addr == base + 1293504:
            sp = u.reg_read(UC_X86_REG_ESP)
            key = struct.unpack("<I", u.mem_read(sp + 4, 4))[0]
            i = (u.reg_read(UC_X86_REG_ECX) - 1085440) // 128
            j = (key - 1081360) // 128 if mode == "pending" else 0
            u.mem_write(valueptr, struct.pack("<d", values[i][j]))
            u.reg_write(UC_X86_REG_EIP, stub)
        elif addr == base + 1036512:
            ret(12)
        elif addr == base + 1726456:
            ret(0)

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + (1306672 if mode == "pending" else 1302928), stop, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == stop
    return u.reg_read(UC_X86_REG_EAX) & 255 if mode == "pending" else u.reg_read(UC_X86_REG_EAX)


if __name__ == "__main__":
    for v in [[], [[0], [1]], [[0], [-1]], [[0, 1], [1, 0]], [[0, 1], [1, -1]], [[0, -1], [1, 0]]]:
        print(
            json.dumps(
                {
                    "input": v,
                    "native_pending": run(v),
                    "native_observed": run(v, "observed")
                    if not v or len(v[0]) == 1
                    else "unsupported: native throws",
                }
            )
        )
