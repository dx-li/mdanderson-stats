"""Bounded original DSF2.2 instruction probe; original DLL is supplied externally."""

import json
import math
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP

p = pefile.PE(sys.argv[1])
base = p.OPTIONAL_HEADER.ImageBase
u = Uc(UC_ARCH_X86, UC_MODE_32)
u.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, p.get_memory_mapped_image())
u.mem_map(0x20000000, 0x200000)
obj, stack, end = 0x20010000, 0x20180000, 0x201F0000
u.mem_write(end, b"\xc3")
u.mem_write(end + 16, b"\xc2\x04\x00")
# Store ST0, service it at the next address, reload result and return.
u.mem_write(
    end + 32,
    b"\xdd\x1d"
    + struct.pack("<I", end + 256)
    + b"\xdd\x05"
    + struct.pack("<I", end + 256)
    + b"\xc3",
)


def pack(addr, fmt, *v):
    u.mem_write(addr, struct.pack("<" + fmt, *v))


def get(addr, fmt):
    return struct.unpack("<" + fmt, u.mem_read(addr, struct.calcsize("<" + fmt)))


ptrs = {}
pending = None
mathcalls = {0x1ECAC: math.log, 0x1ECB8: math.exp, 0x1ECBE: math.sqrt}


def ptr(owner):
    if owner not in ptrs:
        ptrs[owner] = 0x20020000 + len(ptrs) * 0x1000
    return ptrs[owner]


def hook(machine, address, size, data):
    global pending
    if address == base + 0xCAB0:
        i = get(u.reg_read(UC_X86_REG_ESP) + 4, "i")[0]
        assert 0 <= i < 7
        u.reg_write(UC_X86_REG_EAX, ptr(u.reg_read(UC_X86_REG_ECX)) + 8 * i)
        u.reg_write(UC_X86_REG_EIP, end + 16)
    elif address - base in mathcalls:
        pending = mathcalls[address - base]
        u.reg_write(UC_X86_REG_EIP, end + 32)
    elif address == end + 38:
        pack(end + 256, "d", pending(get(end + 256, "d")[0]))


u.hook_add(UC_HOOK_CODE, hook)
initial = u.context_save()


def run(prob, peak, tail, admins, nu1, nu2):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x1000)
    for q in ptrs.values():
        u.mem_write(q, b"\0" * 0x1000)
    D = len(prob)
    pack(obj + 0x7FC, "i", D)
    pack(obj + 0x89C, "i", admins)
    for offset, values in [(0x818, prob), (0x844, peak), (0x870, tail)]:
        pack(ptr(obj + offset), "d" * D, *values)
    pack(stack, "Idd", end, nu1, nu2)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    u.emu_start(base + 0x1A630, base + 0x1AAC2, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x1AAC2
    return dict(
        prob=prob,
        peak=peak,
        tail=tail,
        admins=admins,
        nu1=nu1,
        nu2=nu2,
        area_mean=get(ptr(obj + 0x778), "d" * D),
        area_sd=get(ptr(obj + 0x74C), "d" * D),
        peak_mean=get(ptr(obj + 0x5EC), "d" * D),
        tail_mean=get(ptr(obj + 0x618), "d" * D),
        time_sd=get(obj + 0x5B0, "d")[0],
    )


if __name__ == "__main__":
    rows = []
    import numpy as np

    for d in (1, 3, 7):
        for admins in (1, 3, 5):
            for a, b in ((1.01, 1.1), (1.5, 1.5), (2.0, 3.0), (10.0, 5.0), (100.0, 100.0)):
                rows.append(
                    run(
                        np.linspace(0.05, 0.75, d).tolist(),
                        np.linspace(2.0, 7.0, d).tolist(),
                        np.linspace(5.0, 20.0, d).tolist(),
                        admins,
                        a,
                        b,
                    )
                )
    print(json.dumps(rows, indent=2))
