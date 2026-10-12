"""Bounded original DSF2.2 instruction probe; original DLL is supplied externally."""

import json
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP

path = Path(sys.argv[1])
p = pefile.PE(str(path))
base = p.OPTIONAL_HEADER.ImageBase
u = Uc(UC_ARCH_X86, UC_MODE_32)
u.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, p.get_memory_mapped_image())
u.mem_map(0x20000000, 0x200000)
obj, post, stack, end = 0x20010000, 0x20011000, 0x20180000, 0x201F0000
u.mem_write(end, b"\xc3")
u.mem_write(end + 16, b"\xc2\x08\x00")
u.mem_write(end + 32, b"\xc3")


def pack(addr, fmt, *v):
    u.mem_write(addr, struct.pack("<" + fmt, *v))


def get(addr, fmt):
    return struct.unpack("<" + fmt, u.mem_read(addr, struct.calcsize("<" + fmt)))


matrices = {}
nextptr = 0x20020000
D, S = 3, 3
for owner in (obj, post):
    for offset in (0, 0x20, 0x58, 0x84, 0xB0, 0x128, 0x154):
        matrices[owner + offset] = nextptr
        nextptr += 0x1000


def hook(machine, address, size, data):
    if address in (base + 0xE0F0, base + 0x122F0):
        row, col = get(u.reg_read(UC_X86_REG_ESP) + 4, "ii")
        owner = u.reg_read(UC_X86_REG_ECX)
        assert 0 <= row < D and 0 <= col < S, (row, col)
        ptr = matrices[owner] + (row * S + col) * (8 if address == base + 0xE0F0 else 4)
        u.reg_write(UC_X86_REG_EAX, ptr)
        u.reg_write(UC_X86_REG_EIP, end + 16)
    elif address == base + 0x17EE0:
        u.reg_write(UC_X86_REG_EIP, end + 32)


u.hook_add(UC_HOOK_CODE, hook)
initial = u.context_save()


def call(rva):
    pack(stack, "I", end)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    u.emu_start(base + rva, end, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == end


def run(mean, overdose, highest=(0, 0), cutoff=0.8, target=0.3, mode="interim"):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x200)
    u.mem_write(post, b"\0" * 0x400)
    for ptr in matrices.values():
        u.mem_write(ptr, b"\0" * 0x1000)
    pack(obj + 0x188, "ii", D, S)
    pack(obj + 0x194, "I", post)
    pack(obj + 0x104, "ii", *highest)
    pack(obj + 0xE8, "d", cutoff)
    pack(obj + 0xF0, "d", target)
    pack(matrices[post], "d" * 9, *mean)
    pack(matrices[post + 0x58], "d" * 9, *overdose)
    if mode == "first":
        call(0x16E20)
    elif mode == "final":
        pack(stack, "III", end, end + 512, end + 516)
        u.reg_write(UC_X86_REG_ESP, stack)
        u.reg_write(UC_X86_REG_ECX, obj)
        u.emu_start(base + 0x16F00, end, count=100000)
        assert u.reg_read(UC_X86_REG_EIP) == end
    else:
        call(0x16930)
        call(0x16B50)
    return dict(
        mode=mode,
        mean=mean,
        overdose=overdose,
        highest=highest,
        cutoff=cutoff,
        target=target,
        admissible=get(matrices[obj + 0x154], "i" * 9),
        selected=get(obj, "ii"),
    )


if __name__ == "__main__":
    import numpy as np

    rows = []
    for highest in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)]:
        for pattern in range(512):
            rows.append(
                run(
                    [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
                    [0.9 if pattern & (1 << j) else 0.1 for j in range(9)],
                    highest,
                    target=0.3,
                )
            )
    for tail in (0.0, 0.8, float(np.nextafter(0.8, 0)), float(np.nextafter(0.8, 1)), 1.0):
        rows.append(run([0.3] * 9, [tail] * 9, (2, 2)))
    rows += [
        run(r["mean"], r["overdose"], r["highest"], r["cutoff"], r["target"], mode="final")
        for r in list(rows)
    ]
    rows += [run([0.3] * 9, [v] * 9, (-1, -1), mode="first") for v in (0.0, 0.8, 1.0)]
    print(json.dumps(rows, indent=2))
