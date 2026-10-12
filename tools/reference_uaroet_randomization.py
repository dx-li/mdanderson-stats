"""Bounded original UAROET1.8 randomization reference generator.

Requires pefile and Unicorn. Original executable is supplied, never distributed.
Host exp supplies CRT mathematics; native security-cookie check is bypassed.
Supplied summaries and uniform variates isolate native hybrid selection.
The unused ranking service is bypassed with prefilled ranks; top-rank restriction
flags are disabled. Native inference and random streams are not reproduced.
"""

import hashlib
import itertools
import json
import math
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP, UC_X86_REG_XMM0

path = Path(sys.argv[1])
if (
    hashlib.sha256(path.read_bytes()).hexdigest()
    != "b953affb6cd3968c389d5a35020e594cd25fef3ff61825db8aa282d8216a3eaf"
):
    raise ValueError("original UAROET executable hash mismatch")
p = pefile.PE(str(path))
base = p.OPTIONAL_HEADER.ImageBase
u = Uc(UC_ARCH_X86, UC_MODE_32)
u.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, p.get_memory_mapped_image())
u.mem_map(0x20000000, 0x200000)
u.mem_map(0, 4096)
obj, stack, end = 0x20010000, 0x20180000, 0x201F0000
u.mem_write(end, b"\xdd\x1d" + struct.pack("<I", end + 256))
u.mem_write(end + 16, b"\xc3")


def pack(addr, fmt, *v):
    u.mem_write(addr, struct.pack("<" + fmt, *v))


def hook(machine, address, size, data):
    if address == base + 0x92060:
        u.reg_write(UC_X86_REG_EIP, end + 16)
    elif address == base + 0xA3690:
        v = struct.unpack("<d", struct.pack("<Q", u.reg_read(UC_X86_REG_XMM0) & (2**64 - 1)))[0]
        u.reg_write(UC_X86_REG_XMM0, struct.unpack("<Q", struct.pack("<d", math.exp(v)))[0])
        u.reg_write(UC_X86_REG_EIP, end + 16)


u.hook_add(UC_HOOK_CODE, hook)
initial = u.context_save()


post = 0x20080000
uniform = 0.4
u.mem_write(end + 32, b"\xdd\x05" + struct.pack("<I", end + 256) + b"\xc3")


def extra(machine, address, size, data):
    if address == base + 0x2AC50:
        pack(end + 256, "d", uniform)
        u.reg_write(UC_X86_REG_EIP, end + 32)
    elif address == base + 0x58FD0:
        u.reg_write(UC_X86_REG_EIP, end + 16)


u.hook_add(UC_HOOK_CODE, extra)


def randomize(
    utility,
    treated,
    status=(1, 1, 1),
    hybrid=True,
    surplus=1,
    mean_weight=True,
    value=0.4,
    good=(0.2, 0.4, 0.6),
):
    global uniform
    uniform = value
    D = len(utility)
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x60000)
    u.mem_write(post, b"\0" * 0x8000)
    for off, val in [(0x264, post), (0x1F0, D), (0x1F4, D), (0x6C, D - 1), (0x138, surplus)]:
        pack(obj + off, "i", val)
    pack(obj + 0x19D, "B", int(hybrid))
    pack(obj + 0x19F, "B", int(mean_weight))
    pack(post + 0x7254, "i", D)
    pack(post + 0x19E8, "i", max(range(D), key=lambda d: (utility[d], d)))
    for i in range(D):
        pack(obj + 0xCC + i * 4, "i", status[i])
        pack(obj + 0x14 + i * 4, "i", treated[i])
        pack(post + 0x1220 + i * 8, "d", utility[i])
        pack(post + 0x1360 + i * 8, "d", good[i])
        pack(post + 0x199C + i * 4, "i", i)
    pack(stack, "I", end)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    u.emu_start(base + 0x5EEB0, end, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == end
    return dict(
        selected=struct.unpack("<i", u.mem_read(obj, 4))[0],
        weights=struct.unpack("<" + "d" * D, u.mem_read(obj + 0xE0, D * 8)),
        cutoffs=struct.unpack("<" + "d" * D, u.mem_read(obj + 0x108, D * 8)),
    )


cases = []
for utility, treated, status, hybrid, surplus, mean_weight, value in itertools.product(
    ([20, 40, 60], [60, 60, 60], [80, 20, 50]),
    ([3, 0, 0], [3, 3, 3], [3, 0, 4]),
    ([1, 1, 1], [1, -2, 1], [-1, 1, -2]),
    (False, True),
    (0, 1),
    (False, True),
    (0.1, 0.5, 0.9),
):
    kw = dict(
        utility=utility,
        treated=treated,
        status=status,
        hybrid=hybrid,
        surplus=surplus,
        mean_weight=mean_weight,
        value=value,
    )
    cases.append(dict(inputs=kw, result=randomize(**kw)))
print(
    json.dumps(
        dict(executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases), indent=2
    )
)
