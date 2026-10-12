"""Native UAROET1.8 utility-cutoff schedule, supplied observed-likelihood total.

Executes5d38d–5d40d. No posterior fitting, patient counting or selection occurs.
Requires pefile, Unicorn and the externally supplied hash-pinned executable.
"""

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EBP,
    UC_X86_REG_EDI,
    UC_X86_REG_EDX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESP,
    UC_X86_REG_XMM0,
)

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


def shrink(planned, steps, observed, initial_cutoff, final_cutoff):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x60000)
    u.mem_write(post, b"\0" * 0x8000)
    for off, value in [(0xA8, planned), (0x180, steps), (0x264, post)]:
        pack(obj + off, "i", value)
    pack(obj + 0x170, "d", initial_cutoff)
    pack(obj + 0x178, "d", final_cutoff)
    pack(post + 0x1AF0, "i", observed)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_EBP, stack + 0x500)
    u.reg_write(UC_X86_REG_EDI, obj)
    u.reg_write(UC_X86_REG_EDX, post)
    u.emu_start(base + 0x5D38D, base + 0x5D40D, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x5D40D
    return struct.unpack("<d", u.mem_read(obj + 0x188, 8))[0]


cases = []
for N in (12, 30, 31, 500):
    for K in (0, 1, 2, 3, N - 1, N):
        width = N // (K + 1)
        observed = sorted(
            {
                0,
                1,
                N // 2,
                N - 1,
                N,
                *[min(N, max(0, k * width + off)) for k in (1, K) for off in (-1, 0, 1)],
            }
        )
        for n in observed:
            for a, b in ((100, 0), (50, 25), (100, 100)):
                values = dict(planned=N, steps=K, observed=n, initial_cutoff=a, final_cutoff=b)
                cases.append(dict(inputs=values, result=shrink(**values)))
print(
    json.dumps(
        dict(executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases), indent=2
    )
)
