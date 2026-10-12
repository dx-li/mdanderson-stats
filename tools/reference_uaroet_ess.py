"""Bounded original UAROET1.8 ESS moment-reduction reference generator.

Requires pefile and Unicorn. Original executable is supplied, never distributed.
Host exp supplies CRT mathematics; native security-cookie check is bypassed.
Supplied synthetic probability draws populate moment accumulators. Native
mean, sample variance and beta moment reduction execute; prior sampling is not
reproduced.
"""

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EBP,
    UC_X86_REG_EBX,
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


def ess(efficacy, toxicity, toxicity_only=False):
    e = np.array(efficacy)
    t = np.array(toxicity)
    N, D, E = e.shape
    T = t.shape[-1]
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x60000)
    frame = stack + 0x500
    records = 0x20080000
    u.mem_write(records, b"\0" * 0x10000)
    u.mem_write(stack - 0x1000, b"\0" * 0x2000)
    for off, v in [(0x7254, D), (0x725C, E), (0x7260, T), (0x69C0, 13)]:
        pack(obj + off, "i", v)
    pack(obj + 0x7230, "B", toxicity_only)
    pack(frame - 0x64, "I", records)
    for d in range(D):
        for k in range(13):
            values = np.linspace(0.1, 0.2, N)
            if k in (0, 1, 2) and k + 1 < E:
                values = e[:, d, k + 1]
            elif k in (3, 4, 5) and k - 2 < T:
                values = t[:, d, k - 2]
            elif k == 11:
                values = e[:, d, 0]
            elif k == 12:
                values = t[:, d, 0]
            at = records + (d * 13 + k) * 48
            pack(at, "i", N)
            pack(at + 8, "d", float(values.sum()))
            pack(at + 0x28, "d", float(((values - values.mean()) ** 2).sum()))
    u.reg_write(UC_X86_REG_EBX, obj)
    u.reg_write(UC_X86_REG_EBP, frame)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.emu_start(base + 0x43DC0, base + 0x44483, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x44483

    def get(at, n):
        return struct.unpack("<" + "d" * n, u.mem_read(at, n * 8))

    return dict(
        efficacy=[get(obj + 0x1B98 + d * 32, E) for d in range(D)],
        toxicity=[get(obj + 0x1C38 + d * 32, T) for d in range(D)],
        summary=get(obj + 0x1CD8, 5),
    )


cases = []
rng = np.random.default_rng(991)
for D in (1, 3, 5):
    for E, T in ((2, 2), (3, 4), (4, 3), (4, 4)):
        for N in (8, 32, 128):
            e = rng.dirichlet(np.arange(1, E + 1), size=(N, D)).tolist()
            t = rng.dirichlet(np.arange(T, 0, -1), size=(N, D)).tolist()
            cases.append(dict(efficacy=e, toxicity=t, result=ess(e, t)))
print(
    json.dumps(
        dict(executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases), indent=2
    )
)
