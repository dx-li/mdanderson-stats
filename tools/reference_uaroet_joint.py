"""Original UAROET1.8 Gaussian-copula joint probabilities.

Executes marginal logits, inverse-normal transforms, native Gaussian quadrature
and joint-cell differencing. Only CRT exp/log/asin/sin/sqrt and the stack-cookie
check are supplied by the harness. Original executable is hash-pinned and not
redistributed. No posterior sampling or clinical validation is claimed.
"""

import hashlib
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
    elif address - base in (0xA3690, 0xA3970, 0xA4140, 0xA3C00, 0xA3DD0):
        v = struct.unpack("<d", struct.pack("<Q", u.reg_read(UC_X86_REG_XMM0) & (2**64 - 1)))[0]
        u.reg_write(
            UC_X86_REG_XMM0,
            struct.unpack(
                "<Q",
                struct.pack(
                    "<d",
                    {
                        0xA3690: math.exp,
                        0xA3970: math.log,
                        0xA4140: math.asin,
                        0xA3C00: math.sin,
                        0xA3DD0: math.sqrt,
                    }[address - base](v),
                ),
            )[0],
        )
        u.reg_write(UC_X86_REG_EIP, end + 16)


u.hook_add(UC_HOOK_CODE, hook)
initial = u.context_save()


def joint(parameters, D, E, T):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x60000)
    pack(stack, "I", end)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj + 0x58DF8)
    u.emu_start(base + 0x2D450, end, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == end
    pack(obj, "31d", *parameters)
    for offset, value in [(0x58D98, 5), (0x58DC8, D), (0x58DCC, E), (0x58DD0, T), (0x58DB0, 30)]:
        pack(obj + offset, "i", value)
    pack(stack, "I", end)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    u.emu_start(base + 0x3C150, base + 0x3C1D9, count=2000000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x3C1D9
    return [
        [
            [
                struct.unpack("<d", u.mem_read(obj + 0x8A0 + (d * 16 + e * 4 + t) * 8, 8))[0]
                for t in range(T)
            ]
            for e in range(E)
        ]
        for d in range(D)
    ]


def normal_quantile(probability):
    u.context_restore(initial)
    pack(stack, "Id", end, probability)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.emu_start(base + 0x84EA0, end + 6, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == end + 6
    return struct.unpack("<d", u.mem_read(end + 256, 8))[0]


cases = []
for D in (2, 3, 5):
    for E in (2, 3, 4):
        for T in (2, 3, 4):
            for pattern in ([-1, 0.2, 0.3, 0.1, 0.4], [0.75, 0.01, 0.6, 0.02, 0.3]):
                for rho in (-0.95, -0.6, 0, 0.35, 0.8, 0.99):
                    parameters = [
                        v + 0.1 * k if j == 0 else v
                        for k in range(6)
                        for j, v in enumerate(pattern)
                    ] + [rho]
                    cases.append(
                        dict(
                            doses=D,
                            efficacy_levels=E,
                            toxicity_levels=T,
                            parameters=parameters,
                            joint=joint(parameters, D, E, T),
                        )
                    )
print(
    json.dumps(
        dict(
            executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            cases=cases,
            normal_quantiles=[
                dict(probability=v, value=normal_quantile(v))
                for v in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
            ],
        ),
        indent=2,
    )
)
