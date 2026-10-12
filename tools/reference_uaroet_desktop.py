"""Bounded original UAROET1.8 eta and marginal reference generator.

Requires pefile and Unicorn. Original executable is supplied, never distributed.
Host exp supplies CRT mathematics; native security-cookie check is bypassed.
No inference or controller parity is established by these probes.
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
    elif address == base + 0xA3690:
        v = struct.unpack("<d", struct.pack("<Q", u.reg_read(UC_X86_REG_XMM0) & (2**64 - 1)))[0]
        u.reg_write(UC_X86_REG_XMM0, struct.unpack("<Q", struct.pack("<d", math.exp(v)))[0])
        u.reg_write(UC_X86_REG_EIP, end + 16)


u.hook_add(UC_HOOK_CODE, hook)
initial = u.context_save()


def run(parameters, doses, endpoint, dose, category, lognormal=False):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x60000)
    pack(obj, "d" * len(parameters), *parameters)
    pack(obj + 0x58DC8, "i", doses)
    pack(obj + 0x147, "B", int(lognormal))
    pack(stack, "Iiii", end, endpoint, dose, category)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    u.emu_start(base + 0x3A0A0, end + 6, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == end + 6
    return struct.unpack("<d", u.mem_read(end + 256, 8))[0]


def marginal(parameters, doses, endpoint, dose, categories=4, lognormal=False):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x60000)
    pack(obj, "d" * len(parameters), *parameters)
    pack(obj + 0x147, "B", int(lognormal))
    for offset, val in [
        (0x58D98, 5),
        (0x58DC8, doses),
        (0x58DCC, categories),
        (0x58DD0, categories),
    ]:
        pack(obj + offset, "i", val)
    pack(stack, "Iii", end, dose, endpoint)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    u.emu_start(base + 0x3B7D0, end, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == end
    return struct.unpack(
        "<4d", u.mem_read(obj + (0x1C0 + dose * 40 if endpoint == 0 else 0x288 + dose * 32), 32)
    )


eta = []
marginals = []
for D in (2, 3, 5):
    for pattern in (
        [-1.0, 0.2, 0.3, 0.1, 0.4],
        [-0.5, -0.2, 0.0, 0.8, 0.15],
        [0.75, 0.01, 0.6, 0.02, 0.3],
    ):
        parameters = [
            v + 0.1 * k if j == 0 else v for k in range(6) for j, v in enumerate(pattern)
        ] + [0.0]
        for flag in (False, True):
            for endpoint in (0, 1):
                for dose in range(D):
                    for category in (1, 2, 3):
                        eta.append(
                            dict(
                                parameters=parameters,
                                doses=D,
                                endpoint=endpoint,
                                dose=dose,
                                category=category,
                                lognormal=flag,
                                value=run(parameters, D, endpoint, dose, category, flag),
                            )
                        )
                    for categories in (2, 3, 4):
                        marginals.append(
                            dict(
                                parameters=parameters,
                                doses=D,
                                endpoint=endpoint,
                                dose=dose,
                                categories=categories,
                                lognormal=flag,
                                value=marginal(parameters, D, endpoint, dose, categories, flag),
                            )
                        )
print(
    json.dumps(
        dict(
            executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            eta=eta,
            marginals=marginals,
        ),
        indent=2,
    )
)
