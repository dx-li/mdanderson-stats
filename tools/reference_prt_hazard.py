"""PRT1.0 native beta-shaped hazard calibration and supplied-variate event times.

Runs actual native grid22c4–234c, calibration24f8–274d and generation3784–38ac.
Only the second unit uniform is supplied by a hook; the first is a declared
input. Original executable required locally; no original assets redistributed.
"""

import hashlib
import itertools
import json
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EBP,
    UC_X86_REG_EBX,
    UC_X86_REG_EDI,
    UC_X86_REG_EIP,
    UC_X86_REG_ESI,
    UC_X86_REG_ESP,
    UC_X86_REG_FPCW,
)

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != "52903a7af2e75998bf2a8cbda6e2049e7fa25e0981cf9806317841da13a91b54":
    raise ValueError("original simulation executable hash mismatch")
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase
bp, scale, alpha_ptr, beta_ptr, rates = 0x20100000, 0x472DF0, 0x46FBF0, 0x46FD80, 0xCCB708
stub, supplied = 0x201F0000, 0x201F0100


def run(alpha, beta, probability, window):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    u.mem_write(stub, b"\xdd\x05" + struct.pack("<I", supplied) + b"\xc3")

    def put(address, fmt, *values):
        u.mem_write(address, struct.pack("<" + fmt, *values))

    def get(address, fmt):
        return struct.unpack("<" + fmt, u.mem_read(address, struct.calcsize("<" + fmt)))

    def execute(start, end):
        u.emu_start(base + start, base + end, count=100_000_000)
        if u.reg_read(UC_X86_REG_EIP) != base + end:
            raise RuntimeError("native hazard instruction budget exhausted")

    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_FPCW, 0x37F)
    put(0x78BF48, "i", window)
    u.reg_write(UC_X86_REG_EBX, 99)
    u.reg_write(UC_X86_REG_ESI, 0x47CC78)
    execute(0x22C4, 0x234C)
    grid = get(0x47CC78, "d" * 100)
    put(alpha_ptr, "d", alpha)
    put(beta_ptr, "d", beta)
    put(scale - 0x2EE0, "d", probability)
    put(bp - 0x58, "I", alpha_ptr)
    put(bp - 0x60, "I", beta_ptr)
    put(bp - 0x9C, "I", rates)
    u.reg_write(UC_X86_REG_EDI, scale)
    execute(0x24F8, 0x274D)
    if get(bp - 0xAC, "i")[0] != 1:
        raise RuntimeError("native hazard calibration did not converge")
    result = dict(
        alpha=alpha,
        beta=beta,
        probability=probability,
        window=window,
        grid=grid,
        scale=get(scale, "d")[0],
        iterations=get(bp - 0xA0, "i")[0],
        rates=get(rates, "d" * 100),
        integrated_hazard=get(0x78BEF8, "d")[0],
    )
    # Actual CDF construction:99 interval steps, then the terminal category is1.
    put(bp - 0xE0, "I", 0x320)
    execute(0x35FA, 0x36AB)
    put(0x47D2B0, "d", 1.0)
    result["cdf"] = get(0x47CF98, "d" * 100)
    put(bp - 0x100, "I", 0x20011000)
    put(bp - 0xFC, "I", 0x20011010)
    put(bp - 0xBC, "i", 1)

    def hook(machine, address, size, data):
        machine.reg_write(UC_X86_REG_EIP, stub)

    u.hook_add(UC_HOOK_CODE, hook, begin=base + 0xCCAF, end=base + 0xCCAF)
    samples = []
    for first, second in itertools.product((0.0, 0.01, 0.1, 0.3, 0.7, 0.99), (0.1, 0.9)):
        put(bp - 0x110, "d", first)
        put(supplied, "d", second)
        execute(0x3784, 0x38AC)
        samples.append(
            dict(
                first=first,
                second=second,
                category=get(0x78BFC4, "i")[0],
                delay=get(0x20011000, "d")[0],
            )
        )
    result["samples"] = samples
    return result


cases = [
    run(a, b, p, window)
    for (a, b), p, window in itertools.product(
        ((1.0, 1.0), (5.0, 1.0), (1.0, 5.0), (2.0, 2.0)), (0.1, 0.3, 0.9), (30, 90)
    )
]
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="native hazard/calibration/event arithmetic",
            cases=cases,
        ),
        indent=2,
    )
)
