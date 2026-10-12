"""Native PRT1.0 arrival generation/sorting with supplied unit uniforms."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EBP,
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


def run(values, mean):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp, stub, supplied = 0x20100000, 0x201F0000, 0x201F0100
    u.mem_write(stub, b"\xdd\x05" + struct.pack("<I", supplied) + b"\xc3")
    u.mem_write(0x78BD68, struct.pack("<d", mean))
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_ESI, 800)
    u.reg_write(UC_X86_REG_EDI, 0xCCD648)
    u.reg_write(UC_X86_REG_FPCW, 0x37F)
    index = 0

    def hook(machine, address, size, data):
        nonlocal index
        u.mem_write(supplied, struct.pack("<d", values[index]))
        index += 1
        machine.reg_write(UC_X86_REG_EIP, stub)

    u.hook_add(UC_HOOK_CODE, hook, begin=base + 0xCCAF, end=base + 0xCCAF)
    u.emu_start(base + 0x2E8C, base + 0x2EDB, count=10_000_000)
    if index != 800 or u.reg_read(UC_X86_REG_EIP) != base + 0x2EDB:
        raise RuntimeError(
            f"incomplete native arrival generation {index} {u.reg_read(UC_X86_REG_EIP):x}"
        )
    return struct.unpack("<" + "d" * 800, u.mem_read(0xCCD648, 6400))


cases = []
for mean in (1.0, 7.5, 45.0):
    for seed in (0, 1):
        values = np.random.default_rng(seed).random(800).tolist()
        cases.append(dict(mean=mean, seed=seed, uniforms=values, arrivals=run(values, mean)))
print(
    json.dumps(
        dict(
            executable_sha256=digest, scope="native800-uniform arrival generation/sort", cases=cases
        ),
        indent=2,
    )
)
