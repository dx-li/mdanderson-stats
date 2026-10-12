"""Bounded original UAROET1.8 hybrid-option parser reference generator.

Requires pefile and Unicorn. Original executable is supplied, never distributed.
Host exp supplies CRT mathematics; native security-cookie check is bypassed.
Executes original string comparison and option assignment only.
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
    UC_X86_REG_EIP,
    UC_X86_REG_ESI,
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


def stop_unmatched(machine, address, size, data):
    if address == base + 0x24E4C:
        u.emu_stop()


u.hook_add(UC_HOOK_CODE, stop_unmatched)


def keyword(word, value, initial_flag=False):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x60000)
    frame = stack + 0x500
    u.mem_write(end + 0x1000, word.encode() + b"\0")
    pack(frame + 0x30, "I", end + 0x1000)
    pack(frame + 0x40, "II", len(word), len(word) + 1)
    pack(obj + 0x1E8DC, "B", initial_flag)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_EBP, frame)
    u.reg_write(UC_X86_REG_EDI, obj)
    u.reg_write(UC_X86_REG_ESI, value)
    u.emu_start(base + 0x24E22, base + 0x251EE, count=100000)
    return dict(
        matched=u.reg_read(UC_X86_REG_EIP) == base + 0x251EE,
        flag=bool(u.mem_read(obj + 0x1E8DC, 1)[0]),
    )


cases = []
for word in (
    "UseHybridGreedyArandAssignment",
    "UseHybridGreedyARandAssignment",
    "UseHybridGreedyArandAlgorithm",
    "useHybridGreedyArandAssignment",
):
    for value in (0, 1):
        for initial_flag in (False, True):
            cases.append(
                dict(
                    word=word,
                    value=value,
                    initial_flag=initial_flag,
                    result=keyword(word, value, initial_flag),
                )
            )
print(
    json.dumps(
        dict(executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases), indent=2
    )
)
