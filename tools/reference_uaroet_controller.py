"""Bounded original UAROET1.8 cohort controller reference generator.

Requires pefile and Unicorn. Original executable is supplied, never distributed.
Host exp supplies CRT mathematics; native security-cookie check is bypassed.
Posterior acceptability/status summaries are supplied. Refitting is disabled
for those supplied summaries. Native greedy choice and cohort bookkeeping run;
inference, randomization and highest-tried computation are not claimed.
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
    UC_X86_REG_EAX,
    UC_X86_REG_ECX,
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
statuses = [1, 1, 1]


def controller_services(machine, address, size, data):
    if address == base + 0x5CF30:
        pack(obj + 0xCC, "iii", *statuses)
        pack(obj + 0xC8, "i", sum(v == 1 for v in statuses))
        pack(obj + 0x6C, "i", 2)
        u.reg_write(UC_X86_REG_EIP, end + 16)
    elif address == base + 0x55BA0:
        u.reg_write(UC_X86_REG_EAX, 0)
        u.reg_write(UC_X86_REG_EIP, end + 16)
    elif address == base + 0x5E774:
        u.emu_stop()


u.hook_add(UC_HOOK_CODE, controller_services)


def trial(first=3, new=2, old=3, utility_sequence=None, status_sequence=None):
    global statuses
    u.mem_write(obj, b"\0" * 0x60000)
    u.mem_write(post, b"\0" * 0x8000)
    for off, val in [
        (0x264, post),
        (0x1F0, 3),
        (0x6C, 2),
        (0xAC, first),
        (0xB0, new),
        (0xB4, old),
        (0x1E4, 0),
    ]:
        pack(obj + off, "i", val)
    pack(obj + 0xC4, "B", 1)
    pack(post + 0x7254, "i", 3)
    treated = [0, 0, 0]
    trace = []
    utility_sequence = utility_sequence or [[20, 60, 40]] * 10
    status_sequence = status_sequence or [[1, 1, 1]] * len(utility_sequence)
    for values, conditions in zip(utility_sequence, status_sequence):
        statuses = conditions
        u.context_restore(initial)
        pack(obj + 0x14, "iii", *treated)
        pack(post + 0x1220, "ddd", *values)
        pack(post + 0x1AF0, "i", sum(treated))
        pack(stack, "I", end)
        u.reg_write(UC_X86_REG_ESP, stack)
        u.reg_write(UC_X86_REG_ECX, obj)
        u.emu_start(base + 0x5E520, end, count=100000)
        assert u.reg_read(UC_X86_REG_EIP) in (end, base + 0x5E774)
        selected = struct.unpack("<i", u.mem_read(obj, 4))[0]
        trace.append(
            dict(
                selected=selected,
                stop=bool(u.mem_read(obj + 0x10, 1)[0]),
                new_open=bool(u.mem_read(obj + 0xB8, 1)[0]),
                old_open=bool(u.mem_read(obj + 0xB9, 1)[0]),
            )
        )
        if selected < 0:
            break
        treated[selected] += 1
    return trace


cases = []
for first, new, old in ((1, 1, 1), (3, 1, 1), (3, 2, 3), (2, 3, 1), (1, 2, 4)):
    for utility_sequence in ([[20, 60, 40]] * 10, [[60, 20, 40], [20, 60, 40], [40, 20, 60]] * 4):
        for status_sequence in (
            [[1, 1, 1]] * 12,
            [[-2, -2, -2]] * 12,
            [[1, 1, 1], [1, -2, 1], [-2, 1, 1]] * 4,
        ):
            kw = dict(
                first=first,
                new=new,
                old=old,
                utility_sequence=utility_sequence,
                status_sequence=status_sequence,
            )
            cases.append(dict(inputs=kw, trace=trial(**kw)))
print(
    json.dumps(
        dict(executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases), indent=2
    )
)
