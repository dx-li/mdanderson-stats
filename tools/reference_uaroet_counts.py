"""Original UAROET1.8 complete/assigned/likelihood counts and escalation ceiling.

Requires pefile and Unicorn and the hash-pinned original executable. Execute
count bookkeeping and highest-assigned selection, supplying model likelihood
counts (inference is not executed). Utility reporting is disabled through its
native option. No original executable is redistributed.
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
    UC_X86_REG_ECX,
    UC_X86_REG_EDI,
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


post, model, data, records, utility = (0x20080000, 0x20090000, 0x200F0000, 0x20100000, 0x20110000)


def counts(D, E, T, patients, likelihood_counts):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x110000)
    u.mem_write(stack - 0x1000, b"\0" * 0x2000)
    for off, value in [
        (0x7254, D),
        (0x725C, E),
        (0x7260, T),
        (0x7298, model),
        (0x7294, data),
        (0x728C, utility),
    ]:
        pack(post + off, "I", value)
    pack(utility + 0x4A4, "B", 1)
    pack(data, "II", records, records + 40 * len(patients))
    for i, row in enumerate(patients):
        pack(records + 40 * i, "iiii", *row)
    pack(model + 0x108, "i" * D, *likelihood_counts)
    pack(stack, "I", end)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, post)
    u.emu_start(base + 0x52A40, base + 0x52C4B, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x52C4B
    result = {}
    for name, offset, stride, width in [
        ("assigned", 0x18D4, 4, 1),
        ("complete", 0x18E8, 4, 1),
        ("likelihood", 0x18FC, 4, 1),
        ("efficacy", 0x17D0, 20, E),
        ("toxicity", 0x1834, 16, T),
    ]:
        rows = [
            list(
                struct.unpack("<" + "i" * width, u.mem_read(post + offset + i * stride, 4 * width))
            )
            for i in range(D)
        ]
        result[name] = [r[0] for r in rows] if width == 1 else rows
    for off, value in [(0x260, data), (0x264, post), (0x1F0, D)]:
        pack(obj + off, "I", value)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_EBP, stack + 0x500)
    u.reg_write(UC_X86_REG_EDI, obj)
    u.emu_start(base + 0x5D341, base + 0x5D38D, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x5D38D
    result["highest_assigned"], result["highest_allowed"] = struct.unpack(
        "<ii", u.mem_read(obj + 0x68, 8)
    )
    return result


cases = []
for D in (2, 3, 5):
    for E, T in ((2, 2), (3, 4), (4, 3), (4, 4)):
        histories = [[]]
        for high in range(D):
            for eff, tox in ((-1, -1), (-1, T - 1), (E - 1, -1), (E - 1, T - 1)):
                histories.append([(1, 0, 0, 0), (2, high, eff, tox)])
        histories.append(
            [
                (i + 1, i % D, (i // D) % (E + 1) - 1, (i // (D * (E + 1))) % (T + 1) - 1)
                for i in range(D * (E + 1) * (T + 1))
            ]
        )
        for patients in histories:
            used = [sum(d == dose and tox >= 0 for _, d, eff, tox in patients) for dose in range(D)]
            cases.append(
                dict(
                    doses=D,
                    efficacy_levels=E,
                    toxicity_levels=T,
                    patients=patients,
                    likelihood_counts=used,
                    result=counts(D, E, T, patients, used),
                )
            )
print(
    json.dumps(
        dict(
            executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scope="Native counts and escalation ceiling; supplied model likelihood counts",
            cases=cases,
        ),
        indent=2,
    )
)
