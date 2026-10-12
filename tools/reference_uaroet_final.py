"""UAROET1.8 final selection with supplied posterior summaries and patient records.

Original highest-assigned, acceptability and greedy routines execute. Posterior
refitting is disabled via a zero iteration limit. Stop before report exports;
no native inference or RNG identity is claimed. Requires hash-pinned original PE,
pefile and Unicorn; original software is never redistributed.
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


post, data, records = 0x20080000, 0x200F0000, 0x20100000


def final(patients, utility, tox_tails, eff_tails, simulation=False):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x110000)
    u.mem_write(stack - 0x1000, b"\0" * 0x2000)
    for off, value in [
        (0x260, data),
        (0x264, post),
        (0x1F0, 3),
        (0x1FC, 4),
        (0x228, 3),
        (0x240, 1),
        (0x158, 1),
        (0x154, 0),
        (0x15C, 0),
        (0x1B4, 1),
    ]:
        pack(obj + off, "i", value)
    for off, value in [(0x238, 0.8), (0x250, 0.8), (0x170, 100), (0x178, 100)]:
        pack(obj + off, "d", value)
    pack(post + 0x7254, "i", 3)
    pack(post + 0x19E0, "d", max(utility))
    pack(post + 0x1AF0, "i", len(patients))
    pack(data, "II", records, records + 40 * len(patients))
    assigned = [0] * 3
    used = [0] * 3
    ecounts = [[0] * 4 for _ in range(3)]
    tcounts = [[0] * 4 for _ in range(3)]
    for i, (identifier, d, e, t) in enumerate(patients):
        pack(records + 40 * i, "iiii", identifier, d, e, t)
        assigned[d] += 1
        used[d] += int(t >= 0)
        if e >= 0 and t >= 0:
            ecounts[d][e] += 1
            tcounts[d][t] += 1
    pack(obj + 0x14, "iii", *assigned)
    pack(post + 0x18FC, "iii", *used)
    for d in range(3):
        pack(post + 0x17D0 + d * 20, "iiii", *ecounts[d])
        pack(post + 0x1834 + d * 16, "iiii", *tcounts[d])
        pack(post + 0x18E8 + d * 4, "i", sum(ecounts[d]))
        for off, value in [
            (0x1220, utility[d]),
            (0x1270, 1 / 3),
            (0x12C0, tox_tails[d]),
            (0x1310, eff_tails[d]),
        ]:
            pack(post + off + d * 8, "d", value)
    pack(stack, "I", end)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    entry, stop = (0x5EA70, 0x5EB4A) if simulation else (0x5EC80, 0x5ED74)
    u.emu_start(base + entry, base + stop, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + stop
    return dict(
        selected=struct.unpack("<i", u.mem_read(obj, 4))[0],
        highest_allowed=struct.unpack("<i", u.mem_read(obj + 0x6C, 4))[0],
        status=list(struct.unpack("<iii", u.mem_read(obj + 0xCC, 12))),
    )


cases = []
for high in range(3):
    for e, t in ((-1, -1), (-1, 3), (0, -1), (0, 3), (3, 0)):
        patients = [(1, 0, 3, 0), (2, high, e, t)]
        for utility in ([20, 40, 60], [60, 40, 20], [40, 40, 40]):
            for tox, eff in (([0] * 3, [0] * 3), ([0.9] * 3, [0] * 3), ([0] * 3, [0.9] * 3)):
                inputs = dict(patients=patients, utility=utility, tox_tails=tox, eff_tails=eff)
                for simulation in (False, True):
                    cases.append(
                        dict(
                            inputs=inputs,
                            simulation=simulation,
                            result=final(**inputs, simulation=simulation),
                        )
                    )
print(
    json.dumps(
        dict(
            executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scope=(
                "Native final selection with supplied posterior summaries and zero refit iterations"
            ),
            cases=cases,
        ),
        indent=2,
    )
)
