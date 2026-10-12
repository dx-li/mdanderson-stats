"""Bounded original UAROET1.8 acceptability reference generator.

Requires pefile and Unicorn. Original executable is supplied, never distributed.
Host exp supplies CRT mathematics; native security-cookie check is bypassed.
Posterior summaries are supplied; inference and randomization are not executed.
"""

import hashlib
import itertools
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


post = 0x20080000


def screen(
    tails,
    bad_tox,
    bad_eff,
    eff_tails=None,
    highest=2,
    min_bad_eff=1,
    min_total_tox=0,
    utility=(20, 40, 60),
    best=(0.1, 0.2, 0.7),
    threshold=0.8,
    min_best=0,
    delta=100,
    filter_utility=True,
    min_observed=0,
    observed=(3, 3, 3),
):
    D = len(tails)
    eff_tails = [0] * D if eff_tails is None else eff_tails
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x60000)
    u.mem_write(post, b"\0" * 0x8000)
    u.mem_write(stack - 0x1000, b"\0" * 0x2000)
    for off, value in [
        (0x6C, highest),
        (0x1F0, D),
        (0x1FC, 4),
        (0x228, 3),
        (0x240, 1),
        (0x158, min_bad_eff),
        (0x154, min_total_tox),
        (0x15C, min_observed),
        (0x264, post),
    ]:
        pack(obj + off, "i", value)
    for off, value in [(0x238, threshold), (0x250, threshold), (0x188, delta), (0x190, min_best)]:
        pack(obj + off, "d", value)
    pack(obj + 0x164, "B", int(filter_utility))
    pack(post + 0x7254, "i", D)
    pack(post + 0x19E0, "d", max(utility))
    for i in range(D):
        for off, value in [
            (0x12C0, tails[i]),
            (0x1310, eff_tails[i]),
            (0x1220, utility[i]),
            (0x1270, best[i]),
        ]:
            pack(post + off + i * 8, "d", value)
        pack(post + 0x1834 + i * 16 + 12, "i", bad_tox[i])
        pack(post + 0x17D0 + i * 20, "i", bad_eff[i])
        pack(post + 0x18E8 + i * 4, "i", observed[i])
    u.reg_write(UC_X86_REG_EDI, obj)
    u.reg_write(UC_X86_REG_EDX, post)
    u.reg_write(UC_X86_REG_ESI, 0)
    u.reg_write(UC_X86_REG_EBP, stack + 0x500)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.emu_start(base + 0x5D40D, base + 0x5D6A5, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x5D6A5
    return struct.unpack("<" + "i" * D, u.mem_read(obj + 0xCC, D * 4))


cases = []
for tails in itertools.product((0.2, 0.8, 0.9), repeat=3):
    for bad_tox in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        for highest in (0, 2):
            for eff_tails, bad_eff in [([0, 0, 0], [0, 0, 0]), ([0.9, 0.8, 0.9], [0, 1, 2])]:
                kw = dict(
                    tails=tails,
                    bad_tox=bad_tox,
                    bad_eff=bad_eff,
                    eff_tails=eff_tails,
                    highest=highest,
                )
                cases.append(dict(inputs=kw, status=screen(**kw)))
for delta, min_best, minimum, observed in itertools.product(
    (0, 20, 40, 100), (0, 0.2, 0.3), (0, 3), ((0, 0, 0), (2, 3, 4))
):
    kw = dict(
        tails=[0] * 3,
        bad_tox=[0] * 3,
        bad_eff=[0] * 3,
        delta=delta,
        min_best=min_best,
        min_observed=minimum,
        observed=observed,
    )
    cases.append(dict(inputs=kw, status=screen(**kw)))
for minimum in (0, 1, 2, 3):
    kw = dict(tails=[0.9] * 3, bad_tox=[1, 0, 0], bad_eff=[0] * 3, min_total_tox=minimum)
    cases.append(dict(inputs=kw, status=screen(**kw)))
print(
    json.dumps(
        dict(executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases), indent=2
    )
)
