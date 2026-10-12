"""Bounded original UAROET1.8 posterior-summary reference generator.

Requires pefile and Unicorn. Original executable is supplied, never distributed.
Host exp supplies CRT mathematics; native security-cookie check is bypassed.
Supplied joint/marginal probabilities and utility scores replace the probability
model callback. Original summary packing, tail indicators, best-utility ties and
good-outcome summation execute. Inference is not reproduced.
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


model = 0x20080000
vector = 0x20100000
values = 0x20101000
u.mem_write(end + 32, b"\xdd\x05" + struct.pack("<I", end + 256) + b"\xc2\x04\x00")


def supplied_probabilities(machine, address, size, data):
    if address == base + 0x36C40:
        pack(end + 256, "d", 0.0)
        u.reg_write(UC_X86_REG_EIP, end + 32)


u.hook_add(UC_HOOK_CODE, supplied_probabilities)


def summaries(joint, utility, bad_tox, good_eff, tox_limit, eff_limit, good_mask):
    j = np.array(joint)
    D, E, T = j.shape
    e = j.sum(2)
    t = j.sum(1)
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x60000)
    u.mem_write(model, b"\0" * 0x60000)
    for off, v in [
        (0x7298, model),
        (0x7254, D),
        (0x725C, E),
        (0x7260, T),
        (0x69C0, 13),
        (0x6950, bad_tox),
        (0x6960, good_eff),
    ]:
        pack(obj + off, "i", v)
    pack(obj + 0x6958, "d", tox_limit)
    pack(obj + 0x6968, "d", eff_limit)
    for d in range(D):
        pack(model + 0x1C0 + d * 40, "d" * E, *e[d])
        pack(model + 0x288 + d * 32, "d" * T, *t[d])
        pack(model + 0xE40 + d * 8, "d", utility[d])
        for i in range(E):
            for k in range(T):
                pack(model + 0x8A0 + (d * 16 + i * 4 + k) * 8, "d", j[d, i, k])
    for i in range(E):
        for k in range(T):
            pack(obj + 0x6BE0 + (i * 4 + k) * 4, "i", good_mask[i][k])
    pack(vector, "III", 0, values, D * 13)
    pack(stack, "III", end, 0, vector)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    u.emu_start(base + 0x40BC0, base + 0x4101A, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x4101A
    return (
        np.array(struct.unpack("<" + "d" * (D * 13), u.mem_read(values, D * 13 * 8)))
        .reshape(D, 13)
        .tolist()
    )


cases = []
for D in (2, 3, 5):
    for E, T in ((2, 2), (3, 4), (4, 3)):
        j = np.ones((D, E, T)) / (E * T)
        good_mask = [[int(e >= 1 and t < T - 1) for t in range(T)] for e in range(E)]
        for utility in ([40] * D, list(range(10, 10 + D)), list(range(10 + D, 10, -1))):
            for delta in (-0.01, 0.0, 0.01):
                tx = sum(j.sum(1)[0, 1:])
                ef = sum(j.sum(2)[0, 1:])
                kw = dict(
                    joint=j.tolist(),
                    utility=utility,
                    bad_tox=1,
                    good_eff=1,
                    tox_limit=tx + delta,
                    eff_limit=ef + delta,
                    good_mask=good_mask,
                )
                cases.append(dict(inputs=kw, result=summaries(**kw)))
print(
    json.dumps(
        dict(executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases), indent=2
    )
)
