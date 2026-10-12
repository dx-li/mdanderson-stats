"""Bounded original UAROET1.8 patient likelihood reference generator.

Requires pefile and Unicorn. Original executable is supplied, never distributed.
Host exp supplies CRT mathematics; native security-cookie check is bypassed.
Joint probabilities are supplied; native probability fitting is not executed.
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


def logarithm(machine, address, size, data):
    if address == base + 0xA3970:
        v = struct.unpack("<d", struct.pack("<Q", u.reg_read(UC_X86_REG_XMM0) & (2**64 - 1)))[0]
        u.reg_write(UC_X86_REG_XMM0, struct.unpack("<Q", struct.pack("<d", math.log(v)))[0])
        u.reg_write(UC_X86_REG_EIP, end + 16)


u.hook_add(UC_HOOK_CODE, logarithm)


def likelihood(joint, patients, clock=100.0, partial=True):
    D = len(joint)
    E = len(joint[0])
    T = len(joint[0][0])
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x60000)
    for off, v in [(0x58DC8, D), (0x58DCC, E), (0x58DD0, T), (0x58DE8, len(patients) - 1)]:
        pack(obj + off, "i", v)
    pack(obj + 0x145, "B", int(partial))
    pack(obj + 0x58DD8, "d", clock)
    pack(obj + 0x1B0, "d", 1.0)
    for d in range(D):
        for e in range(E):
            for t in range(T):
                p = joint[d][e][t]
                pack(obj + 0x8A0 + (d * 16 + e * 4 + t) * 8, "d", p)
                pack(obj + 0xB20 + (d * 20 + e * 4 + t) * 8, "d", math.log(p))
    for i, patient in enumerate(patients):
        dose, eff, tox, entry, effdelay, toxdelay = patient
        pack(obj + 0x3B8D8 + i * 12, "iii", dose, eff, tox)
        pack(obj + 0xF58 + i * 24, "ddd", entry, effdelay, toxdelay)
    pack(stack, "I", end)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    u.emu_start(base + 0x3A8A0, end + 6, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == end + 6
    return dict(
        likelihood=struct.unpack("<d", u.mem_read(end + 256, 8))[0],
        log_likelihood=struct.unpack("<d", u.mem_read(obj + 0x128, 8))[0],
        used=struct.unpack("<i", u.mem_read(obj + 0xF8, 4))[0],
    )


cases = []
for D in (1, 3):
    for E, T in ((2, 2), (3, 4), (4, 3)):
        joint = []
        for d in range(D):
            weights = [[float((e + 1) * (t + 2) + d + e * t) for t in range(T)] for e in range(E)]
            total = sum(map(sum, weights))
            joint.append([[v / total for v in row] for row in weights])
        for d in range(D):
            for e, t in [(-1, -1), (-1, T - 1), (E - 1, -1), (0, 0), (E - 1, T - 1)]:
                for clock in (0.0, 1.0, 2.0, 3.0):
                    for partial in (False, True):
                        patients = [(d, e, t, 1.0, 1.0, 2.0)]
                        cases.append(
                            dict(
                                joint=joint,
                                patients=patients,
                                clock=clock,
                                partial=partial,
                                result=likelihood(joint, patients, clock, partial),
                            )
                        )
print(
    json.dumps(
        dict(executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=cases), indent=2
    )
)
