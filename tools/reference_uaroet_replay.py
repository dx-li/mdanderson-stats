"""UAROET1.8 saved-history cohort reconstruction, fresh selector state.

Executes original5e150 and patient data getters. No posterior summaries, RNG,
assignment or fit is evaluated. Requires the original hash-pinned executable,
pefile and Unicorn. No original executable is redistributed.
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


data, records = 0x200F0000, 0x20100000


def replay(doses, first, new, old):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x110000)
    u.mem_write(stack - 0x1000, b"\0" * 0x2000)
    for off, value in [
        (0x260, data),
        (0x1F0, 3),
        (0xAC, first),
        (0xB0, new),
        (0xB4, old),
        (0x1E4, 0),
    ]:
        pack(obj + off, "i", value)
    pack(data, "II", records, records + 40 * len(doses))
    for i, d in enumerate(doses):
        pack(records + 40 * i, "iiii", i + 1, d, -1, -1)
    pack(stack, "I", end)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    u.emu_start(base + 0x5E150, end, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == end
    return dict(
        assigned=list(struct.unpack("<iii", u.mem_read(obj + 0x14, 12))),
        first_seen=list(struct.unpack("<iii", u.mem_read(obj + 0x50, 12))),
        total=struct.unpack("<i", u.mem_read(obj + 0x1B4, 4))[0],
        last=struct.unpack("<i", u.mem_read(obj + 0x1E0, 4))[0],
        new_open=bool(u.mem_read(obj + 0xB8, 1)[0]),
        old_open=bool(u.mem_read(obj + 0xB9, 1)[0]),
        new_start=struct.unpack("<i", u.mem_read(obj + 0xBC, 4))[0],
        old_start=struct.unpack("<i", u.mem_read(obj + 0xC0, 4))[0],
        incomplete=bool(u.mem_read(obj + 0xC5, 1)[0]),
    )


cases = []
for sizes in ((1, 1, 1), (3, 2, 3), (2, 3, 1)):
    for length in range(6):
        for doses in itertools.product(range(3), repeat=length):
            cases.append(
                dict(
                    doses=doses,
                    first=sizes[0],
                    new=sizes[1],
                    old=sizes[2],
                    result=replay(doses, *sizes),
                )
            )
print(
    json.dumps(
        dict(
            executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scope="Fresh-state native saved-history cohort reconstruction",
            cases=cases,
        ),
        indent=2,
    )
)
