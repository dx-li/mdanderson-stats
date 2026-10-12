"""UAROET1.8 native outcome-delay rounding with supplied uniform variates.

Runs61b66–61bef, native2ac60 arithmetic and native a3000 floor, plus actual
patient setters. Only the uniform RNG return is supplied. No trial simulation,
posterior fitting or random-stream identity is claimed.
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
    UC_X86_REG_EAX,
    UC_X86_REG_EBP,
    UC_X86_REG_EBX,
    UC_X86_REG_EDI,
    UC_X86_REG_EIP,
    UC_X86_REG_ESI,
    UC_X86_REG_ESP,
    UC_X86_REG_FPCW,
    UC_X86_REG_MXCSR,
)

path = Path(sys.argv[1])
if (
    hashlib.sha256(path.read_bytes()).hexdigest()
    != "b953affb6cd3968c389d5a35020e594cd25fef3ff61825db8aa282d8216a3eaf"
):
    raise ValueError("original executable hash mismatch")
p = pefile.PE(str(path))
base = p.OPTIONAL_HEADER.ImageBase
u = Uc(UC_ARCH_X86, UC_MODE_32)
u.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, p.get_memory_mapped_image())
u.mem_map(0x20000000, 0x200000)
obj, data, records, stack, end = 0x20010000, 0x20080000, 0x20090000, 0x20180000, 0x201F0000
u.mem_write(end, b"\xdd\x05" + struct.pack("<I", end + 256) + b"\xc3")
u.mem_write(base + 0x1045B0, struct.pack("<I", 1))
values = []
index = 0


def pack(address, fmt, *v):
    u.mem_write(address, struct.pack("<" + fmt, *v))


def hook(machine, address, size, context):
    global index
    if address == base + 0x81B00:
        pack(end + 256, "d", values[index])
        index += 1
        u.reg_write(UC_X86_REG_EIP, end)


u.hook_add(UC_HOOK_CODE, hook)
initial = u.context_save()


def delay(e_window, t_window, uniforms):
    global values, index
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x10000)
    u.mem_write(stack - 0x1000, b"\0" * 0x2000)
    pack(obj + 0xEF0, "I", data)
    pack(data, "II", records, records + 40)
    pack(obj + 0x248, "dd", *e_window)
    pack(obj + 0x238, "dd", *t_window)
    values = uniforms
    index = 0
    u.reg_write(UC_X86_REG_EDI, obj)
    u.reg_write(UC_X86_REG_ESI, 0)
    u.reg_write(UC_X86_REG_EBP, stack + 0x500)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_FPCW, 0x37F)
    u.reg_write(UC_X86_REG_MXCSR, 0x1F80)
    u.emu_start(base + 0x61B66, base + 0x61BEF, count=10000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x61BEF and index == 2
    return struct.unpack("<dd", u.mem_read(records + 0x18, 16))


cases = []
for e, t, uniforms in itertools.product(
    ((0.1, 0.2), (0.1, 1.2), (0.5, 1.5), (1.0, 42.0), (2.3, 2.3)),
    ((0.1, 0.2), (0.5, 1.5), (1.0, 42.0)),
    ((0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 1.0)),
):
    cases.append(
        dict(efficacy_window=e, toxicity_window=t, uniforms=uniforms, result=delay(e, t, uniforms))
    )
# Arrival advancement executes the original virtual-distribution call and loop.
u.mem_write(end + 32, b"\xdd\x05" + struct.pack("<I", end + 320) + b"\xc3")
u.mem_write(end + 128, b"\xc3")
arrival_values = []
arrival_index = 0


def arrival_hook(machine, address, size, context):
    global arrival_index
    if address == end + 128:
        pack(end + 320, "d", arrival_values[arrival_index])
        arrival_index += 1
        u.reg_write(UC_X86_REG_EIP, end + 32)


u.hook_add(UC_HOOK_CODE, arrival_hook)


def advance(previous, wait_until, intervals, enabled):
    global arrival_values, arrival_index
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x10000)
    u.mem_write(stack - 0x1000, b"\0" * 0x2000)
    provider, table = data + 256, data + 512
    pack(obj + 0x258, "I", provider)
    pack(provider, "I", table)
    pack(table, "I", end + 128)
    pack(obj + 0x1F8, "B", int(enabled))
    frame = stack + 0x500
    pack(frame - 0x68, "i", 1)
    pack(frame - 0x78, "d", previous)
    pack(frame - 0x80, "d", wait_until)
    pack(frame - 0x5C, "i", 7)
    pack(frame - 0x70, "i", 2)
    pack(frame - 0x54, "i", 5)
    arrival_values, arrival_index = intervals, 0
    u.reg_write(UC_X86_REG_EDI, obj)
    u.reg_write(UC_X86_REG_EAX, 1)
    u.reg_write(UC_X86_REG_EBX, 5)
    u.reg_write(UC_X86_REG_ESI, 7)
    u.reg_write(UC_X86_REG_EBP, frame)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_FPCW, 0x37F)
    u.emu_start(base + 0x6A812, base + 0x6A690, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x6A690
    return dict(
        clock=struct.unpack("<d", u.mem_read(frame - 0x78, 8))[0],
        discarded=struct.unpack("<i", u.mem_read(frame - 0x70, 4))[0] - 2,
        variates_used=arrival_index,
    )


arrival_cases = []
for previous, wait_until in ((0.0, 0.0), (0.0, 1.0), (10.0, 12.0), (0.1, 0.3)):
    for interval, enabled in itertools.product((0.2, 0.5, 1.0, 2.0), (False, True)):
        values = [interval] * 20
        arrival_cases.append(
            dict(
                previous=previous,
                wait_until=wait_until,
                intervals=values,
                enabled=enabled,
                result=advance(previous, wait_until, values, enabled),
            )
        )
print(
    json.dumps(
        dict(
            executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scope="Native rounded uniform delays with supplied variates",
            cases=cases,
            arrival_cases=arrival_cases,
        ),
        indent=2,
    )
)
