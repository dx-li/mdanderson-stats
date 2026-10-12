"""Bounded original DSF2.2 instruction probe; original DLL is supplied externally."""

import json
import math
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP

p = pefile.PE(sys.argv[1])
base = p.OPTIONAL_HEADER.ImageBase
u = Uc(UC_ARCH_X86, UC_MODE_32)
u.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, p.get_memory_mapped_image())
u.mem_map(0x20000000, 0x200000)
obj, model, stack, end = 0x20010000, 0x20011000, 0x20180000, 0x201F0000
u.mem_write(end, b"\xc3")
u.mem_write(end + 16, b"\xc2\x04\x00")
u.mem_write(end + 32, b"\xc2\x08\x00")
# Return a supplied float in ST0, with 4-byte or 8-byte argument cleanup.
for offset, cleanup in [(64, 4), (80, 8)]:
    u.mem_write(
        end + offset,
        b"\xdd\x05" + struct.pack("<I", end + 256) + b"\xc2" + struct.pack("<H", cleanup),
    )
u.mem_write(
    end + 96,
    b"\xdd\x1d"
    + struct.pack("<I", end + 256)
    + b"\xdd\x05"
    + struct.pack("<I", end + 256)
    + b"\xc3",
)


def pack(addr, fmt, *v):
    u.mem_write(addr, struct.pack("<" + fmt, *v))


def get(addr, fmt):
    return struct.unpack("<" + fmt, u.mem_read(addr, struct.calcsize("<" + fmt)))


state = {}
calls = []


def hook(machine, address, size, data):
    r = address - base
    if r in (0xF200, 0xF540):
        pack(end + 256, "d", state["entry"] if r == 0xF200 else state["onset"])
        u.reg_write(UC_X86_REG_EIP, end + 64)
    elif r == 0xF4B0:
        u.reg_write(UC_X86_REG_EAX, state["outcome"])
        u.reg_write(UC_X86_REG_EIP, end + 16)
    elif r in (0x1AF90, 0x1B020):
        t = get(u.reg_read(UC_X86_REG_ESP) + 4, "d")[0]
        calls.append(("hazard" if r == 0x1AF90 else "survival", t))
        pack(end + 256, "d", 0.01 if r == 0x1AF90 else math.exp(-0.01 * t))
        u.reg_write(UC_X86_REG_EIP, end + 80)
    elif r in (0x1ECAC, 0x1ECB8):
        state["math"] = math.log if r == 0x1ECAC else math.exp
        u.reg_write(UC_X86_REG_EIP, end + 96)
    elif address == end + 102:
        pack(end + 256, "d", state["math"](get(end + 256, "d")[0]))


u.hook_add(UC_HOOK_CODE, hook)
initial = u.context_save()


def run(outcome, onset, study, entry=0, delay=14, horizon=60):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x200)
    u.mem_write(model, b"\0" * 0x200)
    calls.clear()
    state.update(outcome=outcome, onset=onset, entry=entry)
    pack(model + 0x38, "i", horizon)
    pack(model + 0x44, "i", delay)
    pack(obj + 0x38, "I", model)
    pack(obj + 0x28, "d", study)
    pack(obj + 8, "d", -1e300)
    pack(stack, "I", end)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)

    try:
        u.emu_start(base + 0x1B110, end, count=100000)
    except Exception as e:
        raise RuntimeError((hex(u.reg_read(UC_X86_REG_EIP)), calls)) from e
    assert u.reg_read(UC_X86_REG_EIP) == end
    return dict(
        outcome=outcome,
        onset=onset,
        study=study,
        entry=entry,
        delay=delay,
        horizon=horizon,
        calls=list(calls),
    )


if __name__ == "__main__":
    rows = [
        run(outcome, onset, t, entry, delay)
        for outcome in (0, 1, 2)
        for onset in (10, 55, 70)
        for entry in (0, 7)
        for delay in (0, 14, 20)
        for t in (0, 5, 10, 23.999, 24, 24.001, 60, 69, 90)
    ]
    print(json.dumps(rows, indent=2))
