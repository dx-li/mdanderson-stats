"""Execute bounded original CRM Suite x86 branches with explicit service hooks.

Usage: python tools/reference_crm_native_branches.py ORIGINAL_KERNEL_DLL
Requires dnfile and unicorn; original binary is not redistributed. Supplied
posterior flags, matrix access and a fixed exponential draw isolate controller
semantics. This is not posterior, RNG or full native simulator verification.
"""

import hashlib
import json
import struct
import sys

import dnfile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EAX,
    UC_X86_REG_EBP,
    UC_X86_REG_EBX,
    UC_X86_REG_ECX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESP,
)

PATH = sys.argv[1]
assert (
    hashlib.sha256(open(PATH, "rb").read()).hexdigest()
    == "8fc20c5f514d0c0a68877a55126ff97f4a43a88e3ded403adfe982b80bb501e7"
)
p = dnfile.dnPE(PATH)
base = p.OPTIONAL_HEADER.ImageBase
RET = 0x700000


def env():
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, 0x100000)
    u.mem_write(base, p.get_memory_mapped_image())
    u.mem_map(0x600000, 0x200000)
    u.reg_write(UC_X86_REG_ESP, 0x680000)
    return u


def put(u, a, *v):
    u.mem_write(a, struct.pack("<" + "I" * len(v), *v))


def get(u, a):
    return struct.unpack("<I", u.mem_read(a, 4))[0]


def raw(current, recommended, events, observed, target):
    u = env()
    sp = u.reg_read(UC_X86_REG_ESP)
    put(u, sp, RET, 0x610000, 0x620000)
    u.mem_write(sp + 12, struct.pack("<d", target))
    put(u, sp + 20, recommended, 0)
    put(u, 0x610000, 0x611000, 0x611048, 0x611048)
    put(u, 0x611000, current)

    def hook(u, a, s, d):
        if a == RET:
            u.emu_stop()
        if a == base + 0x14350:
            sp = u.reg_read(UC_X86_REG_ESP)
            dose = get(u, sp + 4)
            col = get(u, sp + 8)
            put(u, 0x630000, events[dose] if col == 0 else observed[dose] - events[dose])
            u.reg_write(UC_X86_REG_EAX, 0x630000)
            u.reg_write(UC_X86_REG_EIP, get(u, sp))
            u.reg_write(UC_X86_REG_ESP, sp + 12)

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + 0x33040, RET, count=10000)
    return u.reg_read(UC_X86_REG_EAX)


def flags(stop, wait):
    u = env()
    bp = 0x650000
    out = 0x660000
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_EBX, out)
    put(u, bp - 0xFC, 3)
    u.mem_write(bp - 0xF8, bytes([stop, wait]))
    put(u, bp - 0x238, 1)
    u.reg_write(UC_X86_REG_ECX, 1)

    def hook(u, a, s, d):
        if a == base + 0x32448:
            u.emu_stop()

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + 0x323E6, base + 0x32449, count=1000)
    return get(u, out), get(u, out + 4)


rows = []
for x in [
    (0, 1, [1, 0, 0, 0], [3, 3, 3, 0], 0.3),
    (1, 2, [0, 1, 0, 0], [3, 3, 3, 0], 0.3),
    (0, 1, [0, 0, 0, 0], [0, 0, 0, 0], 0.3),
    (0, 1, [1, 0, 0, 0], [4, 3, 3, 0], 0.25),
    (2, 1, [1, 1, 1, 0], [3, 3, 3, 0], 0.3),
    (0, 3, [1, 0, 0, 0], [3, 3, 3, 0], 0.3),
    (0, 3, [0, 1, 0, 0], [3, 3, 3, 0], 0.3),
    (0, 3, [0, 0, 0, 0], [0, 0, 0, 0], 0.3),
    (0, 3, [1, 0, 0, 0], [4, 3, 3, 0], 0.25),
    (2, 0, [1, 1, 1, 0], [3, 3, 3, 0], 0.3),
]:
    rows.append({"input": x, "native_dose": raw(*x)})
result = {
    "sha256": hashlib.sha256(open(PATH, "rb").read()).hexdigest(),
    "raw": rows,
    "flags": {f"{a}{b}": flags(a, b) for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]},
}


def wait_advance(clock, draw):
    u = env()
    sp = u.reg_read(UC_X86_REG_ESP)
    u.reg_write(UC_X86_REG_EAX, 1)
    put(u, sp + 0x138, 0x620000)
    put(u, sp + 0x16C, 1)
    u.mem_write(sp + 0x144, struct.pack("<d", clock))
    u.mem_write(0x620010, struct.pack("<d", 7.0))
    u.mem_write(0x630000, struct.pack("<d", draw))
    # Replace RNG service in EMULATED memory only by fld fixed double; ret.
    u.mem_write(base + 0x1CA30, b"\xdd\x05" + struct.pack("<I", 0x630000) + b"\xc3")

    def hook(u, a, s, d):
        if a == base + 0x2D310:
            u.emu_stop()

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + 0x2D40E, base + 0x2D45E, count=1000)
    return struct.unpack("<d", u.mem_read(sp + 0x164, 8))[0]


result["wait_clock"] = wait_advance(10.0, 2.5)
print(json.dumps(result, indent=2))
