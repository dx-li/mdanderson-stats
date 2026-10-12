"""Original lognormal quantile/variance calibration, with host CRT math services."""

import json
import math
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP

p = pefile.PE(sys.argv[1])
base = p.OPTIONAL_HEADER.ImageBase
u = Uc(UC_ARCH_X86, UC_MODE_32)
u.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, p.get_memory_mapped_image())
u.mem_map(0x20000000, 0x200000)
u.mem_map(0, 0x1000)
obj, stack, end = 0x20010000, 0x20180000, 0x201F0000
u.mem_write(end, b"\xc3")
u.mem_write(
    end + 32,
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


mathcalls = {0x1ECAC: math.log, 0x1ECB8: math.exp, 0x1ECBE: math.sqrt}
state = {}
trail = []


def hook(machine, address, size, data):
    trail.append(address - base)
    if len(trail) > 20:
        trail.pop(0)
    if address - base in mathcalls:
        state["func"] = mathcalls[address - base]
        u.reg_write(UC_X86_REG_EIP, end + 32)
    elif address == end + 38:
        try:
            value = state["func"](get(end + 256, "d")[0])
        except OverflowError:
            value = float("inf")
        pack(end + 256, "d", value)


u.hook_add(UC_HOOK_CODE, hook)
initial = u.context_save()


def run(prob, horizon):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x1000)
    pack(obj + 0x18, "dd", horizon, prob)
    pack(stack, "I", end)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    try:
        u.emu_start(base + 0x10DB0, end, count=100000)
    except Exception as e:
        raise RuntimeError([hex(v) for v in trail]) from e
    assert u.reg_read(UC_X86_REG_EIP) == end
    return dict(prob=prob, horizon=horizon, mu=get(obj + 8, "d")[0], sigma=get(obj + 16, "d")[0])


if __name__ == "__main__":
    print(
        json.dumps(
            [
                run(prob, horizon)
                for prob in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)
                for horizon in (30, 60, 116)
            ],
            indent=2,
        )
    )
