"""Execute original final-duration reduction and final-analysis clock offset."""

import json
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EDI, UC_X86_REG_EIP, UC_X86_REG_ESI, UC_X86_REG_ESP

p = pefile.PE(sys.argv[1])
base = p.OPTIONAL_HEADER.ImageBase
u = Uc(UC_ARCH_X86, UC_MODE_32)
u.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, p.get_memory_mapped_image())
u.mem_map(0x20000000, 0x200000)
obj, result, stack, end = 0x20010000, 0x20011000, 0x20180000, 0x201F0000
u.mem_write(end, b"\xdd\x05" + struct.pack("<I", end + 256) + b"\xc2\x04\x00")


def pack(addr, fmt, *v):
    u.mem_write(addr, struct.pack("<" + fmt, *v))


def get(addr, fmt):
    return struct.unpack("<" + fmt, u.mem_read(addr, struct.calcsize("<" + fmt)))


state = {}


def hook(machine, address, size, data):
    if address - base in (0xF200, 0xF540):
        index = get(u.reg_read(UC_X86_REG_ESP) + 4, "i")[0]
        value = state["entries" if address - base == 0xF200 else "times"][index]
        pack(end + 256, "d", value)
        u.reg_write(UC_X86_REG_EIP, end)


u.hook_add(UC_HOOK_CODE, hook)
initial = u.context_save()


def run(entries, times):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x1000)
    u.mem_write(result, b"\0" * 0x1000)
    u.mem_write(stack - 0x1000, b"\0" * 0x2000)
    state.update(entries=entries, times=times)
    pack(obj + 0x1C8, "I", result)
    u.reg_write(UC_X86_REG_ESI, obj)
    u.reg_write(UC_X86_REG_EDI, len(entries))
    u.reg_write(UC_X86_REG_ESP, stack)
    u.emu_start(base + 0x14AAA, base + 0x14B9E, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x14B9E
    return dict(
        entries=entries,
        times=times,
        duration=get(result + 0x20, "d")[0],
        analysis_time=get(u.reg_read(UC_X86_REG_ESP), "d")[0],
    )


if __name__ == "__main__":
    rows = [
        run(entries, times)
        for entries in ([0], [0, 7], [0, 7, 8])
        for times in ([10] * len(entries), [60] * len(entries), list(range(1, len(entries) + 1)))
    ]
    print(json.dumps(rows, indent=2))
