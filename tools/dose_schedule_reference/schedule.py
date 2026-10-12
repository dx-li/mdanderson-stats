"""Execute original assignment-to-administration ledger generation."""

import json
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP

p = pefile.PE(sys.argv[1])
base = p.OPTIONAL_HEADER.ImageBase
u = Uc(UC_ARCH_X86, UC_MODE_32)
u.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, p.get_memory_mapped_image())
u.mem_map(0x20000000, 0x200000)
obj, patient, model, stack, end = 0x20010000, 0x20011000, 0x20012000, 0x20180000, 0x201F0000
initial = u.context_save()


def pack(addr, fmt, *v):
    u.mem_write(addr, struct.pack("<" + fmt, *v))


def get(addr, fmt):
    return struct.unpack("<" + fmt, u.mem_read(addr, struct.calcsize("<" + fmt)))


def run(admins, schedule, interval, dose):
    u.context_restore(initial)
    u.mem_write(obj, b"\0" * 0x1000)
    u.mem_write(patient, b"\0" * 0x1000)
    u.mem_write(model, b"\0" * 0x1000)
    pack(obj + 4, "II", patient, patient + 0x278)
    pack(obj + 0x1C, "i", admins)
    pack(obj + 0x28, "I", model)
    pack(model + 0x40, "i", interval)
    pack(patient + 8, "i", dose)
    pack(stack, "Iii", end, 0, schedule)
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, obj)
    u.emu_start(base + 0xF350, end, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == end
    count = admins * (schedule + 1)
    return dict(
        admins=admins,
        schedule=schedule,
        interval=interval,
        dose=dose,
        times=get(patient + 0x20, "d" * count),
        doses=get(patient + 0x1B0, "i" * count),
    )


if __name__ == "__main__":
    print(
        json.dumps(
            [run(a, s, 28, d) for a in (1, 3, 5, 10) for s in range(5) for d in (0, 6)], indent=2
        )
    )
