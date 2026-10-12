"""Original current desktop selector with supplied posterior summaries/utilities.

Requires the hash-pinned EffToxManagedWrapper.dll. Original selection and
admissibility branches execute; vector access, memcpy, security-cookie return
and the explicitly supplied utility callback are host services. This fixture
isolates controller behavior and does not claim posterior/utility integration.
"""

import hashlib
import json
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EAX,
    UC_X86_REG_ECX,
    UC_X86_REG_EDI,
    UC_X86_REG_EIP,
    UC_X86_REG_ESP,
)

path = sys.argv[1]
assert hashlib.sha256(open(path, "rb").read()).hexdigest() == (
    "6b266b3927bec98fd6d64f5767a1110cfccc3fe6bd18c7d438b9cfd538e4479c"
)
pe = pefile.PE(path)
base = pe.OPTIONAL_HEADER.ImageBase
uc = Uc(UC_ARCH_X86, UC_MODE_32)
uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
uc.mem_write(base, pe.get_memory_mapped_image())
uc.mem_map(0, 0x1000)
uc.mem_map(0x20000000, 0x100000)
uc.mem_map(0x30000000, 0x2000)
obj, model, output, stack = 0x20000000, 0x20001000, 0x20002000, 0x200F0000
utility_object, utility_table, utility_value = 0x20003000, 0x20004000, 0x20005000
# fld qword ptr [utility_value]; ret 16: x87 return with thiscall argument cleanup.
uc.mem_write(0x30001000, b"\xdd\x05" + struct.pack("<I", utility_value) + b"\xc2\x10\x00")
utilities = []


def u32(address):
    return struct.unpack("<I", uc.mem_read(address, 4))[0]


def return_from_host(extra=0):
    esp = uc.reg_read(UC_X86_REG_ESP)
    uc.reg_write(UC_X86_REG_EIP, u32(esp))
    uc.reg_write(UC_X86_REG_ESP, esp + 4 + extra)


def hook(machine, address, size, data):
    esp = machine.reg_read(UC_X86_REG_ESP)
    if address in (base + 0xE850, base + 0xE930, base + 0xB190):
        vector = machine.reg_read(UC_X86_REG_ECX)
        index = u32(esp + 4)
        assert index < u32(vector + 8)
        width = 8 if address == base + 0xB190 else 4
        machine.reg_write(UC_X86_REG_EAX, u32(vector + 4) + width * index)
        return_from_host(4)
    elif address == base + 0x3F033:
        return_from_host()
    elif address == base + 0x41710:
        dest, source, n = (u32(esp + i) for i in (4, 8, 12))
        machine.mem_write(dest, bytes(machine.mem_read(source, n)))
        machine.reg_write(UC_X86_REG_EAX, dest)
        return_from_host()
    elif address == 0x30001000:
        index = machine.reg_read(UC_X86_REG_EDI)
        machine.mem_write(utility_value, struct.pack("<d", utilities[index]))


uc.hook_add(UC_HOOK_CODE, hook)
rows = []
for counts in ([0] * 5, [3, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 3, 0, 3, 0], [0, 0, 0, 0, 3]):
    for start in (0, 2):
        for tox, efficacy in ((0.1, 0.1), (0.1, 0.100001), (0.099999, 0.9), (0.9, 0), (0.9, 0.9)):
            for utilities in (
                [0, 1, 2, 3, 4],
                [4, 3, 2, 1, 0],
                [1] * 5,
                [-5, -4, -3, -2, -1],
                [-sys.float_info.max] * 5,
            ):
                uc.mem_write(obj + 4, struct.pack("<I", model))
                uc.mem_write(model + 0x28, struct.pack("<ddI", 0.1, 0.1, start))
                uc.mem_write(model + 0x60, struct.pack("<I", utility_object))
                uc.mem_write(model + 0xBC, struct.pack("<I", 5))
                uc.mem_write(utility_object, struct.pack("<I", utility_table))
                uc.mem_write(utility_table, struct.pack("<I", 0x30001000))
                for j, (vector, values, kind) in enumerate(
                    [
                        (obj + 8, counts, "i"),
                        (obj + 0x30, [tox] * 5, "d"),
                        (obj + 0x58, [efficacy] * 5, "d"),
                        (obj + 0x80, [0.2] * 5, "d"),
                        (obj + 0xA8, [0.7] * 5, "d"),
                        (obj + 0xD0, [0] * 5, "d"),
                        (output + 0xA8, [0] * 5, "d"),
                    ]
                ):
                    pointer = 0x20010000 + j * 0x100
                    uc.mem_write(vector + 4, struct.pack("<II", pointer, 5))
                    uc.mem_write(pointer, struct.pack("<5" + kind, *values))
                uc.mem_write(output, struct.pack("<II", 0xFFFFFFFF, 0))
                uc.mem_write(stack, struct.pack("<II", 0x30000000, output))
                uc.reg_write(UC_X86_REG_ESP, stack)
                uc.reg_write(UC_X86_REG_ECX, obj)
                uc.emu_start(base + 0xFC90, 0x30000000, count=50000)
                assert uc.reg_read(UC_X86_REG_ESP) == stack + 8
                selected = struct.unpack("<i", uc.mem_read(output, 4))[0]
                stopped = bool(uc.mem_read(output + 4, 1)[0])
                rows.append(
                    dict(
                        counts=counts,
                        start=start,
                        toxicity_tail=tox,
                        efficacy_tail=efficacy,
                        utilities=utilities,
                        selected=selected,
                        stopped=stopped,
                    )
                )
print(json.dumps(rows, indent=2))
