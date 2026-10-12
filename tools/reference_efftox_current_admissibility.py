"""Execute original current EffTox admissibility x86 with supplied counts/tails.

Usage: python tools/reference_efftox_current_admissibility.py ORIGINAL_WRAPPER_DLL
Only the checked integer-vector accessor is supplied as a host service. Every
admissibility branch at RVA 0xecc0 executes original instructions. No inference,
DLL startup, Windows exception, or end-to-end clinical validation is claimed.
"""

import hashlib
import json
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_ECX, UC_X86_REG_ESP

path = sys.argv[1]
assert hashlib.sha256(open(path, "rb").read()).hexdigest() == (
    "6b266b3927bec98fd6d64f5767a1110cfccc3fe6bd18c7d438b9cfd538e4479c"
)
pe = pefile.PE(path)
base = pe.OPTIONAL_HEADER.ImageBase
uc = Uc(UC_ARCH_X86, UC_MODE_32)
uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
uc.mem_write(base, pe.get_memory_mapped_image())
uc.mem_map(0x20000000, 0x100000)
uc.mem_map(0x30000000, 0x1000)
obj, model, counts_ptr, stack = 0x20000000, 0x20001000, 0x20002000, 0x200F0000


def u32(address):
    return struct.unpack("<I", uc.mem_read(address, 4))[0]


def hook(machine, address, size, data):
    if address == base + 0xE930:
        esp = machine.reg_read(UC_X86_REG_ESP)
        vector = machine.reg_read(UC_X86_REG_ECX)
        index = u32(esp + 4)
        assert index < u32(vector + 8)
        machine.reg_write(UC_X86_REG_EAX, u32(vector + 4) + index * 4)
        machine.reg_write(UC_X86_REG_ESP, esp + 8)
        from unicorn.x86_const import UC_X86_REG_EIP

        machine.reg_write(UC_X86_REG_EIP, u32(esp))


uc.hook_add(UC_HOOK_CODE, hook)
rows = []
for counts in ([0] * 5, [3, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 3, 0, 3, 0], [0, 0, 0, 0, 3]):
    for start in (0, 2):
        for tox, efficacy in ((0.1, 0.1), (0.1, 0.100001), (0.099999, 0.9), (0.9, 0), (0.9, 0.9)):
            for candidate in range(5):
                uc.mem_write(obj + 4, struct.pack("<I", model))
                uc.mem_write(obj + 12, struct.pack("<II", counts_ptr, 5))
                uc.mem_write(counts_ptr, struct.pack("<5i", *counts))
                uc.mem_write(model + 0x28, struct.pack("<ddI", 0.1, 0.1, start))
                uc.mem_write(model + 0xBC, struct.pack("<I", 5))
                uc.mem_write(stack, struct.pack("<IIdd", 0x30000000, candidate, tox, efficacy))
                uc.reg_write(UC_X86_REG_ESP, stack)
                uc.reg_write(UC_X86_REG_ECX, obj)
                uc.emu_start(base + 0xECC0, 0x30000000, count=10000)
                assert uc.reg_read(UC_X86_REG_ESP) == stack + 24
                rows.append(
                    dict(
                        counts=counts,
                        start=start,
                        candidate=candidate,
                        toxicity_tail=tox,
                        efficacy_tail=efficacy,
                        admissible=bool(uc.reg_read(UC_X86_REG_EAX) & 255),
                    )
                )
print(json.dumps(rows, indent=2))
