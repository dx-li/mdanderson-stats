"""Original current EffTox outer status controller with supplied inference services.

Original RVA0x41c10 executes. Runtime containers/report strings, validation,
counts, pending consistency and dose-selection callbacks are host services.
The probe validates status ordering/cap comparisons, not posterior mathematics.
Requires the hash-pinned original wrapper DLL; no binaries are redistributed.
"""

import hashlib
import json
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP

assert (
    hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
    == "6b266b3927bec98fd6d64f5767a1110cfccc3fe6bd18c7d438b9cfd538e4479c"
)
p = pefile.PE(sys.argv[1])
base = p.OPTIONAL_HEADER.ImageBase
uc = Uc(UC_ARCH_X86, UC_MODE_32)
uc.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
uc.mem_write(base, p.get_memory_mapped_image())
uc.mem_map(0, 4096)
uc.mem_map(0x20000000, 0x100000)
uc.mem_map(0x30000000, 0x8000)
obj, model, patients, out, stack, vt, mvt = (
    0x20000000,
    0x20001000,
    0x20002000,
    0x20003000,
    0x200F0000,
    0x20004000,
    0x20005000,
)


def u(a):
    return struct.unpack("<I", uc.mem_read(a, 4))[0]


def w(a, *v):
    uc.mem_write(a, struct.pack("<" + "I" * len(v), *v))


def ret(n=0, val=None):
    s = uc.reg_read(UC_X86_REG_ESP)
    if val is not None:
        uc.reg_write(UC_X86_REG_EAX, val)
    uc.reg_write(UC_X86_REG_ESP, s + 4 + n)
    uc.reg_write(UC_X86_REG_EIP, u(s))


consistent = True
stopped = False
selected = 2
calls = []


def hook(m, a, size, data):
    s = m.reg_read(UC_X86_REG_ESP)
    this = m.reg_read(UC_X86_REG_ECX)
    if a == 0x30001000:
        ret(val=4)
    elif a == 0x30002000:
        ret(val=0)
    elif a == 0x30003000:
        calls.append("counts")
        ret(12)
    elif a == 0x30004000:
        calls.append("pending")
        ret(8, val=int(consistent))
    elif a == 0x30005000:
        calls.append("select")
        dest = u(s + 8)
        w(dest, selected & 0xFFFFFFFF, int(stopped))
        ret(8)
    elif a == base + 0xE110:
        ret(8, val=this)
    elif a == base + 0x41A30:
        ret(4, val=this)
    elif a == base + 0xE1A0:
        ret(val=this)
    elif a == base + 0x424C0:
        ret(4, val=u(s + 4))
    elif a == base + 0x41B40:
        ret(4, val=this)
    elif a in [base + x for x in (0xAD10, 0xE700, 0xCF00, 0xE3E0, 0x3F033)]:
        ret()


uc.hook_add(UC_HOOK_CODE, hook)
rows = []
for count in (0, 2, 3, 4):
    for pending in (False, True):
        for consistent in (False, True):
            for stopped, selected in ((False, 2), (True, -1)):
                calls = []
                w(obj, vt)
                w(vt, 0x30003000, 0x30004000, 0x30005000)
                w(model, mvt)
                w(mvt, 0x30001000, 0, 0x30002000)
                w(model + 0xBC, 5)
                w(model + 0x3C, 3)
                w(patients, 0x20020000, 0x20020000 + 24 * count)
                w(stack, 0x30000000, out, model, patients, int(pending))
                uc.reg_write(UC_X86_REG_ESP, stack)
                uc.reg_write(UC_X86_REG_ECX, obj)
                try:
                    uc.emu_start(base + 0x41C10, 0x30000000, count=10000)
                except Exception:
                    print(hex(uc.reg_read(UC_X86_REG_EIP)), file=sys.stderr)
                    raise
                status, dose = struct.unpack("<ii", uc.mem_read(out, 8))
                expected = (
                    1 if pending and not consistent else 2 if stopped else 3 if count >= 3 else 0
                )
                assert status == expected, (count, pending, consistent, stopped, status)
                rows.append(
                    dict(
                        patients=count,
                        maximum=3,
                        pending=pending,
                        consistent=consistent,
                        selection_stopped=stopped,
                        selected=selected,
                        status=status,
                        dose=dose,
                        calls=calls,
                    )
                )
print(json.dumps(rows, indent=2))
