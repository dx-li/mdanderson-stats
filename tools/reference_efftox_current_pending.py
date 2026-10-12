"""Original pending-consistency loop with explicitly supplied inference/completions.

The original loop at RVA0xed80 executes, including observed-baseline comparison,
stop/dose equality and short-circuit behavior. Runtime containers, completion
stream and posterior decisions are host services. This does NOT validate the
native completion enumerator, posterior integration, RNG or Windows exceptions.
Requires the original hash-pinned EffToxManagedWrapper.dll; no binary redistribution.
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
uc.mem_map(0, 0x1000)
uc.mem_map(0x20000000, 0x100000)
uc.mem_map(0x30000000, 0x4000)
obj, model, matrix, pending, stack, vt = (
    0x20000000,
    0x20001000,
    0x20002000,
    0x20003000,
    0x200F0000,
    0x20004000,
)
heap = 0x20010000


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


def mat(a):
    global heap
    w(a, vt, heap, 5, 4)
    uc.mem_write(heap, b"\0" * 80)
    heap += 128
    w(a + 0x24, 15)


baseline = (0, False)
completions = []
cursor = 0
active = None
seen = []


def hook(m, a, size, data):
    global cursor, active
    s = m.reg_read(UC_X86_REG_ESP)
    this = m.reg_read(UC_X86_REG_ECX)
    if a == 0x30002000:
        ret(val=4)
    elif a == base + 0x23680:
        ret(8, val=this)
    elif a == base + 0xE1A0:
        w(this, 0, 0)
        ret(val=this)
    elif a == base + 0xE110:
        mat(this)
        ret(8, val=this)
    elif a == 0x30001000:
        mat(this)
        ret(16)
    elif a == base + 0xF790:
        row = baseline if active is None else completions[active]
        out = u(s + 8)
        w(out, row[0] & 0xFFFFFFFF, int(row[1]))
        seen.append(row)
        ret(8)
    elif a == base + 0x23200:
        if cursor < len(completions):
            active = cursor
            cursor += 1
            mat(u(s + 4))
            ret(4, val=1)
        else:
            ret(4, val=0)
    elif a == base + 0xDD30:
        out = u(s + 4)
        mat(out)
        ret(val=out)
    elif a == base + 0x41710:
        dest, source, n = (u(s + i) for i in (4, 8, 12))
        uc.mem_write(dest, bytes(uc.mem_read(source, n)))
        ret(val=dest)
    elif a in [
        base + x for x in (0x3ECA1, 0x3EC8A, 0xC510, 0xE3E0, 0xE480, 0xAB90, 0xCF00, 0x3F033)
    ]:
        ret()


uc.hook_add(UC_HOOK_CODE, hook)
rows = []
for baseline in ((0, False), (2, False), (-1, True)):
    for completions in (
        [],
        [baseline],
        [baseline] * 3,
        [(1, False)],
        [(3, False)] * 3,
        [baseline, (baseline[0], not baseline[1])],
        [baseline, baseline, (-1, True)],
    ):
        heap = 0x20010000
        cursor = 0
        active = None
        seen = []
        w(obj + 4, model)
        w(model, 0x20005000)
        w(0x20005000, 0x30002000)
        w(vt + 4, 0x30001000)
        mat(matrix)
        w(pending + 8, 5)
        w(stack, 0x30000000, matrix, pending)
        uc.reg_write(UC_X86_REG_ESP, stack)
        uc.reg_write(UC_X86_REG_ECX, obj)
        try:
            uc.emu_start(base + 0xED80, 0x30000000, count=100000)
        except Exception:
            print("EIP", hex(uc.reg_read(UC_X86_REG_EIP)), file=sys.stderr)
            raise
        value = bool(uc.reg_read(UC_X86_REG_EAX) & 255)
        assert value == all(x == baseline for x in completions), (baseline, completions, value)
        rows.append(
            dict(baseline=baseline, completions=completions, analyzed=seen, consistent=value)
        )
print(json.dumps(rows, indent=2))
