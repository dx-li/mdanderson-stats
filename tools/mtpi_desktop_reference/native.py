"""Execute hash-pinned original XLL segments; supply path in argv[1]."""

import hashlib
import struct
import sys
from pathlib import Path

import pefile
from scipy.special import betainc
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_EBX, UC_X86_REG_EIP, UC_X86_REG_ESP

path = sys.argv[1]
assert (
    hashlib.sha256(Path(path).read_bytes()).hexdigest()
    == "c946b1bc48baafe1f6477767a6d4f64f97ca7ff95398c9b29ed9159f79d0c858"
)
pe = pefile.PE(path)
base = pe.OPTIONAL_HEADER.ImageBase
u = Uc(UC_ARCH_X86, UC_MODE_32)
u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, pe.get_memory_mapped_image())
u.mem_map(0x20000000, 0x100000)
u.mem_map(0x30000000, 0x1000)
st, frame = 0x20010000, 0x20018000
u.mem_write(0x30000000, b"\xdd\x05" + struct.pack("<I", 0x30000100) + b"\xc3")


def pack(addr, fmt, *x):
    u.mem_write(addr, struct.pack("<" + fmt, *x))


def get(addr, fmt):
    return struct.unpack("<" + fmt, u.mem_read(addr, struct.calcsize("<" + fmt)))


calls = []


def hook(machine, address, size, data):
    if address == base + 0x6080:
        esp = u.reg_read(UC_X86_REG_ESP)
        x, a, b = get(esp + 4, "ddd")
        calls.append((x, a, b))
        pack(0x30000100, "d", float(betainc(a, b, x)))
        u.reg_write(UC_X86_REG_EIP, 0x30000000)


u.hook_add(UC_HOOK_CODE, hook)


def run(n, y, d, target=0.3, eps1=0.05, eps2=0.05, tt=0, toxdose=None):
    global calls
    calls = []
    D = len(n)
    u.mem_write(st, b"\0" * 0x1000)
    for offset, values in [
        (0x400, y),
        (0x42C, n),
        (0x458, [v + 1 for v in y]),
        (0x484, [a - b + 1 for a, b in zip(n, y)]),
    ]:
        ptr = 0x20020000 + offset * 16
        pack(st + offset, "II", ptr, D)
        pack(ptr, "d" * D, *values)
    pack(frame + 0x20, "d", target)
    pack(frame + 0x38, "dd", eps1, eps2)
    pack(st + 0x4C, "III", D, D if toxdose is None else toxdose, tt)
    pack(st + 0x68, "I", d)
    u.reg_write(UC_X86_REG_ESP, st)
    u.reg_write(UC_X86_REG_EBP, frame)
    u.reg_write(UC_X86_REG_EBX, d)
    u.emu_start(base + 0x83D8, base + 0x8789, count=10000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x8789
    return dict(
        n=n,
        y=y,
        d=d,
        target=target,
        tt=tt,
        next=u.reg_read(UC_X86_REG_EBX),
        toxdose=get(st + 0x50, "I")[0],
        newtt=get(st + 0x54, "I")[0],
        stop=get(st + 0x94, "d")[0],
        q=[get(st + x, "d")[0] for x in (0xAC, 0x84, 0x74)],
        beta_calls=calls,
    )
