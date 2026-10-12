"""Native decision-text dispatch with formatting/I/O calls skipped, not reimplemented."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import capstone
import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_EIP, UC_X86_REG_ESP, UC_X86_REG_FPCW

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == "2943f71ab27e300c1168b2d9f6a93815b75efe753b6af2b543c1370d077a1123"
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
calls = {
    i.address: i.size
    for i in md.disasm(pe.get_data(0x5A61, 0x937A - 0x5A61), base + 0x5A61)
    if i.mnemonic == "call"
}
labels = {
    0x5A7D: "stop",
    0x5C06: "deescalate",
    0x5F62: "stay",
    0x62CF: "stay",
    0x65A7: "suspend",
    0x6914: "stay",
    0x6C84: "suspend",
    0x6FF4: "stay",
    0x730A: "stay",
    0x7614: "escalate",
    0x79FD: "escalate",
    0x7E24: "escalate",
    0x824B: "escalate",
    0x8723: "suspend",
    0x8BB8: "suspend",
    0x9066: "suspend",
}


def run(code):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, pe.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000
    value = struct.unpack("<f", struct.pack("<f", code))[0]
    u.mem_write(bp - 0x80, struct.pack("<d", value))
    u.mem_write(bp - 0x34, struct.pack("<i", 3))
    u.mem_write(bp - 8, struct.pack("<i", 6))
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x23F)
    seen = []

    def hook(uc, address, size, _):
        if address - base in labels:
            seen.append(labels[address - base])
        if address in calls:
            uc.reg_write(UC_X86_REG_EIP, address + calls[address])

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + 0x5A61, base + 0x937A, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x937A
    return dict(code=code, stored_code=value, decision_branches=seen)


cases = [
    run(c)
    for c in (
        3,
        4,
        5.1,
        5.11,
        5.2,
        6.0,
        6.1001,
        6.1,
        6.11,
        6.12,
        6.2,
        6.21,
        6.22,
        6.23,
        6.3,
        6.33,
        6.4,
    )
]
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="native decision label dispatch, Fortran formatting/I/O calls skipped",
            cases=cases,
        ),
        indent=2,
    )
)
