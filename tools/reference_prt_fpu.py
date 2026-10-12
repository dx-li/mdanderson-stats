"""Execute the CRT precision initializer in both original PRT executables.

The initializer is found by its two explicit _controlfp arguments; its call and
return run original instructions, with no arithmetic or runtime stubs.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_ESP, UC_X86_REG_FPCW

root = Path(sys.argv[1]).parent
results = []
for name, expected in (
    ("Sims", "52903a7af2e75998bf2a8cbda6e2049e7fa25e0981cf9806317841da13a91b54"),
    ("Conduct", "2943f71ab27e300c1168b2d9f6a93815b75efe753b6af2b543c1370d077a1123"),
):
    path = root / f"PRT_{name}.exe"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("original executable hash mismatch")
    pe = pefile.PE(str(path))
    base = pe.OPTIONAL_HEADER.ImageBase
    pattern = bytes.fromhex("68000003006800000100e8")
    if raw.count(pattern) != 1:
        raise RuntimeError("precision initializer signature is not unique")
    rva = pe.get_rva_from_offset(raw.index(pattern))
    if pe.get_data(rva + 15, 3) != bytes.fromhex("5959c3"):
        raise RuntimeError("unexpected initializer return contract")
    destination = base + rva + 15 + struct.unpack("<i", pe.get_data(rva + 11, 4))[0]
    for initial in (0x7F, 0x27F, 0x37F):
        u = Uc(UC_ARCH_X86, UC_MODE_32)
        u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
        u.mem_write(base, pe.get_memory_mapped_image())
        u.mem_map(0x20000000, 0x200000)
        u.reg_write(UC_X86_REG_EBP, 0x20100000)
        u.reg_write(UC_X86_REG_ESP, 0x200FF000)
        u.reg_write(UC_X86_REG_FPCW, initial)
        u.emu_start(base + rva, base + rva + 17, count=10000)
        results.append(
            dict(
                executable=name,
                executable_sha256=expected,
                initializer_rva=rva,
                controlfp_rva=destination - base,
                initial_fpcw=initial,
                result_fpcw=u.reg_read(UC_X86_REG_FPCW),
            )
        )
print(
    json.dumps(
        dict(scope="actual CRT precision initializer; both PRT1.0 executables", cases=results),
        indent=2,
    )
)
