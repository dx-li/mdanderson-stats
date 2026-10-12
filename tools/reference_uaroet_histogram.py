"""Actual UAROET1.8 posterior histogram index arithmetic on supplied values."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_ESI, UC_X86_REG_XMM1, UC_X86_REG_XMM3

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != "b953affb6cd3968c389d5a35020e594cd25fef3ff61825db8aa282d8216a3eaf":
    raise ValueError("original executable hash mismatch")
pe = pefile.PE(str(path))
u = Uc(UC_ARCH_X86, UC_MODE_32)
base = pe.OPTIONAL_HEADER.ImageBase
u.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, pe.get_memory_mapped_image())
u.mem_map(0x20000000, 4096)
# The caller loads .5 from4c9448; the probability scale is100.
half = struct.unpack("<d", pe.get_data(0xC9448, 8))[0]
assert half == 0.5
cases = []
for value in (
    0.0,
    0.0001,
    0.0049,
    0.005,
    0.0051,
    0.0099,
    0.01,
    0.4949,
    0.495,
    0.5,
    0.9949,
    0.995,
    0.9999,
    1.0,
):
    u.mem_write(0x20000000, struct.pack("<d", value))
    u.reg_write(UC_X86_REG_EAX, 0x20000000)
    u.reg_write(UC_X86_REG_ESI, 0)
    u.reg_write(UC_X86_REG_XMM1, int.from_bytes(struct.pack("<d", half), "little"))
    u.reg_write(UC_X86_REG_XMM3, int.from_bytes(struct.pack("<d", 100.0), "little"))
    u.emu_start(base + 0x49C3F, base + 0x49C5B, count=100)
    cases.append(dict(probability=value, bin=u.reg_read(UC_X86_REG_EAX)))
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="native rounded probability histogram index",
            cases=cases,
        ),
        indent=2,
    )
)
