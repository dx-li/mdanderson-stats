"""Original selected-trial Rselect aggregation arithmetic from EffTox 5.2.3.

Executes the original SSE instructions at RVA0x8d8e..0x8dca. Inputs are the
selected count, sum of selected true utilities, and scenario utility extrema.
No-selection returns zero; zero utility range retains the original nonfinite
result as an explicitly named string, not a recommended scientific default.
"""

import hashlib
import json
import math
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_EBP, UC_X86_REG_EDI, UC_X86_REG_MXCSR

assert hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() == (
    "7570fa52e5c03c4818ae8c4f17913bd75a2f036ebac844aa6c86e2ce3a183fd0"
)
pe = pefile.PE(sys.argv[1])
base = pe.OPTIONAL_HEADER.ImageBase
uc = Uc(UC_ARCH_X86, UC_MODE_32)
uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
uc.mem_write(base, pe.get_memory_mapped_image())
uc.mem_map(0x20000000, 0x10000)
frame, out = 0x20008000, 0x20001000
rows = []
for count, total, low, high in [
    (0, 0, -0.5, 0.7),
    (1, 0.7, -0.5, 0.7),
    (1, -0.5, -0.5, 0.7),
    (3, 0.3, -0.5, 0.7),
    (5, 1, -0.5, 0.7),
    (0, 0, 0.25, 0.25),
    (1, 0.25, 0.25, 0.25),
    (3, 0.3, 0.1, 0.1),
]:
    for offset, value in [(-0x74, low), (-0x7C, high), (-0x2C, total)]:
        uc.mem_write(frame + offset, struct.pack("<d", value))
    uc.reg_write(UC_X86_REG_EAX, out)
    uc.reg_write(UC_X86_REG_EBP, frame)
    uc.reg_write(UC_X86_REG_EDI, count)
    uc.reg_write(UC_X86_REG_MXCSR, 0x1F80)
    uc.emu_start(base + 0x8D8E, base + 0x8DCB, count=100)
    value = struct.unpack("<d", uc.mem_read(out, 8))[0]
    result = value if math.isfinite(value) else "NaN" if math.isnan(value) else str(value)
    rows.append(
        dict(
            selected_trials=count,
            utility_sum=total,
            minimum=low,
            maximum=high,
            native_rselect=result,
        )
    )
print(json.dumps(rows, indent=2, allow_nan=False))
