"""Native PRT conduct predictive enumeration from supplied patient risks and beta moments."""

import hashlib
import itertools
import json
import struct
import sys
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EBP, UC_X86_REG_EIP, UC_X86_REG_ESP, UC_X86_REG_FPCW

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != "2943f71ab27e300c1168b2d9f6a93815b75efe753b6af2b543c1370d077a1123":
    raise ValueError("original conduct executable hash mismatch")
native_image = pefile.PE(str(path))
base = native_image.OPTIONAL_HEADER.ImageBase


def run(probabilities, alpha, beta, target, upper):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (native_image.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, native_image.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(a, fmt, *values):
        u.mem_write(a, struct.pack("<" + fmt, *values))

    put(bp - 0x34, "i", 1)
    put(bp - 0x58, "d", alpha)
    put(bp - 0x60, "d", beta)
    put(0x77DA50, "d", len(probabilities))
    put(0x46CEB0, "d", 0.0)
    put(0x77DC18, "d", target)
    put(0x77DB60, "d", upper)
    put(0x77DD90, "d" * len(probabilities), *probabilities)
    for j, bits in enumerate(itertools.product((0, 1), repeat=len(probabilities))):
        for i, bit in enumerate(bits):
            put(0x7CA068 + 4 * j + 0x1000 * i, "i", bit)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x23F)
    # The first two register values are live at4d3d from the preceding classifier.
    from unicorn.x86_const import UC_X86_REG_EDI, UC_X86_REG_ESI

    u.reg_write(UC_X86_REG_ESI, 1)
    u.reg_write(UC_X86_REG_EDI, 1)
    u.emu_start(base + 0x4D3D, base + 0x4FC8, count=1000000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x4FC8:
        raise RuntimeError("native predictive instruction budget exhausted")
    result = struct.unpack("<d", u.mem_read(bp - 0x78, 8))[0]
    return dict(
        probabilities=probabilities,
        alpha=alpha,
        beta=beta,
        target=target,
        upper=upper,
        predictive_excessive=result,
    )


cases = [
    run(p, a, b, t, h)
    for p, (a, b), t, h in itertools.product(
        ([0.2], [0.2, 0.7], [0.1, 0.4, 0.8], [0.1, 0.2, 0.3, 0.9]),
        ((1.0, 1.0), (2.0, 5.0), (0.5, 0.7)),
        (0.3, 0.5),
        (0.3, 0.9),
    )
]
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope="native fixed-patient-risk predictive enumeration, interior probabilities",
            cases=cases,
        ),
        indent=2,
    )
)
