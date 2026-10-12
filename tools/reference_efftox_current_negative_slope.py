"""Execute original current EffTox negative-slope summary and normal log-density.

Runs original integrand instructions RVA0x20cca..0x20d07 and normal-log-density
arithmetic RVA0x22960..0x22984, with supplied logistic probabilities and prior
normalization. No posterior integration, integration bounds or RNG is supplied
or claimed. All numerical/control-flow instructions in these blocks are original.
"""

import hashlib
import json
import math
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EBP,
    UC_X86_REG_ECX,
    UC_X86_REG_ESP,
    UC_X86_REG_XMM0,
    UC_X86_REG_XMM2,
)

assert hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() == (
    "6b266b3927bec98fd6d64f5767a1110cfccc3fe6bd18c7d438b9cfd538e4479c"
)
pe = pefile.PE(sys.argv[1])
base = pe.OPTIONAL_HEADER.ImageBase
uc = Uc(UC_ARCH_X86, UC_MODE_32)
uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
uc.mem_write(base, pe.get_memory_mapped_image())
uc.mem_map(0x20000000, 0x10000)
obj, stack = 0x20000000, 0x20008000


def write_double(address, value):
    uc.mem_write(address, struct.pack("<d", value))


def read_double(address):
    return struct.unpack("<d", uc.mem_read(address, 8))[0]


def xmm(value):
    return int.from_bytes(struct.pack("<d", value), "little")


rows = []
for slope in (-2.0, -1e-12, 0.0, 2.0):
    for first, current in ((0.8, 0.2), (0.2, 0.8), (0.3, 0.3)):
        for cutoff in (0.0, 0.3, 0.5, 1.0):
            uc.reg_write(UC_X86_REG_EBP, obj)
            uc.reg_write(UC_X86_REG_ESP, stack)
            uc.reg_write(UC_X86_REG_XMM0, xmm(current))
            uc.reg_write(UC_X86_REG_XMM2, xmm(first))
            write_double(stack + 0x30, slope)
            write_double(obj + 0x58, cutoff)
            uc.emu_start(base + 0x20CCA, base + 0x20D07, count=100)
            reported = read_double(stack + 0x38)
            tail = read_double(stack + 0x20)
            assert reported == (first if slope < 0 else current)
            assert tail == float(reported < cutoff)
            rows.append(
                dict(
                    slope=slope,
                    first_dose_toxicity=first,
                    raw_current_toxicity=current,
                    cutoff=cutoff,
                    summary_toxicity=reported,
                    toxicity_tail_indicator=tail,
                )
            )
normal = []
for mean, sd in ((0.0, 1.0), (1.5, 3.0), (-2.0, 0.5)):
    for value in (-3.0, -0.1, 0.0, 0.1, 3.0):
        uc.reg_write(UC_X86_REG_ECX, obj)
        uc.reg_write(UC_X86_REG_ESP, stack)
        write_double(obj + 8, mean)
        write_double(obj + 0x10, sd)
        write_double(obj + 0x48, -math.log(sd * math.sqrt(2 * math.pi)))
        write_double(stack + 4, value)
        uc.emu_start(base + 0x22960, base + 0x22984, count=100)
        actual = struct.unpack("<d", uc.reg_read(UC_X86_REG_XMM0).to_bytes(16, "little")[:8])[0]
        expected = -0.5 * ((value - mean) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))
        assert math.isclose(actual, expected, rel_tol=1e-14)
        normal.append(dict(mean=mean, sd=sd, value=value, log_density=actual))
print(json.dumps(dict(summary_branch=rows, normal_log_density=normal), indent=2))
