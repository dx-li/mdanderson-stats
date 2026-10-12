"""Native PRT predictive/controller block from supplied patient risks and beta moments."""

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


def run(xi, current, pending, alpha, beta, assigned_override=None):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, (native_image.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
    u.mem_write(base, native_image.get_memory_mapped_image())
    u.mem_map(0x20000000, 0x200000)
    bp = 0x20100000

    def put(a, fmt, *values):
        u.mem_write(a, struct.pack("<" + fmt, *values))

    def get(a):
        return struct.unpack("<d", u.mem_read(a, 8))[0]

    put(bp - 0x34, "i", current)
    put(bp - 8, "i", len(xi))
    put(bp - 0x38, "i", int(current < len(xi)))
    for offset, value in [(0x58, alpha), (0x60, beta), (0x48, alpha), (0x50, beta)]:
        put(bp - offset, "d", value)
    put(0x77DB60, "d", 0.9)
    put(0x77DB68, "d", 0.3)
    put(0x77DC48, "d", 0.05)
    put(0x77DC18, "d", 0.3)
    put(0x77C970, "d" * len(xi), *xi)
    assigned = [3 + len(p) for p in pending] if assigned_override is None else assigned_override
    assessed = [a - len(p) for a, p in zip(assigned, pending)]
    put(0x77DA50, "d" * len(xi), *assigned)
    put(0x46CEB0, "d" * len(xi), *assessed)
    for k, probabilities in enumerate(pending):
        if probabilities:
            put(0x77C490 + (k + 1) * 6400, "d" * len(probabilities), *probabilities)
    # Native rows represent all combinations for each possible pending count.
    # Repeated short tapes in the larger row ensure every prefix is complete.
    m = max(map(len, pending))
    for j in range(2**m):
        for i in range(m):
            put(0x7CA068 + 4 * j + 0x1000 * i, "i", (j >> i) & 1)
    u.reg_write(UC_X86_REG_EBP, bp)
    u.reg_write(UC_X86_REG_ESP, bp - 0x1000)
    u.reg_write(UC_X86_REG_FPCW, 0x23F)
    u.emu_start(base + 0x46AB, base + 0x5552, count=1000000)
    if u.reg_read(UC_X86_REG_EIP) != base + 0x5552:
        raise RuntimeError("native controller instruction budget exhausted")
    return dict(
        assigned=assigned,
        assessed=assessed,
        exceedance=xi,
        current=current,
        pending=pending,
        alpha=alpha,
        beta=beta,
        code=get(bp - 0x80),
        pn_current=get(bp - 0x68),
        pe_next=get(bp - 0x70),
        pe_acceptable=get(bp - 0x78),
        pe_negligible=get(0x77DB40),
    )


cases = []
for D, k, m, n, (alpha, beta), kind in itertools.product(
    (3, 10),
    (0, 1, 2),
    (0, 1, 2),
    (0, 1, 2),
    ((0.1, 20.0), (20.0, 0.1), (1.0, 3.0), (3.0, 2.0)),
    ("low", "boundary", "excess_boundary", "excess", "next_excess", "next_boundary"),
):
    current = (1, D - 1, D)[k]
    value = {
        "low": 0.1,
        "boundary": 0.3,
        "excess_boundary": 0.9,
        "excess": 0.95,
        "next_excess": 0.1,
        "next_boundary": 0.1,
    }[kind]
    xi = [0.1] * (current - 1) + [value] * (D - current + 1)
    if kind in ("next_excess", "next_boundary") and current < D:
        xi[current:] = [0.95 if kind == "next_excess" else 0.9] * (D - current)
    pending = [[] for _ in range(D)]
    pending[current - 1] = [0.2, 0.7][:m]
    if current < D:
        pending[current] = [0.3, 0.8][:n]
    cases.append(run(xi, current, pending, alpha, beta))
for D, current, m, (alpha, beta) in itertools.product(
    (3, 6, 10), (1, 2), (0, 1, 2), ((0.1, 20.0), (20.0, 0.1), (1.0, 3.0), (3.0, 2.0))
):
    xi = [0.1] * D
    pending = [[] for _ in range(D)]
    pending[current - 1] = [0.2, 0.7][:m]
    assigned = [0.0] * D
    assigned[current - 1] = 3.0 + m
    cases.append(run(xi, current, pending, alpha, beta, assigned))
print(
    json.dumps(
        dict(
            executable_sha256=digest,
            scope=(
                "full conduct predictive/controller supplied-state grid; "
                "inference and text rendering excluded"
            ),
            cases=cases,
        ),
        indent=2,
    )
)
