"""Execute the complete original current binary posterior integrand.

Hash-pinned wrapper RVA0x20b50 runs, including native normal PDF arithmetic.
Host services supply checked vector/matrix access, exp, logistic and integer
power only. Priors are initialized with explicit Gaussian means/SDs and their
normalization. No posterior quadrature, integration limits or RNG is emulated.
"""

import hashlib
import json
import math
import struct
import sys

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EAX,
    UC_X86_REG_ECX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESP,
    UC_X86_REG_XMM0,
)

path = sys.argv[1]
assert (
    hashlib.sha256(open(path, "rb").read()).hexdigest()
    == "6b266b3927bec98fd6d64f5767a1110cfccc3fe6bd18c7d438b9cfd538e4479c"
)
p = pefile.PE(path)
base = p.OPTIONAL_HEADER.ImageBase
u = Uc(UC_ARCH_X86, UC_MODE_32)
u.mem_map(base, (p.OPTIONAL_HEADER.SizeOfImage + 4095) // 4096 * 4096)
u.mem_write(base, p.get_memory_mapped_image())
u.mem_map(0x20000000, 0x100000)
u.mem_map(0x30000000, 0x1000)
obj, params, out, doses, counts, priors, stack, temp = (
    0x20000000,
    0x20001000,
    0x20002000,
    0x20003000,
    0x20004000,
    0x20005000,
    0x200F0000,
    0x20006000,
)
stub = 0x30000100
u.mem_write(stub, b"\xdd\x05" + struct.pack("<I", temp) + b"\xc3")


def w(a, fmt, *v):
    u.mem_write(a, struct.pack("<" + fmt, *v))


def ri(a):
    return struct.unpack("<I", u.mem_read(a, 4))[0]


def rd(a):
    return struct.unpack("<d", u.mem_read(a, 8))[0]


def xr(reg):
    return struct.unpack("<d", u.reg_read(reg).to_bytes(16, "little")[:8])[0]


def xw(reg, v):
    u.reg_write(reg, int.from_bytes(struct.pack("<d", v), "little"))


def ret(n=0):
    s = u.reg_read(UC_X86_REG_ESP)
    r = ri(s)
    u.reg_write(UC_X86_REG_ESP, s + 4 + n)
    u.reg_write(UC_X86_REG_EIP, r)


def floating(v):
    w(temp, "d", v)
    u.reg_write(UC_X86_REG_EIP, stub)


def hook(m, a, size, data):
    s = u.reg_read(UC_X86_REG_ESP)
    c = u.reg_read(UC_X86_REG_ECX)
    if a in (base + 0x20A70, base + 0xB190):
        i = ri(s + 4)
        assert i < ri(c + 8)
        u.reg_write(UC_X86_REG_EAX, ri(c + 4) + 8 * i)
        ret(4)
    elif a == base + 0x20910:
        row, col = ri(s + 4), ri(s + 8)
        assert row < ri(c + 8) and col < ri(c + 12)
        u.reg_write(UC_X86_REG_EAX, ri(c + 4) + 4 * (row * ri(c + 12) + col))
        ret(8)
    elif a == base + 0x28850:
        v = rd(s + 4)
        floating(1 / (1 + math.exp(-v)))
    elif a == base + 0x13770:
        floating(rd(s + 4) ** ri(s + 12))
    elif a == base + 0x41758:
        xw(UC_X86_REG_XMM0, math.exp(xr(UC_X86_REG_XMM0)))
        ret()


u.hook_add(UC_HOOK_CODE, hook)
nd = 3
w(obj + 4, "III", counts, nd, 4)
w(obj + 0x30, "II", doses, nd)
w(obj + 0x58, "dd", 0.3, 0.5)
w(obj + 0x80, "III", priors, priors + 6 * 80, priors + 6 * 80)
w(params + 4, "II", params + 0x100, 6)
w(out + 4, "II", out + 0x100, 1 + 4 * nd)
w(doses, "3d", -1, 0, 1)
for i in range(6):
    w(priors + i * 80, "I", base + 0x4AA08)
    w(priors + i * 80 + 8, "dd", 0, 2)
    w(priors + i * 80 + 0x48, "d", -math.log(2 * math.sqrt(2 * math.pi)))
rows = []
for beta in (-2.0, 0.0, 2.0):
    for association in (-1.0, 0.0, 1.0):
        for observed in ([0] * 12, [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2], [0, 1, 2, 1] * 3):
            for prior_mean, prior_sd in ((0.0, 2.0), (-0.2, 1.3), (0.5, 3.0)):
                for i in range(6):
                    w(priors + i * 80 + 8, "dd", prior_mean, prior_sd)
                    w(priors + i * 80 + 0x48, "d", -math.log(prior_sd * math.sqrt(2 * math.pi)))
                coefficients = [-1, beta, 0.2, 0.4, -0.1, association]
                w(params + 0x100, "6d", *coefficients)
                w(counts, "12i", *observed)
                w(stack, "III", 0x30000000, params, out)
                u.reg_write(UC_X86_REG_ECX, obj)
                u.reg_write(UC_X86_REG_ESP, stack)
                u.emu_start(base + 0x20B50, 0x30000000, count=100000)
                values = struct.unpack("<13d", u.mem_read(out + 0x100, 104))
                raw_t = [1 / (1 + math.exp(-(-1 + beta * x))) for x in (-1, 0, 1)]
                raw_e = [1 / (1 + math.exp(-(0.2 + 0.4 * x - 0.1 * x * x))) for x in (-1, 0, 1)]
                log_weight = sum(
                    -0.5 * ((v - prior_mean) / prior_sd) ** 2
                    - math.log(prior_sd * math.sqrt(2 * math.pi))
                    for v in coefficients
                )
                for dose, (t, e) in enumerate(zip(raw_t, raw_e)):
                    correction = math.tanh(association / 2) * e * (1 - e) * t * (1 - t)
                    cells = [
                        (1 - e) * t - correction,
                        (1 - e) * (1 - t) + correction,
                        e * t + correction,
                        e * (1 - t) - correction,
                    ]
                    log_weight += sum(
                        n * math.log(cell)
                        for n, cell in zip(observed[4 * dose : 4 * dose + 4], cells)
                    )
                assert math.isclose(values[0], math.exp(log_weight), rel_tol=2e-13)
                summary_t = [raw_t[0]] * 3 if beta < 0 else raw_t
                expected = [
                    1.0,
                    *[float(t < 0.3) for t in summary_t],
                    *[float(e > 0.5) for e in raw_e],
                    *summary_t,
                    *raw_e,
                ]
                for actual, wanted in zip(values, expected):
                    assert math.isclose(actual / values[0], wanted, rel_tol=2e-13, abs_tol=1e-15)
                rows.append(
                    dict(
                        dose_codes=[-1, 0, 1],
                        parameters=coefficients,
                        native_counts=observed,
                        prior_mean=[prior_mean] * 6,
                        prior_sd=[prior_sd] * 6,
                        posterior_integrand=values[0],
                        weighted_outputs=list(values),
                        raw_toxicity=raw_t,
                        raw_efficacy=raw_e,
                    )
                )
print(json.dumps(rows, indent=2))
