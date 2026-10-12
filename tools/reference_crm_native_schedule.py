"""Original CRM Suite prior and duration probes; requires dnfile/unicorn.

Pass the original hash-pinned kernel DLL as the first argument. The imported
branch harness supplies mapped memory; matrix access and C log are substituted.
Original cumulative-prior calibration, shape/rate construction and duration
instructions execute. No native executable is redistributed.
"""

import contextlib
import io
import json
import math
import struct
from pathlib import Path

from unicorn import UC_HOOK_CODE
from unicorn.x86_const import (
    UC_X86_REG_EAX,
    UC_X86_REG_ECX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESI,
    UC_X86_REG_ESP,
    UC_X86_REG_XMM0,
)

with contextlib.redirect_stdout(io.StringIO()):
    from reference_crm_native_branches import RET, base, env, get, put


def prior(probs, T):
    ans = []
    for k in range(6):
        u = env()
        sp = u.reg_read(UC_X86_REG_ESP)
        obj = 0x610000
        vec = 0x620000
        u.reg_write(UC_X86_REG_ECX, obj)
        put(u, sp, RET)
        u.mem_write(sp + 4, struct.pack("<ddiI", T / 6, T / 3, k, vec))

        def hook(u, a, s, d):
            if a == RET:
                u.emu_stop()
            elif a == base + 0x14980:
                sp = u.reg_read(UC_X86_REG_ESP)
                idx = get(u, sp + 4)
                u.mem_write(0x630000, struct.pack("<d", probs[idx]))
                u.reg_write(UC_X86_REG_EAX, 0x630000)
                u.reg_write(UC_X86_REG_EIP, get(u, sp))
                u.reg_write(UC_X86_REG_ESP, sp + 8)
            elif a == base + 0x25462:
                sp = u.reg_read(UC_X86_REG_ESP)
                x = struct.unpack(
                    "<d", int(u.reg_read(UC_X86_REG_XMM0) & ((1 << 64) - 1)).to_bytes(8, "little")
                )[0]
                y = math.log(x)
                u.reg_write(UC_X86_REG_XMM0, int.from_bytes(struct.pack("<d", y), "little"))
                u.reg_write(UC_X86_REG_EIP, get(u, sp))
                u.reg_write(UC_X86_REG_ESP, sp + 4)

        u.hook_add(UC_HOOK_CODE, hook)
        u.emu_start(base + 0x197A0, RET, count=10000)
        u.reg_write(UC_X86_REG_ESP, sp)
        u.reg_write(UC_X86_REG_ECX, obj)
        put(u, sp, RET)
        u.mem_write(sp + 4, struct.pack("<d", T))
        u.emu_start(base + 0x19940, RET, count=10000)
        vals = struct.unpack("<5d", u.mem_read(obj, 40))
        ans.append(
            dict(shape=vals[0], rate=vals[1], mean=vals[2], q_previous=vals[3], q_next=vals[4])
        )
    return ans


def duration(patients, prior_total=0):
    u = env()
    sp = u.reg_read(UC_X86_REG_ESP)
    start = 0x610000
    end = start + 72 * len(patients)
    out = 0x620000
    put(u, sp + 0x158, start, end)
    put(u, sp + 0x154, out)
    u.mem_write(sp + 0x104, struct.pack("<d", 0))
    u.mem_write(out + 0x28, struct.pack("<d", prior_total))
    u.reg_write(UC_X86_REG_ESI, end)
    for i, (enroll, ascertain_delay) in enumerate(patients):
        p = start + 72 * i
        put(u, p + 8, 0)
        u.mem_write(p + 0x10, struct.pack("<d", enroll))
        u.mem_write(p + 0x30, struct.pack("<d", ascertain_delay))

    def hook(u, a, s, d):
        if a == base + 0x2DCEA:
            u.emu_stop()

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + 0x2DC16, base + 0x2DCEA, count=10000)
    return struct.unpack("<d", u.mem_read(out + 0x28, 8))[0] - prior_total


cases = [
    [],
    [(0, 10)],
    [(0, 10), (3, 2)],
    [(0, 1), (3, 10)],
    [(0, 10), (3, 2), (7, 1)],
    [(0.4, 0.1)],
]
print(
    json.dumps(
        {
            "priors": [
                {"probabilities": p, "window": 42, "native_prior": prior(p, 42)}
                for p in [
                    case["probabilities"]
                    for case in json.loads(
                        (
                            Path(__file__).resolve().parents[1]
                            / "tests/fixtures/crm-native-prior.json"
                        ).read_text()
                    )
                ]
            ],
            "durations": [
                {
                    "patient_enrollment_and_final_ascertainment_delay": p,
                    "duration": duration(p, 9.0),
                }
                for p in cases
            ],
        },
        indent=2,
    )
)
