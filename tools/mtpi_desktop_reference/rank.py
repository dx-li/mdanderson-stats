import json

from native import (
    UC_X86_REG_EBP,
    UC_X86_REG_EBX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESP,
    base,
    frame,
    get,
    pack,
    st,
    u,
)
from unicorn.x86_const import UC_X86_REG_EDI

initial = u.context_save()


def select(scores, target, mindose=0):
    u.context_restore(initial)
    u.mem_write(st, b"\0" * 0x1000)
    ptr = 0x20030000
    out = 0x20031000
    pack(st + 0x504, "II", ptr, len(scores))
    pack(ptr, "d" * len(scores), *scores)
    pack(st + 0x530, "II", out, 1)
    pack(st + 0x64, "I", 0)
    pack(st + 0x58, "I", mindose)
    pack(frame + 0x20, "d", target)
    u.reg_write(UC_X86_REG_ESP, st)
    u.reg_write(UC_X86_REG_EBP, frame)
    u.reg_write(UC_X86_REG_EBX, len(scores))
    u.reg_write(UC_X86_REG_EDI, 0)
    u.emu_start(base + 0x8A11, base + 0x8AF6, count=10000)
    assert u.reg_read(UC_X86_REG_EIP) == base + 0x8AF6
    return dict(
        scores=scores,
        target=target,
        mindose=mindose,
        adjusted=get(ptr, "d" * len(scores)),
        selected=get(out, "d")[0],
    )


for scores, target, mindose in [
    ([0.2, 0.2, 0.2], 0.3, 0),
    ([0.3, 0.3, 0.3], 0.25, 0),
    ([0.2, 0.4], 0.3, 0),
    ([0.5, 0.1, 0.3], 0.3, 1),
    ([0.2], 0.3, 0),
    ([0.25, 0.75 - 1e-10], 0.5, 0),
]:
    print(json.dumps(select(scores, target, mindose)))
