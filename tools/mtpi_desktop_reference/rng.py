from native import (
    UC_X86_REG_EIP,
    UC_X86_REG_ESP,
    base,
    get,
    pack,
    st,
    struct,
    u,
)
from unicorn.x86_const import UC_X86_REG_ECX

u.mem_write(0x30000800, b"\xdd\x1d" + struct.pack("<I", 0x30000200))


def call(rva, args=(), ecx=0, save_double=False):
    pack(st, "I" * (len(args) + 1), 0x30000800, *args)
    u.reg_write(UC_X86_REG_ESP, st)
    u.reg_write(UC_X86_REG_ECX, ecx)
    u.emu_start(base + rva, 0x30000800, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == 0x30000800
    if save_double:
        u.emu_start(0x30000800, 0x30000806, count=1)
        return get(0x30000200, "d")[0]


call(0x1AA30, ecx=base + 0x38678)
call(0xF560, (2,))
print([call(0xF5C0, save_double=True) for _ in range(2000)])
