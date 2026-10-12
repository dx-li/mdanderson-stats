import json
import struct

from arand_x86_reference import base, p
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EAX,
    UC_X86_REG_ECX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESP,
    UC_X86_REG_FPCW,
)


def run(now, arrival, event_time, outcome, rva=1794768):
    u = Uc(UC_ARCH_X86, UC_MODE_32)
    u.mem_map(base, p.OPTIONAL_HEADER.SizeOfImage + 4095 & ~4095)
    u.mem_write(base, p.get_memory_mapped_image())
    u.mem_map(0, 4096)
    u.mem_map(1048576, 131072)
    stack = 1114112
    patient = 1052672
    node = 1056768
    obj = 1060864
    key = 1064960
    stop = 1050640
    result = 1069056
    wrapper = 1050624
    u.mem_write(patient + 8, struct.pack("<d", arrival))
    u.mem_write(patient + 44, struct.pack("<I", 1073152))
    u.mem_write(node + 36, struct.pack("<II", 15, obj))
    u.mem_write(obj + 8, struct.pack("<dd", outcome, event_time))
    u.mem_write(obj + 72, b"\x01")
    u.mem_write(wrapper, b"\xdd\x1d" + struct.pack("<I", result) + b"\x90" * 10)
    u.mem_write(stack, struct.pack("<IIIdII", stop, patient, key, now, result, result + 8))
    u.reg_write(UC_X86_REG_ESP, stack)
    u.reg_write(UC_X86_REG_ECX, patient)
    u.reg_write(UC_X86_REG_FPCW, 895)

    def ret(n):
        sp = u.reg_read(UC_X86_REG_ESP)
        addr = struct.unpack("<I", u.mem_read(sp, 4))[0]
        u.reg_write(UC_X86_REG_ESP, sp + 4 + n)
        u.reg_write(UC_X86_REG_EIP, addr)

    def hook(u, addr, size, _):
        if addr == stop:
            u.emu_stop()
        elif addr == base + 1297216:
            u.reg_write(UC_X86_REG_EAX, node)
            ret(4)
        elif addr == base + 1060624:
            u.reg_write(UC_X86_REG_EAX, 0)
            ret(16)
        elif addr == base + 1726456:
            ret(0)

    u.hook_add(UC_HOOK_CODE, hook)
    u.emu_start(base + rva, stop, count=100000)
    assert u.reg_read(UC_X86_REG_EIP) == stop
    return {
        "followup": struct.unpack("<d", u.mem_read(result, 8))[0],
        "event": struct.unpack("<i", u.mem_read(result + 8, 4))[0],
    }


if __name__ == "__main__":
    for v in [0.0, 1.0]:
        for now in [2.0, 4.999999999, 5.0, 5.000000001, 9.0]:
            print(
                json.dumps(
                    {
                        "now": now,
                        "arrival": 1.0,
                        "record_time": 5.0,
                        "stored_indicator": v,
                        "original": run(now, 1.0, 5.0, v),
                    }
                )
            )
