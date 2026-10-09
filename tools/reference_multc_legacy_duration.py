"""Native Multc Lean duration references through bounded x86 emulation.

Research-only dependencies: pefile 2024.8.26 and unicorn 2.1.4. Original DLL
is a local input with a verified checksum; neither it nor its runtime ships
with the package. Only external RNG inputs, vector accessors and security
cookie verification are replaced. Original duration/scaling/clipping and
balking instructions run in Unicorn; no Python duration model is imported.
"""

import hashlib
import json
import struct
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP

DLL_SHA256 = "8ac45568e87530d4bb6732eb8807a22e4e8c5de52a5557a14f750c24867fa807"


def run(
    dll,
    *,
    n,
    response,
    notox,
    prob,
    uniform,
    unit_exponential,
    mean,
    window,
):
    if hashlib.sha256(dll.read_bytes()).hexdigest() != DLL_SHA256:
        raise ValueError("native DLL checksum mismatch")
    p = pefile.PE(str(dll))
    uc = Uc(UC_ARCH_X86, UC_MODE_32)
    image = p.get_memory_mapped_image()
    uc.mem_map(0x10000000, (len(image) + 4095) // 4096 * 4096)
    uc.mem_write(0x10000000, image)
    uc.mem_map(0, 4096)
    uc.mem_map(0x200000, 0x100000)
    uc.mem_map(0x400000, 0x100000)
    design = 0x200000
    scenario = 0x201000
    output = 0x202000
    resp = 0x203000
    tox = 0x204000
    sentinel = 0x205000
    buf = 0x206000
    uc.mem_write(design, struct.pack("<i", n))
    uc.mem_write(resp, struct.pack("<" + "i" * len(response), *response))
    uc.mem_write(tox, struct.pack("<" + "i" * len(notox), *notox))
    uc.mem_write(scenario, struct.pack("<dd", mean, window))
    uc.mem_write(scenario + 0x28, struct.pack("<4d", *prob))
    # Replace external draws; retain the original exponential scale multiplication.
    uc.mem_write(0x10015030, b"\xdd\x05" + struct.pack("<I", buf) + b"\xc3")
    uc.mem_write(0x10014FA0, b"\xdd\x05" + struct.pack("<I", buf + 8) + b"\xc3")
    uc.mem_write(0x10025516, b"\xc3")
    uni = iter(uniform)
    exp = iter(unit_exponential)
    calls = []

    def ret(value, pop):
        sp = uc.reg_read(UC_X86_REG_ESP)
        target = struct.unpack("<I", uc.mem_read(sp, 4))[0]
        uc.reg_write(UC_X86_REG_EAX, value)
        uc.reg_write(UC_X86_REG_ESP, sp + 4 + pop)
        uc.reg_write(UC_X86_REG_EIP, target)

    def hook(uc, addr, size, _):
        if addr == sentinel:
            uc.emu_stop()
        elif addr == 0x10010FD0:
            vec = uc.reg_read(UC_X86_REG_ECX)
            assert vec in (design + 0xC, design + 0x38)
            ret(len(response) if vec == design + 0xC else len(notox), 0)
        elif addr == 0x10017630:
            vec = uc.reg_read(UC_X86_REG_ECX)
            assert vec in (design + 0xC, design + 0x38)
            sp = uc.reg_read(UC_X86_REG_ESP)
            i = struct.unpack("<i", uc.mem_read(sp + 4, 4))[0]
            assert 0 <= i < (len(response) if vec == design + 0xC else len(notox))
            ret((resp if vec == design + 0xC else tox) + i * 4, 4)
        elif addr == 0x10015030:
            v = next(exp)
            uc.mem_write(buf, struct.pack("<d", v))
            calls.append(("exponential", v))
        elif addr == 0x10014FA0:
            v = next(uni)
            uc.mem_write(buf + 8, struct.pack("<d", v))
            calls.append(("uniform", v))

    uc.hook_add(UC_HOOK_CODE, hook)
    sp = 0x4F0000
    uc.mem_write(sp, struct.pack("<5I", sentinel, design, scenario, output, 0))
    uc.reg_write(UC_X86_REG_ESP, sp)
    uc.emu_start(0x100064D0, sentinel, timeout=1000000, count=100000)
    if uc.reg_read(UC_X86_REG_EIP) != sentinel:
        raise RuntimeError("native emulation did not finish within its execution budget")
    return struct.unpack("<4id", uc.mem_read(output, 24)), calls


def references(dll):
    import numpy as np

    rng = np.random.default_rng(6647521)
    output = []
    configurations = [
        ("all_endpoints_disabled", [], [], [0.25] * 4),
        ("response_only", [3, 5, 7, 9, 11, 13], [], [0.1, 0.2, 0.2, 0.5]),
        ("toxicity_only", [], [3, 5, 7, 9, 11, 13], [0.2, 0.1, 0.5, 0.2]),
        ("both", [3, 5, 7, 9, 11, 13], [3, 5, 7, 9, 11, 13], [0.1, 0.1, 0.5, 0.3]),
    ]
    for name, res, tox, prob in configurations:
        for index in range(8):
            cap = 6 + index % 5
            settings = dict(
                n=cap,
                response=[min(cap + 1, x) for x in res],
                notox=[min(cap + 1, x) for x in tox],
                prob=prob,
                mean=[0.2, 1.0, 3.0, 0.7][index % 4],
                window=[2.0, 0.5, 4.0, 8.0][index % 4],
            )
            if not settings["response"]:
                settings["response"] = [cap + 1]
            if not settings["notox"]:
                settings["notox"] = [cap + 1]
            settings["uniform"] = rng.uniform(0.0001, 0.9999, cap).tolist()
            settings["unit_exponential"] = rng.exponential(size=1000).tolist()
            native, calls = run(dll, **settings)
            exp_count = sum(k == "exponential" for k, v in calls)
            u_count = sum(k == "uniform" for k, v in calls)
            settings["uniform"] = settings["uniform"][:u_count]
            settings["unit_exponential"] = settings["unit_exponential"][:exp_count]
            output.append(
                dict(
                    case=f"{name}_{index}",
                    inputs=settings,
                    native=dict(
                        sample_size=native[0],
                        responses=native[1],
                        toxicities=native[2],
                        balks=native[3],
                        duration=native[4],
                        uniforms_consumed=u_count,
                        exponentials_consumed=exp_count,
                    ),
                )
            )
    # Exactly representable category boundaries and a clipped long response.
    for name, u, exp in [
        ("category_boundaries", [0.25, 0.5, 0.75], [10.0, 0.1, 0.5, 1.0]),
        ("long_clipped_responses", [0.1, 0.1, 0.1], [20.0, 0.1, 30.0, 0.1, 40.0]),
    ]:
        settings = dict(
            n=3,
            response=[4],
            notox=[4],
            prob=[0.25] * 4,
            mean=1.0,
            window=2.0,
            uniform=u,
            unit_exponential=exp,
        )
        native, calls = run(dll, **settings)
        output.append(
            dict(
                case=name,
                inputs=settings,
                native=dict(
                    sample_size=native[0],
                    responses=native[1],
                    toxicities=native[2],
                    balks=native[3],
                    duration=native[4],
                    uniforms_consumed=sum(k == "uniform" for k, v in calls),
                    exponentials_consumed=sum(k == "exponential" for k, v in calls),
                ),
            )
        )
    return dict(
        dll_sha256=DLL_SHA256,
        native_entry_rva="0x64d0",
        native_exponential_mean_rva="0x15150",
        emulator="Unicorn 2.1.4 x86-32",
        reference_scope=(
            "Original duration machine instructions, deterministic external RNG inputs; "
            "no Windows application or RNG parity claimed"
        ),
        cases=output,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = references(args.dll)
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/multc-lean-native-duration.json"
    if args.check:
        expected = json.loads(path.read_text())
        assert result == expected
        print("Verified 34 original native duration cases with controlled external draws")
    else:
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {path.name}")


if __name__ == "__main__":
    main()
