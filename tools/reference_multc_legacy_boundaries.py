"""Original Multc Lean compact-boundary CONTROL references, base-R predicates.

The posterior predicate is replaced with independently saved base-R values;
this verifies native loop, vector and prior/cohort/cap precedence, not the
native numerical integrator. The toxicity complementation instructions run
unchanged except for their default-struct initializer. Research-only pefile
and Unicorn dependencies; original DLL is never distributed.
"""

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP

DLL_SHA256 = "8ac45568e87530d4bb6732eb8807a22e4e8c5de52a5557a14f750c24867fa807"
ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "tests/fixtures/multc-lean-boundary-r-tails.csv"
OUTPUT = ROOT / "tests/fixtures/multc-lean-native-boundaries.json"


def machine(dll):
    if hashlib.sha256(dll.read_bytes()).hexdigest() != DLL_SHA256:
        raise ValueError("native DLL checksum mismatch")
    pe = pefile.PE(str(dll))
    image = pe.get_memory_mapped_image()
    uc = Uc(UC_ARCH_X86, UC_MODE_32)
    uc.mem_map(0x10000000, (len(image) + 4095) // 4096 * 4096)
    uc.mem_write(0x10000000, image)
    uc.mem_map(0, 4096)
    uc.mem_map(0x200000, 0x100000)
    uc.mem_map(0x400000, 0x100000)
    return uc


def parameters(row):
    constant = bool(row["h"])
    return struct.pack(
        "<?7x7d",
        constant,
        float(row["h"] or 0),
        float(row["ha"] or 0),
        float(row["hb"] or 0),
        float(row["a"]),
        float(row["b"]),
        float(row["cutoff"]),
        float(row["delta"]),
    )


def execute(dll, row, probabilities):
    uc = machine(dll)
    param, vector, data, sentinel = 0x200000, 0x201000, 0x202000, 0x205000
    uc.mem_write(param, parameters(row))
    cap = int(row["cap"])
    length = 0
    calls = []

    def ret(value, pop):
        sp = uc.reg_read(UC_X86_REG_ESP)
        target = struct.unpack("<I", uc.mem_read(sp, 4))[0]
        uc.reg_write(UC_X86_REG_EAX, value)
        uc.reg_write(UC_X86_REG_ESP, sp + 4 + pop)
        uc.reg_write(UC_X86_REG_EIP, target)

    def hook(uc, address, size, _):
        nonlocal length
        sp = uc.reg_read(UC_X86_REG_ESP)
        if address == sentinel:
            uc.emu_stop()
        elif address == 0x10006340:
            pointer = struct.unpack("<I", uc.mem_read(sp + 4, 4))[0]
            a, b = struct.unpack("<2d", uc.mem_read(pointer + 0x20, 16))
            successes, failures = a - float(row["a"]), b - float(row["b"])
            i, f = round(successes), round(failures)
            assert abs(successes - i) < 1e-12 and abs(failures - f) < 1e-12
            n = i + f
            probability = probabilities[n, i]
            calls.append([n, i, probability])
            ret(int(probability > float(row["cutoff"])), 0)
        elif address == 0x10011670:
            assert uc.reg_read(UC_X86_REG_ECX) == vector
            length = struct.unpack("<i", uc.mem_read(sp + 4, 4))[0]
            assert 1 <= length <= cap
            ret(0, 12)
        elif address == 0x100113D0:
            assert uc.reg_read(UC_X86_REG_ECX) == vector
            index = struct.unpack("<i", uc.mem_read(sp + 4, 4))[0]
            assert 0 <= index < length
            ret(data + 4 * index, 4)

    uc.hook_add(UC_HOOK_CODE, hook)
    sp = 0x4F0000
    uc.mem_write(
        sp,
        struct.pack("<6I", sentinel, cap, int(row["minimum"]), int(row["cohort"]), param, vector),
    )
    uc.reg_write(UC_X86_REG_ESP, sp)
    uc.emu_start(0x10007480, sentinel, timeout=1000000, count=100000)
    if uc.reg_read(UC_X86_REG_EIP) != sentinel:
        raise RuntimeError("native compact-boundary execution exceeded its budget")
    return list(struct.unpack("<" + "i" * length, uc.mem_read(data, 4 * length))), calls


def complement(dll, row):
    uc = machine(dll)
    source, dest, sentinel = 0x200000, 0x201000, 0x205000
    uc.mem_write(source, parameters(row))
    uc.mem_write(0x100047E0, b"\xc3")  # Unrelated default-struct initialization.

    def hook(uc, address, size, _):
        if address == sentinel:
            uc.emu_stop()

    uc.hook_add(UC_HOOK_CODE, hook)
    sp = 0x4F0000
    uc.mem_write(sp, struct.pack("<3I", sentinel, dest, source))
    uc.reg_write(UC_X86_REG_ESP, sp)
    uc.emu_start(0x100059B0, sentinel, timeout=1000000, count=1000)
    if uc.reg_read(UC_X86_REG_EIP) != sentinel:
        raise RuntimeError("native complementation exceeded its execution budget")
    return list(struct.unpack("<?7x7d", uc.mem_read(dest, 64)))


def summarize(dll, case):
    """Execute original study aggregation, substituting trial/service calls."""
    uc = machine(dll)
    sentinel, scenario, output, data = 0x205000, 0x206000, 0x207000, 0x210000
    row = case["response"]["parameters"]
    cap = int(row["cap"])
    # Scenario seed is 17 and trial count is three, including prior-screen cases.
    uc.mem_write(scenario + 0x18, struct.pack("<2i", 17, 3))
    uc.mem_write(0x10025516, b"\xc3")
    vectors = {}
    probability_length = 0
    trial_index = seed_calls = 0

    def ret(value, pop):
        sp = uc.reg_read(UC_X86_REG_ESP)
        target = struct.unpack("<I", uc.mem_read(sp, 4))[0]
        uc.reg_write(UC_X86_REG_EAX, value)
        uc.reg_write(UC_X86_REG_ESP, sp + 4 + pop)
        uc.reg_write(UC_X86_REG_EIP, target)

    def hook(uc, address, size, _):
        nonlocal probability_length, trial_index, seed_calls
        sp = uc.reg_read(UC_X86_REG_ESP)
        vec = uc.reg_read(UC_X86_REG_ECX)
        if address == sentinel:
            uc.emu_stop()
        elif address in (0x10007090, 0x10010FE0):
            ret(0, 0)  # Scenario validation / vector destruction.
        elif address == 0x10011AA0:
            vectors[vec] = []
            ret(0, 4)
        elif address == 0x10005FE0:
            rvec, tvec = struct.unpack("<2I", uc.mem_read(sp + 24, 8))
            vectors[rvec] = case["response"]["vector"]
            vectors[tvec] = case["toxicity"]["vector"]
            ret(0, 0)
        elif address == 0x100113D0:
            index = struct.unpack("<i", uc.mem_read(sp + 4, 4))[0]
            assert vec in vectors and 0 <= index < len(vectors[vec])
            uc.mem_write(data + 0x10000, struct.pack("<i", vectors[vec][index]))
            ret(data + 0x10000, 4)
        elif address in (0x10014F40, 0x10014F60):
            seed_calls += 1
            ret(0, 0)
        elif address == 0x1000F7A0:
            assert vec == output + 0x28
            probability_length = struct.unpack("<i", uc.mem_read(sp + 4, 4))[0]
            assert probability_length == cap + 1
            uc.mem_write(data, bytes(probability_length * 8))
            ret(0, 12)
        elif address == 0x10011520:
            index = struct.unpack("<i", uc.mem_read(sp + 4, 4))[0]
            assert vec == output + 0x28 and 0 <= index < probability_length
            ret(data + index * 8, 4)
        elif address == 0x100064D0:
            assert trial_index < len(case["duration_cases"])
            record = case["duration_cases"][trial_index]["native"]
            result = struct.unpack("<I", uc.mem_read(sp + 12, 4))[0]
            uc.mem_write(
                result,
                struct.pack(
                    "<4id",
                    record["sample_size"],
                    record["responses"],
                    record["toxicities"],
                    record["balks"],
                    record["duration"],
                ),
            )
            trial_index += 1
            ret(0, 0)

    uc.hook_add(UC_HOOK_CODE, hook)
    sp = 0x4F0000
    uc.mem_write(
        sp,
        struct.pack(
            "<9I", sentinel, cap, int(row["minimum"]), int(row["cohort"]), 0, 0, scenario, output, 0
        ),
    )
    uc.reg_write(UC_X86_REG_ESP, sp)
    uc.emu_start(0x100075B0, sentinel, timeout=1000000, count=100000)
    if uc.reg_read(UC_X86_REG_EIP) != sentinel:
        raise RuntimeError("native study-summary execution exceeded its budget")
    fields = (
        "mean_duration",
        "mean_sample_size",
        "mean_responses",
        "mean_toxicities",
        "mean_balks",
    )
    result = dict(zip(fields, struct.unpack("<5d", uc.mem_read(output, 40)), strict=True))
    result["sample_size_probability"] = list(
        struct.unpack("<" + "d" * probability_length, uc.mem_read(data, probability_length * 8))
    )
    result["trials_consumed"] = trial_index
    result["seed_calls"] = seed_calls
    return result


def references(dll):
    import numpy as np
    from reference_multc_legacy_duration import run as run_duration

    rng = np.random.default_rng(748059)
    with TABLE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cases = []
    for name in dict.fromkeys(row["case"] for row in rows):
        case = {"case": name}
        for endpoint in ("response", "toxicity"):
            selected = [r for r in rows if r["case"] == name and r["endpoint"] == endpoint]
            row = selected[0]
            probs = {(int(r["n"]), int(r["i"])): float(r["probability"]) for r in selected}
            vector, calls = execute(dll, row, probs)
            case[endpoint] = {"parameters": row, "vector": vector, "predicate_calls": calls}
            # Save the original complement mapping. Unused union fields are
            # zeroed by the initializer and are not model parameters.
            case[endpoint]["native_complement"] = complement(dll, row)
        case["duration_cases"] = []
        if case["response"]["vector"] != [0] and case["toxicity"]["vector"] != [0]:
            for prob in ([0.1, 0.1, 0.5, 0.3], [0.2, 0.5, 0.1, 0.2], [0.25] * 4):
                row = case["response"]["parameters"]
                cap = int(row["cap"])
                inputs = dict(
                    n=cap,
                    response=case["response"]["vector"],
                    notox=case["toxicity"]["vector"],
                    prob=prob,
                    mean=0.2,
                    window=2.0,
                    uniform=rng.uniform(0.0001, 0.9999, cap).tolist(),
                    unit_exponential=rng.exponential(size=2000).tolist(),
                )
                native, calls = run_duration(dll, **inputs)
                u_count = sum(k == "uniform" for k, v in calls)
                exp_count = sum(k == "exponential" for k, v in calls)
                inputs["uniform"] = inputs["uniform"][:u_count]
                inputs["unit_exponential"] = inputs["unit_exponential"][:exp_count]
                case["duration_cases"].append(
                    dict(
                        inputs=inputs,
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
        case["native_study_summary"] = summarize(dll, case)
        cases.append(case)
    return {
        "dll_sha256": DLL_SHA256,
        "r_table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest(),
        "scope": (
            "Original boundary control/complement instructions; "
            "substituted base-R posterior predicate"
        ),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = references(args.dll)
    if args.check:
        assert result == json.loads(OUTPUT.read_text())
        print(f"Verified {len(result['cases'])} native boundary-control cases and complementation")
    else:
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {OUTPUT.name}: {len(result['cases'])} cases")


if __name__ == "__main__":
    main()
