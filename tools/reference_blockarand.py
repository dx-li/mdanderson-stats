"""Audit rational planning/stopping by interpreting methods from the archived DLL.

Run with uv run --with dnfile --with dncil python tools/reference_blockarand.py.
The DLL is read from ignored research files and is never redistributed. This
small interpreter handles only the arithmetic/branch subset of the two methods;
unsupported instructions raise rather than silently approximating execution.
"""

import hashlib
import json
from pathlib import Path

import dnfile
import numpy as np
from dncil.cil.body.reader import read_method_body_from_bytes

from mdanderson_stats.blockarand import BlockArandDesign, _decision, blockarand_plan

root = Path("research/raw/BlockARAND")
pe = dnfile.dnPE(str(root / "extracted/BlockArandLib.dll"))
methods = {str(m.Name): m for m in pe.net.mdtables.MethodDef}


def execute(name, args):
    method = methods[name]
    code = read_method_body_from_bytes(pe.get_data(method.Rva)).instructions
    offsets = {instruction.offset: i for i, instruction in enumerate(code)}
    stack, locals_ = [], {}
    pc, steps = 0, 0
    while True:
        instruction = code[pc]
        op, arg = str(instruction.opcode), instruction.operand
        pc += 1
        steps += 1
        if steps > 100000:
            raise RuntimeError("IL step limit")
        if op.startswith("ldarg") or op.startswith("ldloc") or op.startswith("stloc"):
            index = arg.index if arg is not None else int(op.rsplit(".", 1)[1])
            if op.startswith("ldarg"):
                stack.append(args[index])
            elif op.startswith("ldloc"):
                stack.append(locals_[index])
            else:
                locals_[index] = stack.pop()
        elif op.startswith("ldc"):
            stack.append(arg if arg is not None else -1 if op.endswith("m1") else int(op[-1]))
        elif op == "ldfld":
            field = str(pe.net.mdtables.Field[(arg.value & 0xFFFFFF) - 1].Name)
            stack.append(stack.pop()[field])
        elif op.startswith("stind"):
            value, ref = stack.pop(), stack.pop()
            ref[0] = value
        elif op == "conv.r8":
            stack.append(float(stack.pop()))
        elif op in ("add", "sub", "div"):
            b, a = stack.pop(), stack.pop()
            stack.append(a + b if op == "add" else a - b if op == "sub" else a / b)
        elif op == "call":
            target = str(pe.net.mdtables.MemberRef[(arg.value & 0xFFFFFF) - 1].Name)
            if target != "Abs":
                raise RuntimeError(f"unsupported call {target}")
            stack.append(abs(stack.pop()))
        elif op == "br.s":
            pc = offsets[arg]
        elif op.startswith(("bge", "blt", "ble")):
            b, a = stack.pop(), stack.pop()
            jump = a >= b if op.startswith("bge") else a < b if op.startswith("blt") else a <= b
            if jump:
                pc = offsets[arg]
        elif op == "ret":
            return
        else:
            raise RuntimeError(f"unsupported instruction {op}")


plans = 0
for minimum in range(2, 11):
    for maximum in range(minimum, 11):
        # Include exact fractions and midpoints, not just generic random values.
        fractions = sorted({k / n for n in range(minimum, maximum + 1) for k in range(1, n)})
        grid = np.concatenate(
            (np.linspace(0, 1, 101), fractions, (np.array(fractions[1:]) + fractions[:-1]) / 2)
        )
        for p in grid:
            numerator, denominator = [None], [None]
            execute(
                "BestRationalApproximation",
                [None, float(p), minimum, maximum, numerator, denominator],
            )
            result = blockarand_plan(p, min_block=minimum, max_block=maximum)
            assert (result.arm_zero_count, result.size) == (numerator[0], denominator[0])
            plans += 1

stops = 0
for early, final in [(0.95, 0.9), (1, 1), (0.4, 0.3), (0.9, 0.95)]:
    design = BlockArandDesign(max_patients=20, burnin=5, early_cutoff=early, final_cutoff=final)
    source_design = {
        "earlyStoppingProbability": early,
        "finalDecisionProbability": final,
        "maxPatients": 20,
    }
    for n in (0, 1, 5, 19, 20):
        for p in np.unique(
            [
                0,
                0.05,
                0.1,
                0.5,
                0.9,
                0.95,
                1,
                early,
                final,
                np.nextafter(early, 1),
                np.nextafter(final, 1),
            ]
        ):
            stop, winner = [None], [None]
            execute(
                "CheckStoppingRule",
                [{"m_design": source_design}, p, {"patientsTreated": n}, stop, winner],
            )
            actual = _decision(p, 1 - p, 0, n, design)
            assert actual.stopped == bool(stop[0])
            assert actual.selected == (None if winner[0] == -1 else winner[0])
            stops += 1

report = {
    "archive_sha256": hashlib.sha256((root / "archive.zip").read_bytes()).hexdigest(),
    "rational_plan_cases": plans,
    "stopping_cases": stops,
    "method": "Interpret original DLL arithmetic and branches; compare Python outputs",
    "scope": "Block selection and stopping parity; not original RNG or beta integrator parity",
}
Path("docs/blockarand-reference.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
