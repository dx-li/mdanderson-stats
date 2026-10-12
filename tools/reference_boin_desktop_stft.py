"""Execute original CalculateW_STFT IL with host collections/settings services.

Requires external BOIN.exe plus dnfile/dncil. No GUI is constructed. The supplied
window is already in days; native unit conversion is a separate inspected method.
"""

import hashlib
import json
import sys
from pathlib import Path

import dnfile
from dncil.cil.body.reader import read_method_body_from_bytes

path = Path(sys.argv[1])
assert hashlib.sha256(path.read_bytes()).hexdigest() == (
    "184b8c6b61a0d2d89815a895e3f27004587f7fbbebbbaf4cc1c813f94fe23997"
)
pe = dnfile.dnPE(str(path))
instructions = read_method_body_from_bytes(pe.get_data(0xD108, 100000)).instructions
positions = {ins.offset: index for index, ins in enumerate(instructions)}


def execute(times, window, weights, uniform):
    settings = dict(
        ToxPriorProbs=weights, UseUniform_ToxPriorProbs=uniform, DLTAssessmentWindowInDAYS=window
    )
    model = {"SimDat": {"TrialDesign": settings, "TITE_BOIN_PendingPatientData": times}}
    frame = {"Model": model}
    local, stack = {}, []
    pc = 0
    for _ in range(10000):
        ins = instructions[pc]
        op, operand = ins.opcode.name, ins.operand
        pc += 1
        if op == "ldarg.0":
            stack.append(frame)
        elif op.startswith("ldloc"):
            index = int(op[-1]) if op[-1].isdigit() else operand.index
            stack.append(local[index])
        elif op.startswith("stloc"):
            index = int(op[-1]) if op[-1].isdigit() else operand.index
            local[index] = stack.pop()
        elif op == "ldc.r8":
            stack.append(operand)
        elif op.startswith("ldc.i4"):
            stack.append(int(op[-1]) if op[-1].isdigit() else int(operand))
        elif op == "dup":
            stack.append(stack[-1])
        elif op == "newobj":
            stack.append([])
        elif op in ("call", "callvirt"):
            name = str(pe.net.mdtables.tables[operand.table].rows[operand.rid - 1].Name)
            if name == "Add":
                value = stack.pop()
                stack.pop().append(value)
            elif name == "Min":
                right, left = stack.pop(), stack.pop()
                stack.append(min(left, right))
            elif name == "get_Item":
                index = stack.pop()
                stack.append(stack.pop()[index])
            elif name == "GetEnumerator":
                stack.append({"items": stack.pop(), "index": -1})
            elif name == "MoveNext":
                iterator = stack.pop()
                iterator["index"] += 1
                stack.append(iterator["index"] < len(iterator["items"]))
            elif name == "get_Current":
                iterator = stack.pop()
                stack.append(iterator["items"][iterator["index"]])
            elif name.startswith("get_"):
                stack.append(stack.pop()[name[4:]])
            else:
                raise AssertionError(name)
        elif op in ("div", "mul", "add", "sub"):
            right, left = stack.pop(), stack.pop()
            stack.append(
                {
                    "div": lambda: left / right,
                    "mul": lambda: left * right,
                    "add": lambda: left + right,
                    "sub": lambda: left - right,
                }[op]()
            )
        elif op in ("br.s", "br", "leave.s"):
            pc = positions[operand]
        elif op in ("brfalse.s", "brtrue.s"):
            value = bool(stack.pop())
            if value == (op == "brtrue.s"):
                pc = positions[operand]
        elif op == "bgt.s":
            right, left = stack.pop(), stack.pop()
            if left > right:
                pc = positions[operand]
        elif op == "ret":
            return stack.pop()
        else:
            raise AssertionError(op)
    raise RuntimeError("bounded IL execution exceeded instruction budget")


rows = []
for window in (30.0, 90.0, 91.31):
    for weights in ([1 / 3] * 3, [0.1, 0.2, 0.7]):
        for uniform in (False, True):
            for fraction in (0.0, 0.125, 1 / 3, 0.5, 2 / 3, 0.999, 1.0):
                times = [fraction * window, window / 4, window / 2]
                rows.append(
                    dict(
                        window_days=window,
                        weights=weights,
                        uniform=uniform,
                        followup_days=times,
                        total=execute(times, window, weights, uniform),
                    )
                )
print(json.dumps({"rva": "0xd108", "cases": rows}, indent=2))
