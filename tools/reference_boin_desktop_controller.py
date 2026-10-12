"""Execute original BOIN 1.1.0 GetDecision IL with supplied analysis services.

Requires dnfile/dncil and the external hash-pinned calculations DLL. Runtime
lists/properties, outcome tally and R analysis/report calls are host services.
This isolates original cohort/pending/capacity precedence, not R statistics.
"""

import hashlib
import json
import sys

import dnfile
from dncil.cil.body.reader import read_method_body_from_bytes

path = sys.argv[1]
assert hashlib.sha256(open(path, "rb").read()).hexdigest() == (
    "5651ac132ae977b0026afd5110edfce13e9e8b200e24db599eb18c94900f772c"
)
pe = dnfile.dnPE(path)
body = read_method_body_from_bytes(pe.get_data(0x20FC, 100000)).instructions
positions = {i.offset: j for j, i in enumerate(body)}


def name(token):
    return str(pe.net.mdtables.tables[token.table].rows[token.rid - 1].Name)


def execute(design, patients, analysis_status):
    decision = {"ExtraInfo": "", "Status": 0, "TreatmentIndexA": -1, "TreatmentIndexB": -1}
    controller = {"_decision": decision, "_sProcessOutput": ""}
    args = [controller, design, patients, "2026-10-12"]
    local, stack, calls = {}, [], []
    pc = 0
    for _ in range(10000):
        ins = body[pc]
        pc += 1
        op, v = ins.opcode.name, ins.operand
        if op.startswith("ldarg."):
            stack.append(args[v.index if op.endswith("s") else int(op[-1])])
        elif op.startswith("ldloc."):
            stack.append(local.get(v.index if op.endswith("s") else int(op[-1]), 0))
        elif op.startswith("stloc."):
            local[v.index if op.endswith("s") else int(op[-1])] = stack.pop()
        elif op.startswith("ldc.i4."):
            stack.append(-1 if op.endswith("m1") else v if op.endswith("s") else int(op[-1]))
        elif op == "ldnull":
            stack.append(None)
        elif op == "ldfld":
            stack.append(stack.pop()[name(v)])
        elif op == "stfld":
            value, obj = stack.pop(), stack.pop()
            obj[name(v)] = value
        elif op in ("ldstr", "ldsfld"):
            # Reporting prose is immaterial and is not redistributed.
            stack.append("")
        elif op == "pop":
            stack.pop()
        elif op == "dup":
            stack.append(stack[-1])
        elif op in ("add", "sub"):
            b, a = stack.pop(), stack.pop()
            stack.append(a + b if op == "add" else a - b)
        elif op in ("call", "callvirt"):
            method = name(v)
            if method == "get_Count":
                stack.append(len(stack.pop()))
            elif method == "get_Item":
                index, obj = stack.pop(), stack.pop()
                stack.append(obj[index])
            elif method == "get_NewLine":
                stack.append("")
            elif method.startswith("get_"):
                stack.append(stack.pop()[method[4:]])
            elif method.startswith("set_"):
                value, obj = stack.pop(), stack.pop()
                obj[method[4:]] = value
            elif method == "Concat":
                # This GetDecision call uses the four-string overload.
                del stack[-4:]
                stack.append("")
            elif method == "ComputeCombinationSelectMTDUsingLocalR":
                del stack[-4:]
                calls.append("final_selection")
            elif method in ("ComputeDecision_MTDcontour", "ComputeDecision_SingleMTD"):
                stack.pop()
                calls.append("contour" if "contour" in method else "single")
                decision.update(Status=analysis_status, TreatmentIndexA=1, TreatmentIndexB=2)
            elif method == "TallyOutcome":
                stack.pop()
                calls.append("tally")
                controller["_numTotalPending"] = sum(p["OutcomeCode"] == 2 for p in patients)
            elif method in ("SetBoundaryTable", "DisplayDecisionDetails"):
                stack.pop()
            else:
                raise NotImplementedError(method)
        elif op.startswith(("brtrue", "brfalse")):
            if bool(stack.pop()) == op.startswith("brtrue"):
                pc = positions[v]
        elif op.startswith(("bge", "blt", "beq", "bne")):
            b, a = stack.pop(), stack.pop()
            if {"bge": a >= b, "blt": a < b, "beq": a == b, "bne": a != b}[op.split(".")[0]]:
                pc = positions[v]
        elif op in ("br", "br.s", "leave", "leave.s"):
            pc = positions[v]
        elif op == "ret":
            assert stack.pop() is decision
            return {
                k: decision[k] for k in ("Status", "Cohort", "TreatmentIndexA", "TreatmentIndexB")
            }, calls
        else:
            raise NotImplementedError((ins.offset, op))
    raise RuntimeError("instruction budget exhausted")


rows = []
for size in (1, 3):
    for maximum in (2, 6):
        for contour in (False, True):
            design = dict(
                CohortSize=size,
                MaximumSampleSize=maximum,
                FindContours=contour,
                StartingDoseLevel_DrugA=1,
                StartingDoseLevel_DrugB=2,
            )
            for enrolled in range(9):
                for pending in (False, True):
                    for status in (0, 2, 3, 5):
                        patients = [
                            dict(
                                Cohort=i // size + 1,
                                DoseIndexA=i // size % 2,
                                DoseIndexB=1,
                                OutcomeCode=2 if pending and i == 0 else 0,
                            )
                            for i in range(enrolled)
                        ]
                        decision, calls = execute(design, patients, status)
                        rows.append(
                            dict(
                                design=design,
                                patients=patients,
                                supplied_analysis_status=status,
                                decision=decision,
                                calls=calls,
                            )
                        )
print(json.dumps({"rva": "0x20fc", "cases": rows}, indent=2))
