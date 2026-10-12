"""Original managed cohort-replay gate IL with supplied patient/model getters.

Executes IL346..588 of CRMTrialConductor.GetDecision. Stops at entry to the
strict-cohort replay block; posterior evaluation and descriptive strings are
outside this slice. Integer native status 0 means treatment is allowed.
"""

import json

from patient_reference import CilMethodBody, CilMethodBodyReaderBytes, legacy, name, p

instructions = CilMethodBody(
    CilMethodBodyReaderBytes(p.get_data(0xE620 if legacy else 0xE634, 10000))
).instructions
positions = {i.offset: n for n, i in enumerate(instructions)}


def run(cohorts, size, status):
    result = {"status": status, "cohort": None}
    patients = [{"cohort": c, "dose": 1} for c in cohorts]
    args = {1: {"size": size}, 4: patients}
    locals_ = {1: result}
    stack = []
    pc = positions[346]
    for _ in range(10000):
        i = instructions[pc]
        if i.offset in (594, 1310, 1377, 1380):
            assert not stack
            return {
                "cohorts": cohorts,
                "size": size,
                "status": status,
                "next_cohort": result["cohort"],
                "replay": i.offset == 594,
                "prefix": locals_.get(2) if i.offset == 594 else None,
            }
        if i.offset == 549:
            pc = positions[582]
            continue
        pc += 1
        op, v = i.mnemonic, i.operand
        if op.startswith(("ldloc", "stloc")):
            key = v.index if v is not None else int(op.rsplit(".", 1)[1])
            if op.startswith("ldloc"):
                stack.append(locals_[key])
            else:
                locals_[key] = stack.pop()
        elif op.startswith("ldarg"):
            key = v.index if v is not None else int(op.rsplit(".", 1)[1])
            stack.append(args[key])
        elif op.startswith("ldc.i4"):
            stack.append(v if v is not None else int(op.rsplit(".", 1)[1]))
        elif op == "dup":
            stack.append(stack[-1])
        elif op in ("add", "sub"):
            b, a = stack.pop(), stack.pop()
            stack.append(a + b if op == "add" else a - b)
        elif op in ("call", "callvirt"):
            method = name(v)
            if method == "get_Status":
                stack.append(stack.pop()["status"])
            elif method == "get_StartingDoseIndex":
                stack.pop()
                stack.append(0)
            elif method == "get_Count":
                stack.append(len(stack.pop()))
            elif method == "get_Item":
                idx, obj = stack.pop(), stack.pop()
                stack.append(obj[idx])
            elif method == "get_Cohort":
                stack.append(stack.pop()["cohort"])
            elif method == "get_TreatmentIndex":
                stack.append(stack.pop()["dose"])
            elif method == "get_CohortSize":
                stack.append(stack.pop()["size"])
            elif method == "set_Cohort":
                value, obj = stack.pop(), stack.pop()
                obj["cohort"] = value
            elif method == "get_DoseEscalationRule":
                stack.pop()
                stack.append(1)
            else:
                raise ValueError(method)
        elif op.startswith("brfalse"):
            if not stack.pop():
                pc = positions[v]
        elif op in ("br", "br.s"):
            pc = positions[v]
        elif op.startswith(("blt", "bge", "ble", "beq", "bne")):
            b, a = stack.pop(), stack.pop()
            condition = {"blt": a < b, "bge": a >= b, "ble": a <= b, "beq": a == b, "bne": a != b}[
                op[:3]
            ]
            if condition:
                pc = positions[v]
        else:
            raise ValueError((i.offset, op, v))
    raise ValueError("instruction budget")


if __name__ == "__main__":
    for status in (0, 1, 2, 3):
        for size in (1, 3):
            for cohorts in ([], [1], [1, 1], [1, 1, 1], [1, 1, 1, 2], [1, 1, 1, 2, 2]):
                print(json.dumps(run(cohorts, size, status), sort_keys=True))
