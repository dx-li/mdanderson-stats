"""Execute original managed current EffTox completed-cohort prefix instructions.

Original GetDecision IL12..140 executes; stop before model construction at141.
Supplied field/array storage stands in for managed runtime objects. No posterior,
GUI, native RNG or full method execution is claimed by this isolated probe.
"""

import hashlib
import json
import sys

import dnfile
from dncil.cil.body.reader import read_method_body_from_bytes

assert hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() == (
    "6b266b3927bec98fd6d64f5767a1110cfccc3fe6bd18c7d438b9cfd538e4479c"
)
pe = dnfile.dnPE(sys.argv[1])
ins = read_method_body_from_bytes(pe.get_data(0x6CA4, 10000)).instructions
positions = {i.offset: n for n, i in enumerate(ins)}


def name(token):
    return str(pe.net.mdtables.tables[token.table].rows[token.rid - 1].Name)


def run(cohorts, size):
    result = {}
    local = {22: result}
    args = {1: {"m_cohortSize": size}, 2: [{"m_iCohortNumber": n} for n in cohorts]}
    stack = []
    pc = positions[12]
    for _ in range(10000):
        i = ins[pc]
        if i.offset == 141:
            assert not stack
            return dict(
                cohorts=cohorts,
                size=size,
                prefix=local[4],
                new_cohort=bool(local[6]),
                next_cohort=result["iCohortNumber"],
            )
        pc += 1
        op, v = i.mnemonic, i.operand
        if op.startswith(("ldloc", "stloc")):
            key = v.index if v is not None else int(op.rsplit(".", 1)[1])
            if op.startswith("ldloc"):
                stack.append(local[key])
            else:
                local[key] = stack.pop()
        elif op.startswith("ldarg"):
            key = v.index if v is not None else int(op.rsplit(".", 1)[1])
            stack.append(args[key])
        elif op.startswith("ldc.i4"):
            stack.append(v if v is not None else int(op.rsplit(".", 1)[1]))
        elif op == "initobj":
            stack.pop().clear()
        elif op == "ldlen":
            stack.append(len(stack.pop()))
        elif op == "ldelema":
            idx, array = stack.pop(), stack.pop()
            stack.append(array[idx])
        elif op == "ldfld":
            stack.append(stack.pop()[name(v)])
        elif op == "stfld":
            value, obj = stack.pop(), stack.pop()
            obj[name(v)] = value
        elif op in ("add", "sub"):
            b, a = stack.pop(), stack.pop()
            stack.append(a + b if op == "add" else a - b)
        elif op.startswith("brtrue"):
            if stack.pop():
                pc = positions[v]
        elif op in ("br", "br.s"):
            pc = positions[v]
        elif op.startswith(("blt", "bge", "bne")):
            b, a = stack.pop(), stack.pop()
            if {"blt": a < b, "bge": a >= b, "bne": a != b}[op[:3]]:
                pc = positions[v]
        else:
            raise ValueError((i.offset, op, v))
    raise ValueError("instruction budget")


rows = []
for size in (1, 2, 3):
    for cohorts in (
        [],
        [0],
        [0, 0],
        [0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [3, 3, 4],
        [3, 4, 3],
    ):
        rows.append(run(cohorts, size))
print(json.dumps(rows, indent=2))
