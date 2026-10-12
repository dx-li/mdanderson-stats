"""Original inverse-CDF assignment IL with a supplied uniform and bounded vector.

The caller chooses the vector population. This executes GetTreat (RVA0x6ccb4),
not the eligible-map or forced-equal caller and not the original RNG.
"""

import json
import math

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(weights, uniform):
    ins = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(0x6CCB4, 10000))).instructions
    positions = {x.offset: i for i, x in enumerate(ins)}
    memory = {2008: len(weights)}
    memory.update({3000 + 8 * i: x for i, x in enumerate(weights)})
    stack, local = [], {}
    pc = 0
    for _ in range(1000):
        x = ins[pc]
        pc += 1
        op, v = x.mnemonic, x.operand
        if op.startswith("ldarg."):
            stack.append(1000 if op == "ldarg.0" else 2000)
        elif op == "ldc.i4":
            stack.append(v)
        elif op.startswith("ldc.i4."):
            stack.append(int(op.rsplit(".", 1)[1]))
        elif op.startswith(("ldloc.", "stloc.")):
            index = int(op.rsplit(".", 1)[1])
            if op.startswith("ldloc"):
                stack.append(local[index])
            else:
                local[index] = stack.pop()
        elif op == "add":
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif op.startswith("ldind."):
            stack.append(memory[stack.pop()])
        elif op == "call":
            name = str(p.net.mdtables.tables[v.table].rows[v.rid - 1].Name)
            b, a = stack.pop(), stack.pop()
            if name.endswith(".GetUniform"):
                assert (a, b) == (1140, 1144)
                stack.append(uniform)
            elif name.endswith(".()"):
                assert a == 2000 and 0 <= b < len(weights)
                stack.append(3000 + 8 * b)
            else:
                raise ValueError(name)
        elif op in ("ble.un.s", "bge.s", "bgt.s"):
            b, a = stack.pop(), stack.pop()
            if {"ble.un.s": a <= b, "bge.s": a >= b, "bgt.s": a > b}[op]:
                pc = positions[v]
        elif op == "ret":
            return {"weights": weights, "uniform": uniform, "index": stack.pop()}
        else:
            raise ValueError((x.offset, op))
    raise RuntimeError("instruction budget exhausted")


if __name__ == "__main__":
    for weights, uniform in [
        ([1.0], 0.0),
        ([0.5, 0.5], 0.0),
        ([0.5, 0.5], 0.5),
        ([0.5, 0.5], math.nextafter(0.5, 1)),
        ([0, 0.5, 0.5], 0.0),
        ([0, 0.5, 0.5], math.nextafter(0.0, 1)),
        ([0.25, 0.75], math.nextafter(1.0, 0)),
    ]:
        print(json.dumps(run(weights, uniform), sort_keys=True))
