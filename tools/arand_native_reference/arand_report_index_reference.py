"""Execute original report percentile index instructions on a supplied sorted vector.

Set ARAND_REFERENCE_LIBRARY to the hash-pinned library. Only the report's
927..1092 IL slice is executed: vector bounds/data and Math.Ceiling are supplied.
No sorting, whole report, wrapper dispatch or RNG equivalence is claimed.
"""

import json
import math

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(size):
    if not isinstance(size, int) or not 0 <= size <= 10000:
        raise ValueError("size must be an integer in 0..10000")
    instructions = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(0x60A14, 100000))).instructions
    selected = [i for i in instructions if 927 <= i.offset < 1093]
    memory = {1240: 10000, 1244: 10000 + 8 * size}
    memory.update({10000 + 8 * i: float(i) for i in range(size)})
    local = {0x1A: 0.0, 0x19: 0.0}
    stack = []
    reads = []
    for ins in selected:
        op, operand = ins.mnemonic, ins.operand
        if op == "ldarg.0":
            stack.append(1000)
        elif op.startswith("ldc.i4."):
            stack.append(int(operand) if op.endswith(".s") else int(op.rsplit(".", 1)[1]))
        elif op in ("ldc.i4", "ldc.r8"):
            stack.append(operand)
        elif op.startswith(("ldloc", "stloc")):
            index = operand.index if operand is not None else int(op.rsplit(".", 1)[1])
            if op.startswith("ldloc"):
                stack.append(local[index])
            else:
                local[index] = stack.pop()
        elif op in ("add", "sub", "mul", "shr"):
            b, a = stack.pop(), stack.pop()
            stack.append(
                {
                    "add": lambda: a + b,
                    "sub": lambda: a - b,
                    "mul": lambda: a * b,
                    "shr": lambda: int(a) >> int(b),
                }[op]()
            )
        elif op in ("ldind.i4", "ldind.r8"):
            address = stack.pop()
            if op == "ldind.r8":
                reads.append((address - 10000) // 8)
            stack.append(memory[address])  # out-of-range reads fail
        elif op in ("conv.r8", "conv.r.un"):
            stack[-1] = float(stack[-1])
        elif op == "conv.i4":
            stack[-1] = int(stack[-1])
        elif op == "blt.un":
            b, a = stack.pop(), stack.pop()
            assert operand == 1093
            if a < b:
                break
        elif op == "call":
            name = str(p.net.mdtables.tables[operand.table].rows[operand.rid - 1].Name)
            assert name == "Ceiling"
            stack.append(math.ceil(stack.pop()))
        else:
            raise ValueError((ins.offset, op))
    return {"size": size, "indices": reads, "stored_values": [local[0x1A], local[0x19]]}


if __name__ == "__main__":
    for size in (0, 1, 20, 39, 40, 99, 100, 101, 120, 199, 200, 1000):
        print(json.dumps(run(size), sort_keys=True))
