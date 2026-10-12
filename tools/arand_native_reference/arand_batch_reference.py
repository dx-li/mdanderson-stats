"""Original batch construction slice (952..1071) and reset slice (154..179)."""

import json
import math

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(a, b, maximum):
    memory = {1848: a, 1856: b}
    local = {}
    increments = []
    for rva, start, stop in [(0xB3818, 952, 1072), (0xB418C, 154, 180)]:
        instructions = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(rva, 100000))).instructions
        instructions = [i for i in instructions if start <= i.offset < stop]
        positions = {i.offset: j for j, i in enumerate(instructions)}
        stack = []
        pc = 0
        for _ in range(100000):
            if pc == len(instructions):
                break
            i = instructions[pc]
            pc += 1
            op, v = i.mnemonic, i.operand
            if op == "ldarg.0":
                stack.append(1000)
            elif op.startswith("ldc.i4."):
                stack.append(int(v) if op.endswith(".s") else int(op.rsplit(".", 1)[1]))
            elif op in ("ldc.i4", "ldc.r8"):
                stack.append(v)
            elif op.startswith(("ldloc", "stloc")):
                key = v.index if v is not None else int(op.rsplit(".", 1)[1])
                if op.startswith("ldloca"):
                    stack.append(100000 + key)
                    memory[100000 + key] = local[key]
                elif op.startswith("ldloc"):
                    stack.append(local[key])
                else:
                    local[key] = stack.pop()
            elif op.startswith("ldind"):
                stack.append(memory[stack.pop()])
            elif op.startswith("stind"):
                value, address = stack.pop(), stack.pop()
                memory[address] = value
            elif op in ("add", "mul"):
                right, left = stack.pop(), stack.pop()
                stack.append(left + right if op == "add" else left * right)
            elif op == "neg":
                stack[-1] = -stack[-1]
            elif op.startswith("conv."):
                stack[-1] = int(stack[-1]) if op == "conv.i4" else float(stack[-1])
            elif op in ("bge.s", "blt.s"):
                right, left = stack.pop(), stack.pop()
                if (left >= right) if op == "bge.s" else (left < right):
                    pc = positions[v]
            elif op == "br":
                assert v == 1747
                break
            elif op == "call":
                name = str(p.net.mdtables.tables[v.table].rows[v.rid - 1].Name)
                if name in ("Ceiling", "Exp"):
                    stack.append(
                        math.ceil(stack.pop()) if name == "Ceiling" else math.exp(stack.pop())
                    )
                elif name.endswith("GetMaxQtyPatsInTrial"):
                    stack.pop()
                    stack.append(maximum)
                elif name.endswith(".push_back"):
                    address = stack.pop()
                    stack.pop()
                    increments.append(memory[address])
                else:
                    raise ValueError(name)
            else:
                raise ValueError((i.offset, op))
        else:
            raise ValueError("budget")
        if rva == 0xB3818:
            initial = memory[1764]
    return dict(
        a=a,
        b=b,
        maximum=maximum,
        initial=initial,
        reset=memory[1764],
        increments=increments,
        batch_index=memory[1876],
    )


if __name__ == "__main__":
    for a, b, maximum in [
        (3, 0, 12),
        (3.5, 0, 12),
        (5, 0.1, 30),
        (0.5, 0, 6),
        (10, 0.03, 100),
        (2, -0.1, 20),
    ]:
        print(json.dumps(run(a, b, maximum), sort_keys=True))
