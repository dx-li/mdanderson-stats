"""Execute original single-trial clock slices with a supplied gap service.

This tests initialization and the clock update, not the entire trial body.
The first GetTreatment call precedes the first gap draw in that body.
"""

import json

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(gaps):
    instructions = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(0x59360, 10000))).instructions
    memory = {1016: 999.0, 1144: 5000, 5048: 6000}
    locals_ = {2: 1016}
    stack = []
    current_gap = 0.0

    def execute(start, end):
        for instruction in instructions:
            if not start <= instruction.offset <= end:
                continue
            op, value = instruction.mnemonic, instruction.operand
            if op == "ldarg.0":
                stack.append(1000)
            elif op in ("ldc.i4", "ldc.i4.s", "ldc.r8"):
                stack.append(value)
            elif op.startswith(("ldloc", "stloc")):
                key = value.index if value is not None else int(op.rsplit(".", 1)[1])
                if op.startswith("ldloc"):
                    stack.append(locals_[key])
                else:
                    locals_[key] = stack.pop()
            elif op.startswith("ldind"):
                stack.append(memory[stack.pop()])
            elif op.startswith("stind"):
                value, address = stack.pop(), stack.pop()
                memory[address] = value
            elif op == "add":
                stack.append(stack.pop() + stack.pop())
            elif op == "dup":
                stack.append(stack[-1])
            elif op == "calli":
                target, obj = stack.pop(), stack.pop()
                assert (target, obj) == (6000, 1144)
                stack.append(current_gap)
            else:
                raise ValueError((op, value))
        assert not stack

    execute(92, 105)
    clocks = [memory[1016]]
    for current_gap in gaps:
        execute(452, 482)
        clocks.append(memory[1016])
    return {"gaps": gaps, "clocks": clocks}


if __name__ == "__main__":
    for gaps in ([], [0.25], [0.25, 1.5, 0.125], [2.0, 3.0, 4.0, 5.0]):
        print(json.dumps(run(gaps), sort_keys=True))
