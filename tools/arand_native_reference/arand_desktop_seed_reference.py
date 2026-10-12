"""Original wrapper SetSeed IL: record scenario seeds through vtable services."""

import json

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(seed, scenarios):
    if not 0 <= seed <= 1_000_000_000 or not 0 <= scenarios <= 100:
        raise ValueError("probe bounds")
    ins = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(0x2C60, 10000))).instructions
    positions = {x.offset: i for i, x in enumerate(ins)}
    mem = {1084: 5000, 1088: 5000 + 4 * scenarios}
    for i in range(scenarios):
        mem[5000 + 4 * i] = 6000 + 100 * i
        mem[6000 + 100 * i] = 7000
    mem[7020] = 7500
    stack = []
    local = {}
    calls = []
    pc = 0
    for _ in range(10000):
        x = ins[pc]
        pc += 1
        op, v = x.mnemonic, x.operand
        if op.startswith("ldarg."):
            stack.append(1000 if op == "ldarg.0" else seed)
        elif op.startswith("ldc.i4."):
            stack.append(int(v) if op.endswith(".s") else int(op.rsplit(".", 1)[1]))
        elif op.startswith(("ldloc", "stloc")):
            key = v.index if v is not None else int(op.rsplit(".", 1)[1])
            if op.startswith("ldloc"):
                stack.append(local[key])
            else:
                local[key] = stack.pop()
        elif op.startswith("ldind"):
            stack.append(mem[stack.pop()])
        elif op.startswith("stind"):
            value, address = stack.pop(), stack.pop()
            mem[address] = value
        elif op in ("add", "sub", "mul", "shr"):
            b, a = stack.pop(), stack.pop()
            stack.append(
                {
                    "add": lambda: a + b,
                    "sub": lambda: a - b,
                    "mul": lambda: a * b,
                    "shr": lambda: a >> b,
                }[op]()
            )
        elif op.startswith(("bge", "blt")):
            b, a = stack.pop(), stack.pop()
            if (a >= b) if op.startswith("bge") else (a < b):
                pc = positions[v]
        elif op == "calli":
            target, value, obj = stack.pop(), stack.pop(), stack.pop()
            assert target == 7500
            calls.append([(obj - 6000) // 100, value])
        elif op == "ret":
            return dict(seed=seed, scenarios=scenarios, calls=calls)
        else:
            raise ValueError((op, v))
    raise ValueError("budget")


if __name__ == "__main__":
    for seed, n in ((0, 0), (0, 1), (12345, 2), (42, 5), (1_000_000_000, 3)):
        print(json.dumps(run(seed, n), sort_keys=True))
