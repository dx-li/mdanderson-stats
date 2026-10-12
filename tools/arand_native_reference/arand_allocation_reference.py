import json

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(probabilities, weights, count=10, runin=0, power=1, floor=0):
    ins = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(447944, 100000))).instructions
    positions = {v.offset: k for k, v in enumerate(ins)}
    na = len(probabilities[0])
    no = len(weights)
    mem = {
        1008: 2000,
        1012: 2500,
        1016: 2800,
        2804: 28100,
        1152: 3300,
        3300: 3400,
        1160: runin,
        3000: 3100,
        3004: na,
        3100: 4000,
    }
    strings = {5216: "group"}
    nxt = {}
    for i in range(na):
        mem[4016 + 100 * i] = i
        nxt[4000 + 100 * i] = 4000 + 100 * (i + 1) if i + 1 < na else 3100
        strings[4020 + 100 * i] = str(i)
    for i in range(no):
        mem[3440 + 100 * i] = weights[i]
        nxt[3400 + 100 * i] = 3400 + 100 * (i + 1) if i + 1 < no else 3300
        strings[3416 + 100 * i] = str(i)
    stack = []
    calls = []
    pc = 0

    def la(i):
        return 100000 + i * 256

    def popn(n):
        out = stack[-n:] if n else []
        if n:
            del stack[-n:]
        return out

    for _ in range(20000):
        x = ins[pc]
        pc += 1
        op = x.mnemonic
        v = x.operand
        if op.startswith("ldc.i4."):
            stack.append(
                -1
                if op.endswith("m1")
                else int(v)
                if op.endswith("s")
                else int(op.rsplit(".", 1)[1])
            )
        elif op in ("ldc.i4", "ldc.r8"):
            stack.append(v)
        elif op.startswith("ldarg."):
            stack.append({"ldarg.0": 1000, "ldarg.1": 7000, "ldarg.2": 5216, "ldarg.3": 3000}[op])
        elif op.startswith("ldloca"):
            stack.append(la(v.index))
        elif op.startswith("ldloc"):
            stack.append(mem.get(la(v.index if v is not None else int(op.rsplit(".", 1)[1])), 0))
        elif op.startswith("stloc"):
            mem[la(v.index if v is not None else int(op.rsplit(".", 1)[1]))] = stack.pop()
        elif op.startswith("ldind"):
            stack.append(mem.get(stack.pop(), 0))
        elif op.startswith("stind"):
            a, b = popn(2)
            mem[a] = b
        elif op == "ldsflda":
            stack.append(8000)
        elif op == "dup":
            stack.append(stack[-1])
        elif op == "pop":
            stack.pop()
        elif op in ("add", "sub", "mul", "div"):
            a, b = popn(2)
            stack.append(
                {
                    "add": lambda: a + b,
                    "sub": lambda: a - b,
                    "mul": lambda: a * b,
                    "div": lambda: a / b,
                }[op]()
            )
        elif op.startswith("conv."):
            pass
        elif op == "ceq":
            a, b = popn(2)
            stack.append(int(a == b))
        elif op in ("br", "br.s", "leave", "leave.s"):
            pc = positions[v]
        elif op.startswith(("brtrue", "brfalse")):
            if bool(stack.pop()) == op.startswith("brtrue"):
                pc = positions[v]
        elif op.startswith(("bge", "bgt", "ble", "blt", "beq", "bne")):
            a, b = popn(2)
            if {
                "bge": a >= b,
                "bgt": a > b,
                "ble": a <= b,
                "blt": a < b,
                "beq": a == b,
                "bne": a != b,
            }[op[:3]]:
                pc = positions[v]
        elif op == "localloc":
            stack.pop()
            stack.append(50000)
        elif op == "call":
            n = str(p.net.mdtables.tables[v.table].rows[v.rid - 1].Name)
            if n == "__CxxQueryExceptionSize":
                stack.append(1)
            elif n.endswith(".GetQtyOfPatsInTrial"):
                stack.pop()
                stack.append(count)
            elif n.startswith("MDACC_Biostat.Vector") and n.endswith(".{ctor}"):
                a, b, c = popn(3)
                for i in range(b):
                    mem[9000 + i * 8] = c
                stack.append(a)
            elif n.startswith("MDACC_Biostat.Vector") and n.endswith(".()"):
                a, i = popn(2)
                stack.append(9000 + i * 8)
            elif n.endswith(".SetStopTrial"):
                calls.append(["stop", popn(2)[1]])
            elif n.endswith(".GetOutcomeMapPtr"):
                stack.pop()
                stack.append(3200)
            elif n.endswith(".GetPatientGroupMapPtr"):
                stack.pop()
                stack.append(5000)
            elif n.startswith("std._Tree_unchecked_const_iterator") and n.endswith(".{ctor}"):
                a, b, c = popn(3)
                mem[a] = b
                stack.append(a)
            elif n.startswith("std._Tree_unchecked_const_iterator") and n.endswith(".++"):
                a = stack.pop()
                mem[a] = nxt[mem[a]]
                stack.append(a)
            elif n.startswith("std._Tree_const_iterator") and n.endswith(".=="):
                a, b = popn(2)
                stack.append(int(mem[a] == mem[b]))
            elif n.startswith("std.vector") and n.endswith(".{ctor}"):
                a, b = popn(2)
                mem[a] = 9000
                mem[a + 4] = 9000 + 24 * b
                stack.append(a)
            elif n.startswith("std.vector") and n.endswith("._Tidy"):
                stack.pop()
            elif n.startswith("std.basic_string") and n.endswith("._Tidy"):
                popn(3)
            elif n.startswith("std.basic_string") and n.endswith(".assign"):
                a, b, c = popn(3)
                strings[a] = strings[b]
                stack.append(a)
            elif n.startswith("std.basic_string") and n.endswith(".="):
                a, b = popn(2)
                strings[a] = strings[b]
                stack.append(a)
            elif n.startswith("std.basic_string") and n.endswith(".compare"):
                a, b = popn(2)
                stack.append(int(strings[a] != strings[b]))
            elif n.endswith(".GetProbabilityBest"):
                _, a, g, o = popn(4)
                stack.append(probabilities[int(strings[o])][int(strings[a])])
            elif n.startswith("?GetStatus@Arm"):
                a, g = popn(2)
                raise ValueError("Service outside this reference fixture scope")
            elif n.endswith(".SetWhichArmIsWinner"):
                t, g, w, e = popn(4)
                calls.append(["winner", strings[g], strings[w], e])
            else:
                raise ValueError((op, n))
        elif op == "calli":
            target = stack.pop()
            t, vec = popn(2)
            assert target == 28100
            raw = [mem[9000 + i * 8] for i in range(na)]
            import arand_x86_reference

            return {"raw": raw, "native_tuned": arand_x86_reference.run(raw, power, floor)}
        elif op == "ret":
            return calls
        elif op == "nop":
            pass
        else:
            raise ValueError((op, v))
    raise ValueError("budget")


if __name__ == "__main__":
    for cfg in [
        dict(probabilities=[[0.8, 0.2], [0.1, 0.9]], weights=[0.25, 0.75]),
        dict(probabilities=[[0.8, 0.2], [0.1, 0.9]], weights=[1, 1]),
        dict(probabilities=[[0.8, 0.2], [0.1, 0.9]], weights=[0.25, 0.75], count=4, runin=5),
        dict(
            probabilities=[[0.05, 0.15, 0.8], [0.1, 0.1, 0.8]],
            weights=[0.5, 0.5],
            power=2,
            floor=0.2,
        ),
    ]:
        print(json.dumps({"input": cfg, "native": run(**cfg)}))
