import json

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(probabilities, statuses, cutoff=0.5, upper=0.95, completed=True, arm_rules=None):
    ins = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(460944, 100000))).instructions
    positions = {v.offset: k for k, v in enumerate(ins)}
    if isinstance(probabilities[0], (int, float)):
        probabilities = [probabilities]
    na = len(statuses)
    no = len(probabilities)
    mem = {
        1012: 2000,
        1128: 3000,
        3000: 3100,
        3100: 4000,
        1008: 2500,
        1032: cutoff,
        1072: upper,
        1115: int(arm_rules is not None),
        1164: int(completed),
        3200: 3300,
        3300: 3400,
        3204: no,
        5000: 5100,
        5100: 5200,
    }
    strings = {8000: "None", 5216: "group"}
    nxt = {5200: 5100}
    for i in range(na):
        mem[4000 + 100 * i + 40] = 6000 + i * 100
        nxt[4000 + 100 * i] = 4000 + 100 * (i + 1) if i + 1 < na else 3100
        strings[4016 + 100 * i] = str(i)
    for i in range(no):
        nxt[3400 + 100 * i] = 3400 + 100 * (i + 1) if i + 1 < no else 3300
        strings[3416 + 100 * i] = str(i)
    if arm_rules is not None:
        for i in range(na):
            mem[20040 + 100 * i] = 30000
        for offset in (4, 8, 12, 16, 20, 24):
            mem[30000 + offset] = 30000 + offset
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
            stack.append(1000)
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
        elif op == "cpblk":
            a, b, size = popn(3)
            assert size == 4
            mem[a] = mem[b]
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
                    "div": lambda: int(a / b),
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
            elif n.endswith(".SetStopTrial"):
                calls.append(["stop", popn(2)[1]])
            elif n.startswith("std._Tree") and n.endswith(".find"):
                tree, a, g = popn(3)
                assert tree == 1168
                mem[a] = 20000 + 100 * int(strings[g])
                stack.append(a)
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
            elif n.endswith(".GetPerGroupStoppingRule"):
                stack.pop()
                stack.append(0)
            elif n.endswith(".GetProbabilityBest"):
                _, a, g, o = popn(4)
                stack.append(probabilities[int(strings[o])][int(strings[a])])
            elif n.startswith("?GetStatus@Arm"):
                a, g = popn(2)
                stack.append(statuses[(a - 6000) // 100])
            elif n.endswith(".SetWhichArmIsWinner"):
                t, g, w, e = popn(4)
                calls.append(["winner", strings[g], strings[w], e])
            else:
                raise ValueError((op, n))
        elif op == "calli":
            target = stack.pop()
            if target in (30004, 30008, 30012, 30016, 30020, 30024):
                arm = (stack.pop() - 20040) // 100
                key = {
                    30004: "upper",
                    30008: "lower",
                    30012: "cutoff",
                    30016: "threshold",
                    30020: "futility_cutoff",
                    30024: "enforce",
                }[target]
                stack.append(arm_rules[arm][key])
            else:
                raise ValueError(target)
        elif op == "ret":
            return calls
        elif op == "nop":
            pass
        else:
            raise ValueError((op, v))
    raise ValueError("budget")


if __name__ == "__main__":
    for cfg in [
        dict(probabilities=[0.7, 0.7], statuses=[0, 0]),
        dict(probabilities=[0.9, 0.7], statuses=[1, 0]),
        dict(probabilities=[0.9, 0.7], statuses=[4, 0]),
        dict(probabilities=[0.9, 0.7], statuses=[2, 0]),
        dict(probabilities=[0.5, 0.4], statuses=[0, 0]),
        dict(probabilities=[0.4, 0.3], statuses=[0, 0], upper=0.35),
        dict(probabilities=[[0.7, 0.3], [0.3, 0.7]], statuses=[0, 0]),
        dict(probabilities=[0.96, 0.04], statuses=[0, 0], completed=False),
    ]:
        print(json.dumps({"input": cfg, "native": run(**cfg)}))
