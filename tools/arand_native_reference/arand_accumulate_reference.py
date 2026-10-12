"""Execute the full original noncovariate per-replication accumulation body.

Services supply maps, strings, final trial records and the SimArm counter sinks.
No posterior or RNG calculations are substituted into a claimed numeric result.
"""

import json

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(counts, statuses, winner=None, early=False, completed=False, duration=5):
    na = len(counts)
    ins = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(0x59AC0, 100000))).instructions
    positions = {v.offset: k for k, v in enumerate(ins)}
    mem = {
        1036: 2000,
        2712: duration,
        2548: int(completed),
        1104: 3000,
        3000: 4000,
        1112: 6000,
        1116: 6024,
        9940: 0,
    }
    strings = {6000: "group"}
    nxt = {}
    for i in range(na):
        node = 4000 + 100 * i
        mem[node + 40] = 8000 + 100 * i
        mem[node + 32] = len(str(i))
        mem[node + 36] = 15
        strings[node + 16] = str(i)
        nxt[node] = 4000 + 100 * (i + 1) if i + 1 < na else 3000
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
            elif n.startswith("std.vector") and n.endswith(".push_back"):
                obj, value = popn(2)
                calls.append(["duration", mem[value]])
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
            elif n.startswith("std._Tree") and n.endswith(".find"):
                tree, a, g = popn(3)
                assert tree == 1184 and strings[g] == "group"
                mem[a] = 9900
                stack.append(a)
            elif n.startswith("std.basic_string") and n.endswith("._Tidy"):
                popn(3)
            elif n.startswith("std.basic_string") and n.endswith(".{ctor}"):
                a = stack.pop()
                strings[a] = ""
                stack.append(a)
            elif n.startswith("std.basic_string") and n.endswith(".="):
                a, b = popn(2)
                strings[a] = strings[b]
                mem[a + 16] = len(strings[a])
                mem[a + 20] = 15
                stack.append(a)
            elif n.startswith("std.basic_string") and n.endswith(".compare"):
                a, start, length, b, size = popn(5)
                stack.append(int(strings[a] != strings[b]))
            elif n.endswith(".GetWhichArmIsWinner"):
                obj, a, g = popn(3)
                strings[a] = "None" if winner is None else str(winner)
                mem[a + 24] = int(early)
                stack.append(a)
            elif n.endswith(".GetQtyPatients"):
                obj, a, g = popn(3)
                stack.append(counts[int(strings[a])])
            elif n.startswith("?GetStatusForGroup@"):
                obj, a, g = popn(3)
                stack.append(statuses[int(strings[a])])
            elif n.endswith(".AccumulateNumPats") or n.startswith("?IncrementFinalStatus@"):
                obj, g, value = popn(3)
                kind = "enrollment" if n.endswith(".AccumulateNumPats") else "status"
                calls.append([kind, (obj - 8000) // 100, value])
            elif n.endswith((".IncrementQtySelected", ".IncrementQtySelectedEarly")):
                obj, g = popn(2)
                kind = "selected_early" if n.endswith(".IncrementQtySelectedEarly") else "selected"
                calls.append([kind, (obj - 8000) // 100])
            else:
                raise ValueError((x.offset, n))
        elif op == "ret":
            return dict(
                calls=calls,
                duration_sum=mem[1192],
                duration_square_sum=mem[1200],
                completed=mem.get(1208, 0),
                any_status_1_or_2=mem[9940],
            )
        elif op == "nop":
            pass
        else:
            raise ValueError((x.offset, op, v))
    raise ValueError("budget")


if __name__ == "__main__":
    for statuses in ([0, 1], [2, 3], [4, 5], [9, 0]):
        for winner in (None, 0, 1):
            case = dict(
                counts=[2, 4],
                statuses=statuses,
                winner=winner,
                early=winner == 1,
                completed=winner == 0,
                duration=7,
            )
            print(json.dumps(dict(input=case, native=run(**case)), sort_keys=True))
