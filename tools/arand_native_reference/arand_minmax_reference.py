import json

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(
    statuses,
    now=5,
    count=5,
    min_count=0,
    max_count=100,
    min_time=0,
    max_time=100,
    followup=0,
    stop_one=True,
    arm_count=None,
    arm_min=None,
    arm_max=None,
    group_count=None,
    group_min=None,
    group_max=None,
    start=None,
    duration=None,
    *,
    event_log=None,
    share_statuses=False,
):
    statuses = statuses if share_statuses else list(statuses)
    na = len(statuses)
    ins = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(456644, 100000))).instructions
    positions = {v.offset: k for k, v in enumerate(ins)}
    arm_count = arm_count or [0] * na
    arm_min = arm_min or [0] * na
    arm_max = arm_max or [100] * na
    group_count = group_count or arm_count
    group_min = group_min or arm_min
    group_max = group_max or arm_max
    start = start or [0] * na
    duration = duration or [0] * na
    mem = {
        1000: 2800,
        2812: 28120,
        2824: 28240,
        2828: 28280,
        1012: 2000,
        1128: 3000,
        3000: 3100,
        3100: 4000,
        1008: 2500,
        1024: followup,
        5000: 5100,
        5004: 1,
        5100: 5200,
        7000: now,
        1112: int(stop_one),
    }
    for k in range(104, 112):
        mem[1000 + k] = 1
    strings = {5216: "group"}
    nxt = {5200: 5100}
    for i in range(na):
        mem[4000 + 100 * i + 40] = 6000 + i * 100
        nxt[4000 + 100 * i] = 4000 + 100 * (i + 1) if i + 1 < na else 3100
        strings[4016 + 100 * i] = str(i)
    stack = []
    calls = event_log if event_log is not None else []
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
            stack.append(1000 if op == "ldarg.0" else 7000)
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
                    "div": lambda: int(a / b),
                }[op]()
            )
        elif op.startswith("conv."):
            pass
        elif op == "ceq":
            a, b = popn(2)
            stack.append(int(a == b))
        elif op == "cgt":
            a, b = popn(2)
            stack.append(int(a > b))
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
                raise ValueError("Service outside this reference fixture scope")
            elif n.startswith("ClinicalTrials.Trial.Get"):
                stack.pop()
                stack.append(
                    {
                        "ClinicalTrials.Trial.GetMinLengthOfTrial": min_time,
                        "ClinicalTrials.Trial.GetMaxLengthOfTrial": max_time,
                        "ClinicalTrials.Trial.GetMinQtyPatsInTrial": min_count,
                        "ClinicalTrials.Trial.GetMaxQtyPatsInTrial": max_count,
                        "ClinicalTrials.Trial.GetQtyOfPatsInTrial": count,
                    }[n]
                )
            elif n in (
                "ClinicalTrials.Arm.GetQtyOfPatients",
                "ClinicalTrials.Arm.GetMinQtyPatients",
                "ClinicalTrials.Arm.GetMaxQtyPatients",
            ):
                row = p.net.mdtables.tables[v.table].rows[v.rid - 1]
                num = row.Signature.value[1]
                args = popn(num)
                i = (args[0] - 6000) // 100
                arr = {
                    "ClinicalTrials.Arm.GetQtyOfPatients": arm_count if num == 1 else group_count,
                    "ClinicalTrials.Arm.GetMinQtyPatients": arm_min if num == 1 else group_min,
                    "ClinicalTrials.Arm.GetMaxQtyPatients": arm_max if num == 1 else group_max,
                }[n]
                stack.append(arr[i])
            elif n.endswith(".GetStartTime"):
                stack.append(start[(stack.pop() - 6000) // 100])
            elif n.endswith(".GetMaxLengthInTrial"):
                stack.append(duration[(stack.pop() - 6000) // 100])
            elif n.startswith("?SetStatusForAllGroups@Arm"):
                a, z = popn(2)
                i = (a - 6000) // 100
                statuses[i] = z
                calls.append(["armstatus", i, z])
            elif n.startswith("?SetStatus@Arm"):
                a, g, z = popn(3)
                i = (a - 6000) // 100
                statuses[i] = z
                calls.append(["groupstatus", i, z])
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
            if target == 28120:
                a, t = popn(2)
                calls.append(["evaluate", t])
            elif target == 28240:
                a, g = popn(2)
                calls.append(["stop_group", strings[g]])
            elif target == 28280:
                stack.pop()
                calls.append(["stop_all"])
            else:
                raise ValueError(target)
        elif op == "ret":
            return {"calls": calls, "statuses": statuses, "time": mem[7000], "completed": mem[1164]}
        elif op == "nop":
            pass
        else:
            raise ValueError((op, v))
    raise ValueError("budget")


if __name__ == "__main__":
    for cfg in [
        dict(statuses=[0, 0], now=5, max_time=5),
        dict(statuses=[0, 0], now=5.1, max_time=5, followup=2),
        dict(statuses=[0, 0], now=5, max_count=5, followup=2),
        dict(statuses=[0, 0], now=5, max_count=5, min_time=10),
        dict(statuses=[0, 2]),
        dict(statuses=[1, 2], stop_one=False),
        dict(statuses=[3, 0], arm_count=[2, 2], arm_min=[3, 0]),
        dict(statuses=[0, 0], start=[10, 0], duration=[5, 0]),
        dict(statuses=[5, 0], now=10, start=[10, 0], duration=[5, 0]),
        dict(statuses=[0, 0], now=15.1, start=[10, 0], duration=[5, 0]),
        dict(statuses=[4, 0], group_count=[0, 0], group_min=[2, 0]),
    ]:
        print(json.dumps({"input": cfg, "native": run(**cfg)}))
