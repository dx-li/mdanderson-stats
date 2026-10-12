import json
import math

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(
    mode,
    now,
    count,
    threshold=1,
    step=1,
    targets=(1, 2, 3),
    pending=False,
    observed=0,
    index=0,
    *,
    event_log=None,
    share_statuses=False,
):
    body = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(738820, 100000)))
    ins = body.instructions
    positions = {v.offset: k for k, v in enumerate(ins)}
    mem = {
        1000 + 720: mode,
        1000 + 712: now,
        1000 + 728: threshold,
        1000 + 764: threshold,
        1000 + 760: step,
        1000 + 812: len(targets),
        1000 + 800: index,
        1000 + 768: threshold,
        1000 + 876: index,
        1000 + 864: 6000,
        1000 + 868: 6000 + 4 * len(targets),
        1000 + 384: 9000,
        9012: 7000,
    }
    for k, v in enumerate(targets):
        mem[6000 + 4 * k] = v
    stack = []
    local = {}
    calls = event_log if event_log is not None else []
    pc = 0

    def popn(n):
        out = stack[-n:] if n else []
        if n:
            del stack[-n:]
        return out

    for _ in range(10000):
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
            stack.append(100000 + v.index * 256)
        elif op.startswith("ldloc"):
            i = v.index if v is not None else int(op.rsplit(".", 1)[1])
            stack.append(local.get(i, 0))
        elif op.startswith("stloc"):
            i = v.index if v is not None else int(op.rsplit(".", 1)[1])
            local[i] = stack.pop()
        elif op.startswith("ldind"):
            stack.append(mem.get(stack.pop(), 0))
        elif op.startswith("stind"):
            a, b = popn(2)
            mem[a] = b
        elif op == "dup":
            stack.append(stack[-1])
        elif op == "pop":
            stack.pop()
        elif op in ("add", "sub", "mul", "div", "shr"):
            a, b = popn(2)
            stack.append(
                {
                    "add": lambda: a + b,
                    "sub": lambda: a - b,
                    "mul": lambda: a * b,
                    "div": lambda: a / b,
                    "shr": lambda: int(a) >> int(b),
                }[op]()
            )
        elif op.startswith("conv."):
            if op == "conv.r8":
                stack[-1] = float(stack[-1])
            elif op in ("conv.i4", "conv.u1"):
                stack[-1] = int(stack[-1])
        elif op == "ceq":
            a, b = popn(2)
            stack.append(int(a == b))
        elif op in ("br", "br.s", "leave", "leave.s"):
            pc = positions[v]
        elif op.startswith(("brtrue", "brfalse")):
            a = stack.pop()
            if bool(a) == op.startswith("brtrue"):
                pc = positions[v]
        elif op.startswith(("bge", "bgt", "ble", "blt", "beq", "bne")):
            a, b = popn(2)
            c = {
                "bge": a >= b,
                "bgt": a > b,
                "ble": a <= b,
                "blt": a < b,
                "beq": a == b,
                "bne": a != b,
            }[op[:3]]
            if c:
                pc = positions[v]
        elif op == "switch":
            a = stack.pop()
            if 0 <= a < len(v):
                pc = positions[v[a]]
        elif op == "localloc":
            stack.pop()
            stack.append(50000)
        elif op == "call":
            row = p.net.mdtables.tables[v.table].rows[v.rid - 1]
            n = str(row.Name)
            if n == "__CxxQueryExceptionSize":
                stack.append(1)
            elif n.endswith("GetQtyOfPatsInTrial"):
                stack.pop()
                stack.append(count)
            elif n.endswith("PendingPatientOutcome"):
                popn(2)
                stack.append(int(pending))
            elif n.endswith("GetNumberOfObservedOutcomes"):
                popn(2)
                stack.append(observed)
            elif n.startswith("?SetStatus@Trial"):
                calls.append(["status", popn(2)[1]])
            elif n == "Ceiling":
                stack.append(math.ceil(stack.pop()))
            elif n.startswith("MDACC_Biostat.Vector") and n.endswith(".()"):
                ptr, i = popn(2)
                mem[8000 + 8 * i] = targets[i]
                stack.append(8000 + 8 * i)
            else:
                raise ValueError((op, n))
        elif op == "calli":
            target = stack.pop()
            a, b = popn(2)
            assert target == 7000
            calls.append(["look", b])
        elif op == "ret":
            return {
                "calls": calls,
                "next_time": mem[1728],
                "next_enrollment": mem[1764],
                "next_outcome": mem[1768],
                "index": mem[1800],
            }
        else:
            raise ValueError((op, v))
    raise ValueError("budget")


if __name__ == "__main__":
    for config in [
        dict(mode=0, now=2.5, count=5),
        dict(mode=1, now=2.5, count=5),
        dict(mode=1, now=1, count=5),
        dict(mode=2, now=2.5, count=5),
        dict(mode=3, now=0, count=4, threshold=3, step=2),
        dict(mode=4, now=5, count=3, threshold=3, pending=True),
        dict(mode=4, now=5, count=3, threshold=3, pending=False),
        dict(mode=5, now=5, count=5, observed=2, threshold=1),
    ]:
        print(json.dumps({"input": config, "native": run(**config)}))
