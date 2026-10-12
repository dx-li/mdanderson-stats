"""Original IL pipeline; callbacks provide containers/posteriors. No native RNG claim."""

import json

import arand_allocation_reference as allocation
import arand_final_reference as final
import arand_groupfinal_reference as groupfinal
import arand_mincriteria_reference as minimum
import arand_minmax_reference as mm
import arand_scheduler_reference as sc
import arand_status_reference as ss
from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(
    mode=0,
    now=5,
    count=3,
    threshold=3,
    pending=False,
    probabilities=(0.6, 0.4),
    statuses=(0, 0),
    max_count=100,
    followup=0,
    min_count=0,
):
    state = {"statuses": list(statuses), "stopped": False, "winner": None, "time": now}
    trace = []

    class Log(list):
        def append(self, c):
            super().append(c)
            trace.append(c)
            if c[0] == "status" and len(c) == 2:
                state["statuses"][:] = [c[1]] * len(statuses)
            elif c[0] in ("look", "evaluate"):
                evaluate(c[1])
            elif c[0] == "stop":
                state["stopped"] = bool(c[1])
            elif c[0] == "stop_all":
                finish(False)
            elif c[0] == "stop_group":
                finish(True)

    def finish(group):
        events = (groupfinal if group else final).run(
            probabilities,
            state["statuses"],
            cutoff=0.5,
            completed=state.get("completed", False) if not group else False,
        )
        trace.extend(events)
        state["winner"] = events[-1][2]
        if not group:
            state["stopped"] = True

    def evaluate(t):
        trace.append(["posterior_service", t, list(probabilities)])
        ss.run(probabilities, state["statuses"], event_log=Log(), share_statuses=True)

    ins = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(738120, 100000))).instructions
    positions = {v.offset: k for k, v in enumerate(ins)}
    mem = {1384: 2800, 2816: 28160, 1144: 2900, 2924: 29240, 1712: now}
    strings = {8000: "Stop Trial"}
    stack = []
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
        elif op.startswith("ldarg"):
            i = v.index if v is not None else int(op.rsplit(".", 1)[1])
            stack.append([1000, 7000, 5216, 0, 0][i])
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
        elif op in ("add", "sub", "mul", "div", "or"):
            a, b = popn(2)
            stack.append(
                {
                    "add": lambda: a + b,
                    "sub": lambda: a - b,
                    "mul": lambda: a * b,
                    "div": lambda: int(a / b),
                    "or": lambda: a | b,
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
        elif op == "localloc":
            stack.pop()
            stack.append(50000)
        elif op == "call":
            n = str(p.net.mdtables.tables[v.table].rows[v.rid - 1].Name)
            if n == "__CxxQueryExceptionSize":
                stack.append(1)
            elif n.endswith(".CheckMinCriteria"):
                popn(2)
                ok = minimum.run(
                    state["statuses"],
                    now=state["time"],
                    count=count,
                    min_count=min_count,
                    check_arm_min=False,
                    check_group_min=False,
                )
                trace.append(["minimum", ok])
                stack.append(int(ok))
            elif n.endswith(".CheckTime"):
                stack.pop()
                sc.run(
                    mode,
                    now,
                    count,
                    threshold=threshold,
                    pending=pending,
                    event_log=Log(),
                    share_statuses=True,
                )
            elif n.endswith(".ShouldBeStopped"):
                stack.pop()
                stack.append(int(state["stopped"]))
            elif n.startswith("std.basic_string") and n.endswith("._Tidy"):
                a, b, c = popn(3)
                strings.setdefault(a, "")
            elif n.startswith("std.basic_string") and n.endswith(".{ctor}"):
                a, b = popn(2)
                strings[a] = strings[b]
                stack.append(a)
            elif n.startswith("std.basic_string") and n.endswith((".=", "._Assign_rv")):
                a, b = popn(2)
                strings[a] = strings[b]
                if n.endswith(".="):
                    stack.append(a)
            else:
                raise ValueError((op, n))
        elif op == "calli":
            target = stack.pop()
            if target == 28160:
                a, t = popn(2)
                trace.append(["check_minmax", mem[t]])
                state["completed"] = count >= max_count
                result = mm.run(
                    state["statuses"],
                    now=mem[t],
                    count=count,
                    min_count=min_count,
                    max_count=max_count,
                    followup=followup,
                    event_log=Log(),
                    share_statuses=True,
                )
                mem[t] = result["time"]
                state["time"] = result["time"]
            elif target == 29240:
                a, out, g, extra, override = popn(5)
                elig = [i for i, z in enumerate(state["statuses"]) if z == 0]
                raw = (
                    allocation.run([[probabilities[i] for i in elig]], [1])
                    if len(elig) > 1
                    else {"native_tuned": [1]}
                )
                trace.append(["allocation", elig, list(raw["native_tuned"])])
                strings[out] = "allocation-service"
                stack.append(out)
            else:
                raise ValueError(target)
        elif op == "ret":
            return {
                "trace": trace,
                "statuses": state["statuses"],
                "winner": state["winner"],
                "stopped": state["stopped"],
                "time": state["time"],
                "return": strings[stack.pop()],
            }
        elif op == "nop":
            pass
        else:
            raise ValueError((op, v))
    raise ValueError("budget")


if __name__ == "__main__":
    for cfg in [
        dict(mode=4, pending=True),
        dict(mode=4, pending=False),
        dict(mode=0),
        dict(mode=0, max_count=3, followup=2),
        dict(mode=0, probabilities=[0.96, 0.04]),
        dict(mode=0, min_count=5),
    ]:
        print(json.dumps({"input": cfg, "original_pipeline": run(**cfg)}))
