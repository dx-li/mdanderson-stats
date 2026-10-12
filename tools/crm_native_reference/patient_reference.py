"""Original CRM patient outcome IL with explicit vector/sort runtime services.

Dates are whole days, so the native 1e-6 tie tolerance is an exact tie here.
The vector sorting service orders times and event kinds; native partitioned
exposure calculation is skipped after the original outcome/censor writes.
"""

import hashlib
import json
import sys

import dnfile
from dncil.cil.body import CilMethodBody
from dncil.cil.body.reader import CilMethodBodyReaderBytes

p = dnfile.dnPE(sys.argv[1])
digest = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
assert digest in {
    "8fc20c5f514d0c0a68877a55126ff97f4a43a88e3ded403adfe982b80bb501e7",
    "a170b501674307aee638b03915c5fb61337f7008a3a65c06b505098903e30768",
}
legacy = digest.startswith("a170b501")
instructions = CilMethodBody(
    CilMethodBodyReaderBytes(p.get_data(0xE214 if legacy else 0xE228, 10000))
).instructions
positions = {i.offset: n for n, i in enumerate(instructions)}


def name(token):
    return str(p.net.mdtables.tables[token.table].rows[token.rid - 1].Name)


def run(arrival, window, updated, toxicity_delay, at):
    assert all(isinstance(x, int) for x in (arrival, window, updated, toxicity_delay, at))
    mem = {1016: arrival, 1024: toxicity_delay, 1032: window, 1040: updated}
    stack = []
    vectors = {}
    next_address = 100000
    pc = 0
    for _ in range(10000):
        i = instructions[pc]
        pc += 1
        op, v = i.mnemonic, i.operand
        if op.startswith("ldarg."):
            stack.append([1000, at, 6][int(op.rsplit(".", 1)[1])])
        elif op.startswith("ldc.i4"):
            stack.append(int(v) if v is not None else int(op.rsplit(".", 1)[1]))
        elif op.startswith(("ldloc", "stloc")):
            key = v.index if v is not None else int(op.rsplit(".", 1)[1])
            address = 10000 + 256 * key
            if op.startswith("ldloca"):
                stack.append(address)
            elif op.startswith("ldloc"):
                stack.append(mem[address])
            else:
                mem[address] = stack.pop()
        elif op.startswith("ldind") or op == "ldobj":
            stack.append(mem[stack.pop()])
        elif op.startswith("stind"):
            value, address = stack.pop(), stack.pop()
            mem[address] = value
        elif op in ("add", "sub"):
            b, a = stack.pop(), stack.pop()
            stack.append(a + b if op == "add" else a - b)
        elif op == "ldsfld":
            stack.append("native IsLess comparator")
        elif op == "call":
            method = name(v)
            if method.startswith("std.pair"):
                value, key, obj = stack.pop(), stack.pop(), stack.pop()
                mem[obj], mem[obj + 8] = mem[key], mem[value]
                stack.append(obj)
            elif method.endswith("push_back"):
                pair, vector = stack.pop(), stack.pop()
                if vector not in vectors:
                    vectors[vector] = [next_address, []]
                    next_address += 256
                vectors[vector][1].append((mem[pair], mem[pair + 8]))
            elif method.endswith((".begin", ".end")):
                output, vector = stack.pop(), stack.pop()
                start, values = vectors[vector]
                for j, (kind, value) in enumerate(values):
                    mem[start + 16 * j], mem[start + 16 * j + 8] = kind, value
                mem[output] = start + (16 * len(values) if method.endswith(".end") else 0)
                stack.append(output)
            elif method.startswith("std.sort"):
                comparator, end, begin = stack.pop(), stack.pop(), stack.pop()
                assert comparator == "native IsLess comparator"
                vector = next(v for v in vectors.values() if v[0] == begin)
                assert end == begin + 16 * len(vector[1])
                vector[1].sort(key=lambda x: (x[1], x[0]))
            elif method.endswith("CalcPartitionedTimeToEvent"):
                assert stack.pop() == 6
                assert stack.pop() == 1000
            elif method.endswith("._Tidy"):
                stack.pop()
            else:
                raise ValueError(method)
        elif op == "pop":
            stack.pop()
        elif op == "switch":
            index = stack.pop()
            if 0 <= index < len(v):
                pc = positions[v[index]]
        elif op == "br" or op.startswith(("br.", "leave")):
            pc = positions[v]
        elif op in ("nop",):
            pass
        elif op == "ret":
            assert not stack
            return {
                "input": [arrival, window, updated, toxicity_delay, at],
                "outcome": mem[1008],
                "followup": mem[1048],
            }
        else:
            raise ValueError((i.offset, op, v))
    raise ValueError("instruction budget")


if __name__ == "__main__":
    for arrival in (0, 10):
        for window, updated, toxicity_delay in (
            (6, 2, 7),
            (6, 6, 7),
            (6, 9, 7),
            (6, 2, 2),
            (6, 6, 6),
        ):
            for offset in (0, 1, 2, 5, 6, 10):
                print(
                    json.dumps(
                        run(arrival, window, updated, toxicity_delay, arrival + offset),
                        sort_keys=True,
                    )
                )
