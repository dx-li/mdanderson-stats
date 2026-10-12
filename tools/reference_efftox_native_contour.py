"""Bounded original EffTox IL execution; requires dnfile and dncil.

Usage: python tools/reference_efftox_native_contour.py ORIGINAL_DLL > fixtures.json
Original DLL is hash-pinned and never redistributed. Runtime containers and
vector operations use host services; original optimizer and LU instructions
execute. Windows process, exception and singular-input parity is not claimed.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import dnfile
from dncil.cil.body.reader import read_method_body_from_bytes

DLL = sys.argv[1]
assert (
    hashlib.sha256(open(DLL, "rb").read()).hexdigest()
    == "7570fa52e5c03c4818ae8c4f17913bd75a2f036ebac844aa6c86e2ce3a183fd0"
)


class VM:
    def __init__(self):
        self.pe = dnfile.dnPE(DLL)
        self.mem = {}
        self.top = 4096
        self.bodies = {}
        self.matrices = {}
        self.calls = {}
        self.linear = {}
        self.stdvec = {}
        self.runs = []

    def alloc(self, n=256):
        p = self.top
        self.top += max(n, 256)
        return p

    def vec(self, values, p=None):
        if p is None:
            p = self.alloc()
        data = self.alloc(8 * len(values))
        self.mem[p + 4] = data
        self.mem[p + 8] = len(values)
        for i, x in enumerate(values):
            self.mem[data + 8 * i] = x
        return p

    def vals(self, p):
        return [self.mem[self.mem[p + 4] + 8 * i] for i in range(self.mem[p + 8])]

    def call(self, idx, args):
        self.calls[idx] = self.calls.get(idx, 0) + 1
        if idx in (0x27, 0x49, 0x9D, 0x103):
            return None
        if idx == 0x100:
            return args[0]
        if idx in (0x56, 0xB2):
            self.vec([0] * args[1], args[0])
            return None
        if idx in (0x9B, 0x118):
            return self.mem[args[0] + 4] + 8 * args[1]
        if idx == 0xFD:
            p, q = args
            n, m = self.mem[q + 8], self.mem[q + 12]
            data = self.alloc(8 * n * m)
            old = self.matrices[q]
            for k in range(n * m):
                self.mem[data + 8 * k] = self.mem[old + 8 * k]
            self.matrices[p] = data
            self.mem[p + 8] = n
            self.mem[p + 12] = m
            return p
        if idx == 0x10B:
            self.stdvec[args[0]] = []
            return args[0]
        if idx == 0x10C:
            return self.vec([0.5, 0.5, -0.5], args[0])
        if idx == 0x10F:
            p, v = args
            vs = self.stdvec[p]
            vs.append(self.vals(v))
            data = self.alloc(40 * len(vs))
            for i, x in enumerate(vs):
                self.vec(x, data + 40 * i)
            self.mem[p] = data
            self.mem[p + 4] = data + 40 * len(vs)
            return data + 40 * (len(vs) - 1)
        if idx == 0x28:
            p, n, m = args
            self.matrices[p] = self.alloc(n * m * 8)
            self.mem[p + 8] = n
            self.mem[p + 12] = m
            return p
        if idx == 0xF9:
            p, q = args
            self.matrices[p] = self.matrices[q]
            self.mem[p + 8] = self.mem[q + 8]
            self.mem[p + 12] = self.mem[q + 12]
            return None
        if idx == 0x105:
            return args[0] ** args[1]
        if idx == 0x257:
            return 0
        if idx in (0x107, 0x128, 0x115):
            return None
        if idx in (0x146, 0x147):
            p, n, _ = args
            size = 40 if idx == 0x146 else 8
            data = self.alloc(size * n)
            self.mem[p] = data
            self.mem[p + 4] = data + size * n
            self.mem[p + 8] = data + size * n
            return None
        if idx == 0x2C:
            return self.vec([0.0] * args[1], args[0])
        if idx == 0x2B:
            return None
        if idx == 0x55:
            return self.vec(self.vals(args[1]), args[0])
        if idx == 0xAA:
            return self.mem[args[0] + 4] + 8 * args[1]
        if idx in (0xFC, 0x97):
            p, i, j = args
            return self.matrices[p] + 8 * (self.mem[p + 12] * i + j)
        if idx in (0x120, 0x121, 0x122):
            out, a, b = args
            v = (
                [self.mem[a] * x for x in self.vals(b)]
                if idx == 0x120
                else [x - y if idx == 0x121 else x + y for x, y in zip(self.vals(a), self.vals(b))]
            )
            return self.vec(v, out)
        if idx == 0x129:
            return self.vec(
                [x + y for x, y in zip(self.vals(args[0]), self.vals(args[1]))], args[0]
            )
        if idx == 0x10D:
            return self.vec([x * self.mem[args[1]] for x in self.vals(args[0])], args[0])
        if idx == 0x133:
            return math.sqrt(sum(x * x for x in self.vals(args[0])))
        if idx not in self.bodies:
            self.bodies[idx] = read_method_body_from_bytes(
                self.pe.get_data(self.pe.net.mdtables.MethodDef.rows[idx - 1].Rva, 100000)
            ).instructions
        instructions = self.bodies[idx]
        pos = {i.offset: k for k, i in enumerate(instructions)}
        s = []
        loc = {}
        pc = 0

        def lp(k):
            if k not in loc:
                loc[k] = self.alloc()
            return loc[k]

        for steps in range(200000):
            ins = instructions[pc]
            pc += 1
            o = ins.opcode.name
            v = ins.operand
            if o.startswith("ldc.i4."):
                s.append(-1 if o.endswith("m1") else v if o.endswith("s") else int(o[-1]))
            elif o in ("ldc.r8", "ldc.i4"):
                s.append(v)
            elif o.startswith("ldarg."):
                s.append(args[v.index if o.endswith("s") else int(o[-1])])
            elif o.startswith("ldloca"):
                s.append(lp(v.index))
            elif o.startswith("ldloc."):
                s.append(self.mem.get(lp(v.index if o.endswith("s") else int(o[-1])), 0))
            elif o.startswith("stloc."):
                self.mem[lp(v.index if o.endswith("s") else int(o[-1]))] = s.pop()
            elif o.startswith("ldind."):
                s.append(self.mem.get(s.pop(), 0))
            elif o.startswith("stind."):
                x, p = s.pop(), s.pop()
                self.mem[p] = x
            elif o == "ldsflda":
                s.append(v.value)
            elif o == "calli":
                del s[-6:]
                s.append(0)
            elif o == "localloc":
                s.append(self.alloc(s.pop()))
            elif o == "cpblk":
                n, src, dst = s.pop(), s.pop(), s.pop()
                assert n == 8
                self.mem[dst] = self.mem[src]
            elif o == "dup":
                s.append(s[-1])
            elif o == "pop":
                s.pop()
            elif o in ("add", "sub", "mul", "div"):
                b, a = s.pop(), s.pop()
                s.append(
                    {
                        "add": lambda: a + b,
                        "sub": lambda: a - b,
                        "mul": lambda: a * b,
                        "div": lambda: a / b,
                    }[o]()
                )
            elif o == "neg":
                s.append(-s.pop())
            elif o == "shl":
                b, a = s.pop(), s.pop()
                s.append(a << b)
            elif o.startswith("conv."):
                pass
            elif o in ("leave.s", "leave", "br", "br.s"):
                pc = pos[v]
            elif o.startswith(("brtrue", "brfalse")):
                if bool(s.pop()) == o.startswith("brtrue"):
                    pc = pos[v]
            elif o.startswith(("bge", "bgt", "ble", "blt", "beq", "bne")):
                b, a = s.pop(), s.pop()
                base = o.split(".")[0]
                if {
                    "bge": a >= b,
                    "bgt": a > b,
                    "ble": a <= b,
                    "blt": a < b,
                    "beq": a == b,
                    "bne": a != b,
                }[base]:
                    pc = pos[v]
            elif o == "call":
                t = v.value >> 24
                i = v.value & 0xFFFFFF
                if t == 10:
                    assert str(self.pe.net.mdtables.MemberRef.rows[i - 1].Name) == "Abs"
                    s.append(abs(s.pop()))
                    continue
                row = self.pe.net.mdtables.MethodDef.rows[i - 1]
                sig = row.Signature.value
                n = sig[1]
                a = s[-n:] if n else []
                if n:
                    del s[-n:]
                r = self.call(i, a)
                ret = 2
                while sig[ret] in (0x20, 0x1F):
                    ret += 2
                if sig[ret] != 1:
                    s.append(r)
            elif o == "ret":
                result = s.pop() if s else None
                if idx == 0x106:
                    self.runs.append(
                        {
                            "start": self.vals(args[2]),
                            "coefficients": self.vals(args[3]),
                            "objective": result,
                            "iterations": self.mem[args[0]],
                            "evaluations": self.mem[args[0] + 4],
                        }
                    )
                return result
            elif o == "nop":
                pass
            else:
                raise RuntimeError((hex(idx), hex(ins.offset), o, v))
        raise RuntimeError("budget")

    def setup(self, targets, start):
        obj = self.alloc()
        data = self.alloc()
        self.matrices[obj] = data
        self.mem[obj + 8] = len(targets)
        self.mem[obj + 12] = 2
        for i, row in enumerate(targets):
            for j, x in enumerate(row):
                self.mem[data + 8 * (2 * i + j)] = x
        return obj

    def solve(self, targets):
        obj = self.setup(targets, None)
        solver = self.alloc()
        outs = [self.alloc() for _ in range(3)]
        f = self.call(0x102, [solver, obj, *outs])
        return {"targets": targets, "coefficients": [self.mem.get(p) for p in outs], "objective": f}

    def run(self, targets, start):
        obj = self.setup(targets, start)
        nm = self.alloc()
        v = self.vec(start)
        out = self.vec([])
        f = self.call(0x106, [nm, obj, v, out, 200, 0.001, 0.05])
        return {
            "start": start,
            "coefficients": self.vals(out),
            "objective": f,
            "iterations": self.mem[nm],
            "evaluations": self.mem[nm + 4],
        }


if __name__ == "__main__":
    cases = json.loads(
        (
            Path(__file__).resolve().parents[1] / "tests/fixtures/efftox-native-contour.json"
        ).read_text()
    )
    results = []
    for case in cases:
        vm = VM()
        result = vm.solve(case["targets"])
        result.update(
            case=case["case"], runs=vm.runs, fallback=result["coefficients"] == [0.5, 0.5, -0.5]
        )
        results.append(result)
    print(json.dumps(results, indent=2))
