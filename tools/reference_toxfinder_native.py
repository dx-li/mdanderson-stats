"""Hash-pinned, bounded original ToxFinder x86 reference probes.

Requires pefile, unicorn and numpy. No original binaries are redistributed.
Runtime substitutions: allocation/free, pow; information probe additionally uses
NumPy LAPACK eigensystem/Cholesky and skips unused postfit covariance. Controller
probes supply declared probability/score callbacks. This is not Windows/RNG parity.
"""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import pefile
from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_32, Uc
from unicorn.x86_const import (
    UC_X86_REG_EAX,
    UC_X86_REG_ECX,
    UC_X86_REG_EIP,
    UC_X86_REG_ESI,
    UC_X86_REG_ESP,
    UC_X86_REG_FPCW,
)

p = None


class Native:
    def __init__(self):
        self.u = Uc(UC_ARCH_X86, UC_MODE_32)
        self.u.mem_map(0x400000, (p.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
        self.u.mem_write(0x400000, p.get_memory_mapped_image())
        self.u.mem_map(0x100000, 0x300000)
        self.u.mem_map(0, 4096)
        self.heap = 0x200000
        self.handlers = {}
        self.stop = 0x104200
        self.sp = 0x130000
        self.u.hook_add(UC_HOOK_CODE, self.hook)

    def alloc(self, n):
        addr = self.heap
        self.heap += (n + 15) & ~15
        return addr

    def put(self, a, fmt, *v):
        self.u.mem_write(a, struct.pack("<" + fmt, *v))

    def get(self, a, fmt):
        return struct.unpack("<" + fmt, self.u.mem_read(a, struct.calcsize("<" + fmt)))

    def vec(self, x):
        a = self.alloc(8 * len(x))
        self.put(a, "d" * len(x), *x)
        return a

    def matrix(self, n, m=None):
        if m is None:
            m = n
        a = self.alloc(n * 4)
        for i in range(n):
            self.put(a + 4 * i, "I", self.alloc(m * 8))
        return a

    def write_matrix(self, a, x):
        for i, row in enumerate(x):
            self.put(self.get(a + 4 * i, "I")[0], "d" * len(row), *row)

    def read_matrix(self, a, n, m=None):
        if m is None:
            m = n
        return [list(self.get(self.get(a + 4 * i, "I")[0], "d" * m)) for i in range(n)]

    def ret(self, n=0, val=None):
        st = self.u.reg_read(UC_X86_REG_ESP)
        a = self.get(st, "I")[0]
        self.u.reg_write(UC_X86_REG_ESP, st + 4 + n)
        self.u.reg_write(UC_X86_REG_EIP, a)
        if val is not None:
            self.u.reg_write(UC_X86_REG_EAX, val)

    def hook(self, u, a, s, _):
        st = u.reg_read(UC_X86_REG_ESP)
        if a == self.stop:
            u.emu_stop()
        elif a in self.handlers:
            self.handlers[a](st)
        elif a == 0x404F20:
            lo, n, lom, m = self.get(st + 4, "4I")
            self.ret(val=self.matrix(n - lo, m - lom))
        elif a == 0x404E40:
            lo, n = self.get(st + 4, "2I")
            self.ret(val=self.alloc(8 * (n - lo)))
        elif a in (0x404FC0, 0x404EA0):
            self.ret()

    def call(self, a, args=b"", this=0):
        self.u.mem_write(self.sp, struct.pack("<I", self.stop) + args)
        self.u.reg_write(UC_X86_REG_ESP, self.sp)
        self.u.reg_write(UC_X86_REG_ECX, this)
        self.u.emu_start(a, self.stop, count=20000000)
        assert self.u.reg_read(UC_X86_REG_EIP) == self.stop, hex(self.u.reg_read(UC_X86_REG_EIP))


def simplex(start, step, objective, maxiter=250):
    r = Native()
    n = len(start)
    opt = r.alloc(48)
    obj = r.alloc(8)
    vt = r.alloc(8)
    stub = 0x104400
    tmp = 0x105000
    r.put(opt + 8, "2I", 1, maxiter)
    r.put(opt + 24, "dd", -1e10, 1e-6)
    r.put(obj, "I", vt)
    r.put(vt, "I", stub)
    r.u.mem_write(stub, b"\xdd\x05" + struct.pack("<I", tmp) + b"\xc2\x04\x00")
    trace = []

    def f(st):
        par = r.get(st + 4, "I")[0]
        x = r.get(par, "d" * n)
        v = objective(x)
        trace.append([list(x), v])
        r.put(tmp, "d", v)

    r.handlers[stub] = f
    out = r.vec([0.0] * n)
    r.call(0x407660, struct.pack("<5I", obj, n, r.vec(start), r.vec(step), out), opt)
    return {
        "point": list(r.get(out, "d" * n)),
        "converged": bool(r.u.reg_read(UC_X86_REG_EAX) & 255),
        "iterations": r.get(opt + 20, "I")[0],
        "trace": trace,
    }


def native_map(shapes, rates, doses, outcomes):
    r = Native()
    prior = r.alloc(16)
    r.put(prior + 4, "3I", 6, r.vec(shapes), r.vec(rates))
    data = r.alloc(48)
    outcome = r.alloc(len(outcomes))
    r.u.mem_write(outcome, bytes(outcomes))
    r.put(
        data + 12,
        "4I",
        len(doses),
        outcome,
        r.vec([x for x, y in doses]),
        r.vec([y for x, y in doses]),
    )
    model = r.alloc(32)
    tmp = 0x105000
    stub = 0x104300
    r.u.mem_write(
        stub,
        b"\xdd\x1d"
        + struct.pack("<I", tmp)
        + b"\xdd\x1d"
        + struct.pack("<I", tmp + 8)
        + b"\x90\xdd\x05"
        + struct.pack("<I", tmp + 16)
        + b"\xc3",
    )
    r.handlers[0x421070] = lambda st: r.u.reg_write(UC_X86_REG_EIP, stub)

    def power(st):
        ex, ba = r.get(tmp, "dd")
        r.put(tmp + 16, "d", ba**ex)

    r.handlers[stub + 12] = power
    r.handlers[0x42091D] = lambda st: r.ret(val=r.alloc(r.get(st + 4, "I")[0]))
    r.handlers[0x42085F] = lambda st: r.ret()
    r.handlers[0x404010] = lambda st: r.ret(20)
    traces = []

    def fit_done(st):
        ptr = r.get(model + 4, "I")[0]
        traces.append(list(r.get(ptr, "6d")))

    r.handlers[0x403A64] = fit_done
    r.handlers[0x403A78] = fit_done
    r.call(0x403920, struct.pack("<2I", prior, data), model)
    return {
        "shapes": shapes,
        "rates": rates,
        "doses": doses,
        "outcomes": outcomes,
        "passes": traces,
        "point": list(r.get(r.get(model + 4, "I")[0], "6d")),
    }


def native_information(shapes, rates, doses, outcomes, samples, candidate):
    r = Native()
    prior = r.alloc(16)
    r.put(prior + 4, "3I", 6, r.vec(shapes), r.vec(rates))
    data = r.alloc(64)
    outcome = r.alloc(len(outcomes))
    r.u.mem_write(outcome, bytes(outcomes))
    r.put(data + 8, "I", len(doses) + 1)
    r.put(
        data + 12,
        "4I",
        len(doses),
        outcome,
        r.vec([x for x, y in doses]),
        r.vec([y for x, y in doses]),
    )
    model = r.alloc(32)
    tmp = 0x105000
    stub = 0x104300
    r.u.mem_write(
        stub,
        b"\xdd\x1d"
        + struct.pack("<I", tmp)
        + b"\xdd\x1d"
        + struct.pack("<I", tmp + 8)
        + b"\x90\xdd\x05"
        + struct.pack("<I", tmp + 16)
        + b"\xc3",
    )
    r.handlers[0x421070] = lambda st: r.u.reg_write(UC_X86_REG_EIP, stub)

    def power(st):
        ex, ba = r.get(tmp, "dd")
        r.put(tmp + 16, "d", ba**ex)

    r.handlers[stub + 12] = power
    r.handlers[0x42091D] = lambda st: r.ret(val=r.alloc(r.get(st + 4, "I")[0]))
    r.handlers[0x42085F] = lambda st: r.ret()
    r.handlers[0x404010] = lambda st: r.ret(20)
    import numpy as np

    this = r.alloc(48)
    cfg = r.alloc(256)
    r.put(this + 20, "I", cfg)
    r.put(this + 32, "I", prior)
    mat = r.matrix(len(samples), 6)
    r.write_matrix(mat, samples)
    r.put(model, "I", 0x4561F4)
    r.put(model + 16, "2I", len(samples), mat)
    raw = []
    eigen = []
    fitted = []

    def eig(st):
        n, original, va, ve = r.get(st + 4, "4I")
        h = np.array(r.read_matrix(original, n))
        vals, vecs = np.linalg.eigh(h)
        raw.append(h.tolist())
        eigen.append(vals.tolist())
        r.put(va, "d" * n, *vals)
        r.write_matrix(ve, vecs)
        r.ret()

    def chol(st):
        n, original, out = r.get(st + 4, "3I")
        r.write_matrix(out, np.linalg.cholesky(r.read_matrix(original, n)))
        r.ret()

    def map_done(st):
        obj = r.u.reg_read(UC_X86_REG_ESI)
        fitted.append(list(r.get(r.get(obj + 4, "I")[0], "6d")))

    r.handlers[0x404460] = eig
    r.handlers[0x4042F0] = chol
    r.handlers[0x403A78] = map_done
    retcode = 0x104500
    r.u.mem_write(
        retcode,
        b"\xdd\x1d"
        + struct.pack("<I", tmp + 24)
        + b"\xe9"
        + struct.pack("<i", r.stop - (retcode + 11)),
    )
    r.u.mem_write(r.sp, struct.pack("<IIIddI", retcode, data, model, *candidate, 0))
    r.u.reg_write(UC_X86_REG_ESP, r.sp)
    r.u.reg_write(UC_X86_REG_ECX, this)
    r.u.emu_start(0x402040, r.stop, count=30000000)
    assert r.u.reg_read(UC_X86_REG_EIP) == r.stop
    return {
        "shapes": shapes,
        "rates": rates,
        "doses": doses,
        "outcomes": outcomes,
        "samples": samples,
        "candidate": candidate,
        "fitted_modes": fitted,
        "raw_hessians": raw,
        "raw_eigenvalues": eigen,
        "native_information": r.get(tmp + 24, "d")[0],
    }


def allocation(mode, score, goal=0.4, grid=10, ckp=1.0, allocate=True, levels=None):
    r = Native()
    this = r.alloc(48)
    cfg = r.alloc(256)
    model = r.alloc(32)
    vt = r.alloc(8)
    data = r.alloc(48)
    xout = r.vec([0, 0, 0])
    yout = r.vec([0, 0, 0])
    tmp = 0x105000
    probcode = 0x104400
    scorecode = 0x104500
    r.put(this + 20, "2I", cfg, grid)
    r.put(cfg + 24, "d", goal)
    r.put(cfg + 96, "d", ckp)
    r.put(cfg + 236, "I", mode)
    r.put(model, "I", vt)
    r.put(vt + 4, "I", probcode)
    if levels is not None:
        levmat = r.matrix(len(levels), 2)
        r.write_matrix(levmat, levels)
        r.put(cfg + 80, "2I", len(levels), levmat)
    r.u.mem_write(probcode, b"\xdd\x05" + struct.pack("<I", tmp) + b"\xc2\x10\x00")
    r.u.mem_write(scorecode, b"\xdd\x05" + struct.pack("<I", tmp + 8) + b"\xc2\x1c\x00")
    trace = []

    def prob(st):
        x, y = r.get(st + 4, "dd")
        r.put(tmp, "d", 0.1 + 0.4 * x + 0.2 * y)

    def objective(st):
        x, y = r.get(st + 12, "dd")
        z = score(x, y)
        trace.append([x, y, z])
        r.put(tmp + 8, "d", z)
        r.u.reg_write(UC_X86_REG_EIP, scorecode)

    r.handlers[probcode] = prob
    r.handlers[0x402040] = objective
    r.call(0x401D70, struct.pack("<6I", model, data, xout, yout, int(allocate), 0), this)
    return {
        "mode": mode,
        "goal": goal,
        "grid": grid,
        "ckp": ckp,
        "allocate": allocate,
        "levels": levels,
        "x": r.get(xout, "3d"),
        "y": r.get(yout, "3d"),
        "scores": trace,
    }


def stage1(levels, history, outcomes, goal, predict):
    r = Native()
    this = r.alloc(48)
    cfg = r.alloc(256)
    model = r.alloc(32)
    vt = r.alloc(8)
    data = r.alloc(48)
    xout = r.vec([0, 0, 0])
    yout = r.vec([0, 0, 0])
    tmp = 0x105000
    probcode = 0x104400
    levmat = r.matrix(len(levels), 2)
    r.write_matrix(levmat, levels)
    r.put(this + 20, "2I", cfg, 30)
    r.put(this + 4, "2I", r.vec([0.0] * (2 * len(levels))), r.vec([0.0] * (2 * len(levels))))
    r.put(cfg + 24, "d", goal)
    r.put(cfg + 80, "2I", len(levels), levmat)
    r.put(model, "I", vt)
    r.put(vt + 4, "I", probcode)
    outcomesptr = r.alloc(len(outcomes))
    r.u.mem_write(outcomesptr, bytes(outcomes))
    r.put(
        data + 12,
        "4I",
        len(history),
        outcomesptr,
        r.vec([x for x, y in history]),
        r.vec([y for x, y in history]),
    )
    r.u.mem_write(probcode, b"\xdd\x05" + struct.pack("<I", tmp) + b"\xc2\x10\x00")
    trace = []

    def prob(st):
        x, y = r.get(st + 4, "dd")
        v = predict(x, y)
        trace.append([x, y, v])
        r.put(tmp, "d", v)

    r.handlers[probcode] = prob
    r.call(0x401400, struct.pack("<4I", model, data, xout, yout), this)
    n = r.get(this + 12, "I")[0]
    return {
        "levels": levels,
        "history": history,
        "outcomes": outcomes,
        "target": goal,
        "dose": [r.get(xout, "d")[0], r.get(yout, "d")[0]],
        "candidates": list(
            zip(r.get(r.get(this + 4, "I")[0], "d" * n), r.get(r.get(this + 8, "I")[0], "d" * n))
        ),
        "first_toxic_index": r.get(this + 16, "I")[0],
        "trace": trace,
    }


def surface(grid, x, y):
    r = Native()
    r.u.reg_write(UC_X86_REG_FPCW, 0x37F)
    cfg = r.alloc(256)
    mat = r.matrix(len(grid), len(grid[0]))
    r.write_matrix(mat, grid)
    r.put(cfg + 224, "3I", len(grid[0]), len(grid), mat)
    tmp = 0x105000
    retcode = 0x104500
    r.u.mem_write(
        retcode,
        b"\xdd\x1d" + struct.pack("<I", tmp) + b"\xe9" + struct.pack("<i", r.stop - (retcode + 11)),
    )
    r.u.mem_write(r.sp, struct.pack("<Idd", retcode, x, y))
    r.u.reg_write(UC_X86_REG_ESP, r.sp)
    r.u.reg_write(UC_X86_REG_ECX, cfg)
    r.u.emu_start(0x4057D0, r.stop, count=10000)
    assert r.u.reg_read(UC_X86_REG_EIP) == r.stop
    return r.get(tmp, "d")[0]


def elicitor_control(mode="success"):
    r = Native()
    prior = r.alloc(40)
    config = r.alloc(256)
    shape = r.vec([0.0] * 6)
    rate = r.vec([0.0] * 6)
    r.put(prior + 4, "3I", 6, shape, rate)
    r.put(config + 0x18, "d", 0.3)
    r.put(config + 0xA0, "2d", 0.5, 350 / 600)
    r.put(config + 0xB0, "d", 0.05)
    r.put(config + 0xB8, "2d", 1400 / 1200, 700 / 600)
    r.put(config + 0xC8, "d", 0.6)
    r.put(config + 0xD0, "2d", 2000 / 1200, 800 / 600)
    r.handlers[0x42091D] = lambda st: r.ret(val=r.alloc(r.get(st + 4, "I")[0]))
    r.handlers[0x42085F] = lambda st: r.ret()
    r.handlers[0x4208BA] = lambda st: r.ret(val=0)
    records = []
    optimizers = []
    attempts = []
    errors = []

    def err(st):
        addr = r.get(st + 4, "I")[0]
        msg = bytes(r.u.mem_read(addr, 128)).split(b"\0")[0].decode()
        errors.append(msg)
        r.u.emu_stop()

    r.handlers[0x404DA0] = err

    def probs(st):
        la, lb, o1, o2 = r.get(st + 4, "2d2I")
        obj = r.u.reg_read(UC_X86_REG_ECX)
        if mode == "success":
            p1, p2 = 0.99 + 0.002 * (la - math.log(0.12)), 0.99 + 0.002 * (lb - math.log(2))
        elif mode == "precheck_failure":
            p1, p2 = 0.0, 1.0
        elif mode == "constant_plateau_success":
            p1, p2 = 0.5, 0.5
        elif mode == "optimizer_failure":
            p1, p2 = (
                0.3 + 0.3 * ((len(records) * 7) % 11) / 11,
                0.4 + 0.2 * ((len(records) * 3) % 7) / 7,
            )
        elif mode == "interior_success":
            p1, p2 = (
                0.99 - 0.08 * math.tanh(la - math.log(0.12)) ** 2,
                0.99 - 0.08 * math.tanh(lb - math.log(2)) ** 2,
            )
        r.put(o1, "d", p1)
        r.put(o2, "d", p2)
        records.append(
            {
                "log_variances": [la, lb],
                "probabilities": [p1, p2],
                "objective_parameters": list(r.get(obj + 8, "6d")),
            }
        )
        r.ret(24)

    r.handlers[0x4052E0] = probs

    def optimizer(st):
        obj, n, start, step, out = r.get(st + 4, "5I")
        opt = r.u.reg_read(UC_X86_REG_ECX)
        optimizers.append(
            {
                "start": list(r.get(start, "2d")),
                "step": list(r.get(step, "2d")),
                "attempts": r.get(opt + 8, "I")[0],
                "iteration_budget": r.get(opt + 12, "I")[0],
                "cutoff": r.get(opt + 24, "d")[0],
                "tolerance": r.get(opt + 32, "d")[0],
                "refresh": r.get(opt + 40, "B")[0],
            }
        )

    r.handlers[0x4075A0] = optimizer

    def attempt(st):
        obj, n, start, step, out = r.get(st + 4, "5I")
        attempts.append({"start": list(r.get(start, "2d")), "step": list(r.get(step, "2d"))})

    r.handlers[0x407660] = attempt
    try:
        r.call(0x405DE0, struct.pack("<3Id", config, 0, 3, 0.3 / 0.7), prior)
    except AssertionError:
        if not errors:
            raise
    return {
        "mode": mode,
        "shapes": list(r.get(shape, "6d")),
        "rates": list(r.get(rate, "6d")),
        "calls": records,
        "optimizers": optimizers,
        "attempts": attempts,
        "errors": errors,
    }


def elicitation_probability(tape, zlow, zhigh, qlow, target):
    r = Native()
    obj = r.alloc(80)
    r.call(0x4051F0, struct.pack("<6d", 0.4, 8, zlow, zhigh, qlow, target), obj)
    tmp = 0x105000
    stub = 0x104300
    r.u.mem_write(
        stub,
        b"\xdd\x1d"
        + struct.pack("<I", tmp)
        + b"\xdd\x1d"
        + struct.pack("<I", tmp + 8)
        + b"\x90\xdd\x05"
        + struct.pack("<I", tmp + 16)
        + b"\xc3",
    )
    r.handlers[0x421070] = lambda st: r.u.reg_write(UC_X86_REG_EIP, stub)

    def power(st):
        ex, ba = r.get(tmp, "dd")
        r.put(tmp + 16, "d", ba**ex)

    r.handlers[stub + 12] = power
    gamstub = 0x104400
    r.u.mem_write(gamstub, b"\xdd\x05" + struct.pack("<I", tmp + 32) + b"\xc3")
    calls = []

    def gamma(st):
        ix = len(calls)
        sh, ra = r.get(st + 4, "2d")
        calls.append((sh, ra))
        r.put(tmp + 32, "d", tape[(ix // 2) % len(tape)][ix % 2])
        r.u.reg_write(UC_X86_REG_EIP, gamstub)

    r.handlers[0x406330] = gamma
    out0, out1 = r.alloc(8), r.alloc(8)
    r.call(0x4052E0, struct.pack("<2d2I", math.log(0.1), math.log(4), out0, out1), obj)
    ps = [r.get(out0, "d")[0], r.get(out1, "d")[0]]
    calls.clear()
    logvars = r.vec([math.log(0.1), math.log(4)])
    retcode = 0x104500
    r.u.mem_write(
        retcode,
        b"\xdd\x1d"
        + struct.pack("<I", tmp + 48)
        + b"\xe9"
        + struct.pack("<i", r.stop - (retcode + 11)),
    )
    r.u.mem_write(r.sp, struct.pack("<II", retcode, logvars))
    r.u.reg_write(UC_X86_REG_ESP, r.sp)
    r.u.reg_write(UC_X86_REG_ECX, obj)
    r.u.emu_start(0x405290, r.stop, count=20000000)
    assert r.u.reg_read(UC_X86_REG_EIP) == r.stop
    native_obj = r.get(tmp + 48, "d")[0]
    return {
        "tape": tape,
        "doses": [zlow, zhigh],
        "thresholds": [qlow, target],
        "probabilities": ps,
        "gamma_call_count": len(calls),
        "first_shape_rate_pairs": calls[:2],
        "objective": native_obj,
        "independent_objective": sum((x - 0.99) ** 2 for x in ps),
    }


def main():
    global p
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("fixture_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    expected = "3951dc95cdfaaad0da0526cfb25d28b7f970d4d82a6836203da4a469282d8070"
    if hashlib.sha256(args.executable.read_bytes()).hexdigest() != expected:
        raise ValueError("ToxFinder executable hash mismatch")
    p = pefile.PE(str(args.executable))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "simplex",
        "map",
        "information",
        "stage1",
        "allocation30",
        "surface",
        "elicitor-control",
        "elicitation-probability",
    ):
        file = f"toxfinder-native-{name}.json"
        rows = json.loads((args.fixture_directory / file).read_text())
        results = []
        for row in rows:
            if name == "simplex":
                objective = (
                    (lambda x: 4.0)
                    if row.get("constant")
                    else lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2
                )
                result = {**row, **simplex(row["start"], row["step"], objective, row["budget"])}
            elif name == "map":
                result = native_map(row["shapes"], row["rates"], row["doses"], row["outcomes"])
            elif name == "information":
                result = native_information(
                    row["shapes"],
                    row["rates"],
                    row["doses"],
                    row["outcomes"],
                    row["samples"],
                    row["candidate"],
                )
            elif name == "stage1":
                result = stage1(
                    row["levels"],
                    row["history"],
                    row["outcomes"],
                    row["target"],
                    lambda x, y: x * x,
                )
            elif name == "allocation30":
                score = (
                    (lambda x, y: 1.0) if row["objective"] == "constant" else lambda x, y: 2 * x + y
                )
                result = allocation(
                    row["mode"],
                    score,
                    row["goal"],
                    row["grid"],
                    row["ckp"],
                    row["allocate"],
                    row["levels"],
                )
            elif name == "elicitor-control":
                result = elicitor_control(row["mode"])
                result["probability_calls"] = [
                    {"log_variances": r["log_variances"], "probabilities": r["probabilities"]}
                    for r in result.pop("calls")
                ]
            elif name == "elicitation-probability":
                result = elicitation_probability(row["tape"], *row["doses"], *row["thresholds"])
            else:
                result = {
                    "dose": row["dose"],
                    "native": surface(
                        [[0, 0.1, 0.4], [0.2, 0.5, 0.7], [0.3, 0.6, 0.9]], *row["dose"]
                    ),
                }
            if name == "allocation30":
                result["objective"] = row["objective"]
            results.append(result)
        (args.output_directory / file).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
