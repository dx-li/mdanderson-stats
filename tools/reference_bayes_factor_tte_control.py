"""Bounded original managed BayesFactorTTE v1.1 calendar/control execution.

Research-only dnfile/dncil inspect a checksum-verified local DLL. Original
arrival truncation, ledger, stopping comparisons and trial loop run unchanged.
Constructors/runtime objects and exponential/Bayes-factor services receive
explicit tapes; this isolates controller rules from RNG and quadrature identity.
No production implementation is imported, and original binaries are not shipped.
"""

import argparse
import hashlib
import json
from pathlib import Path

import dnfile
from dncil.cil.body.reader import read_method_body_from_bytes

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "tests/fixtures/bayes-factor-tte-control.json"
DLL_SHA256 = "baa2915f31d1edb617b656a8dbc4eee78ed8629373715b88654fd68a2a00af8f"


class Address:
    def __init__(self, mapping, key):
        self.mapping, self.key = mapping, key

    def get(self):
        return self.mapping[self.key]

    def set(self, value):
        self.mapping[self.key] = value


class NativeControl:
    def __init__(self, dll, exponentials, probabilities, maximum):
        if hashlib.sha256(dll.read_bytes()).hexdigest() != DLL_SHA256:
            raise ValueError("original managed DLL checksum mismatch")
        self.pe = dnfile.dnPE(str(dll))
        self.methods = {}
        self.owners = {}
        for t in self.pe.net.mdtables.TypeDef:
            for m in t.MethodList:
                self.owners[id(m.row)] = str(t.TypeName)
                self.methods[(str(t.TypeName), str(m.row.Name), m.row.Signature.value[1])] = m.row
        self.exponentials = iter(exponentials)
        self.probabilities = iter(probabilities)
        self.maximum = maximum
        self.history = []
        self.calls = []
        self.ledger = None
        self.boundary_mode = False

    def row(self, token):
        return self.pe.net.mdtables.tables[token.table].rows[token.rid - 1]

    def invoke(self, owner, name, arguments):
        instance, *params = arguments
        row = self.methods[(owner, name, len(params))]
        body = read_method_body_from_bytes(self.pe.get_data(row.Rva, 0x10000))
        instructions = body.instructions
        positions = {ins.offset: i for i, ins in enumerate(instructions)}
        stack, locals_, pc = [], {}, 0
        for _ in range(100000):
            ins = instructions[pc]
            op, value = ins.opcode.name, ins.operand
            pc += 1
            if op.startswith("ldc.i4."):
                stack.append(
                    -1
                    if op.endswith("m1")
                    else int(value)
                    if op.endswith("s")
                    else int(op.rsplit(".", 1)[1])
                )
            elif op in ("ldc.i4", "ldc.r8"):
                stack.append(value)
            elif op.startswith("ldarg."):
                stack.append(arguments[value.index if op.endswith("s") else int(op[-1])])
            elif op == "ldloca.s":
                stack.append(Address(locals_, value.index))
            elif op.startswith(("ldloc.", "stloc.")):
                index = value.index if op.endswith("s") else int(op[-1])
                if op.startswith("ld"):
                    stack.append(locals_.get(index))
                else:
                    locals_[index] = stack.pop()
            elif op == "ldfld":
                stack.append(stack.pop()[str(self.row(value).Name)])
            elif op == "stfld":
                field_value, obj = stack.pop(), stack.pop()
                obj[str(self.row(value).Name)] = field_value
            elif op == "ldsfld":
                field = str(self.row(value).Name)
                stack.append({"Zero": 0.0, "_DaysPerYear": 365.25}[field])
            elif op == "ldstr":
                stack.append(self.pe.net.user_strings.get(value.rid).value)
            elif op == "dup":
                stack.append(stack[-1])
            elif op in ("add", "sub", "mul", "div"):
                right, left = stack.pop(), stack.pop()
                stack.append(
                    {
                        "add": lambda: left + right,
                        "sub": lambda: left - right,
                        "mul": lambda: left * right,
                        "div": lambda: left / right,
                    }[op]()
                )
            elif op in ("conv.i4", "conv.r8"):
                stack.append(int(stack.pop()) if op == "conv.i4" else float(stack.pop()))
            elif op == "ldelem.i4":
                index, array = stack.pop(), stack.pop()
                stack.append(array[index])
            elif op == "stelem.i4":
                item, index, array = stack.pop(), stack.pop(), stack.pop()
                array[index] = item
            elif op in ("ldind.ref", "ldind.i4"):
                stack.append(stack.pop().get())
            elif op in ("stind.ref", "stind.i4", "stind.r8"):
                item, address = stack.pop(), stack.pop()
                address.set(item)
            elif op == "pop":
                stack.pop()
            elif op in ("br.s", "br", "leave.s"):
                pc = positions[value]
            elif op in ("brtrue.s", "brfalse.s"):
                if bool(stack.pop()) == op.startswith("brtrue"):
                    pc = positions[value]
            elif op in ("blt.s", "bge.s", "ble.un.s", "bge.un.s", "beq.s"):
                right, left = stack.pop(), stack.pop()
                if {
                    "blt.s": left < right,
                    "bge.s": left >= right,
                    "ble.un.s": left <= right,
                    "bge.un.s": left >= right,
                    "beq.s": left == right,
                }[op]:
                    pc = positions[value]
            elif op in ("call", "callvirt", "newobj"):
                called = self.row(value)
                signature = called.Signature.value
                count, returns = signature[1:3]
                params = stack[-count:] if count else []
                if count:
                    del stack[-count:]
                obj = None if op == "newobj" or not signature[0] & 0x20 else stack.pop()
                called_owner = (
                    str(getattr(called.Class.row, "TypeName", ""))
                    if value.table == 0x0A
                    else self.owners[id(called)]
                )
                result = self.service(called_owner, str(called.Name), obj, params, op == "newobj")
                if op == "newobj" or returns != 1:
                    stack.append(result)
            elif op == "ret":
                return stack.pop() if stack else None
            else:
                raise ValueError(f"unsupported original instruction {op}")
        raise RuntimeError("original managed instruction budget exceeded")

    def service(self, owner, name, obj, params, constructor):
        if constructor:
            if owner in ("", "StringBuilder"):
                return []
            if owner == "BayesFactor":
                return {}
            if owner == "Decision":
                return {"hypothesis": 1}
            if owner == "PatientLogSummary":
                self.ledger = {
                    "timeEnrollDays": [0] * self.maximum,
                    "timeEventDays": [0] * self.maximum,
                }
                return self.ledger
            if owner == "TrialConductor":
                return {"m_BayesFactor": {}, "m_Simulator": {"m_design": params[0]}}
            raise ValueError(f"unsupported constructor {owner}")
        if owner == "TrialDesign" and name == "get_AltMeanTTEMonths":
            return obj["altMedianTTEMonths"] / __import__("math").log(2)
        if owner == "BiostatTimeSpan" and name == "InitMonths":
            return params[-1] * 30.4375
        if owner == "BiostatTimeSpan" and name == "TotalYears":
            return params[0] / 365.25
        if name == "Abs":
            return abs(params[0])
        if name == "Add" and isinstance(obj, list):
            obj.append(params[0])
            return None
        if owner == "RNG" and name == "GetExponential":
            value = next(self.exponentials)
            self.calls.append({"mean": params[0], "value": value})
            return value
        if owner == "TrialDesign" and name == "get_accrualRatePerDay":
            return obj["accrualRatePerMonth"] / 30.4375
        if owner == "BayesFactor" and name == "GetBayesFactor":
            if self.boundary_mode:
                params[3].set(params[0] / (params[1] + 1))
                return None
            probability = next(self.probabilities)
            params[3].set(probability / (1 - probability))
            self.history.append(
                {
                    "events": params[1],
                    "exposure_days": params[0],
                    "patients": self.ledger["patientsTreated"],
                    "probability": probability,
                }
            )
            return None
        if owner == "Decision" and name == "get_stop":
            return obj["hypothesis"] != 1
        return self.invoke(owner, name, [obj, *params])


def references(dll):
    output = []
    exponentials = [1.9, 0.8, 0.2, 2.9, 3.7, 1.2, 2.0, 4.9]
    for probabilities in ([0.5, 0.5, 0.5], [0.1], [0.5, 0.9]):
        original = NativeControl(dll, exponentials, probabilities, 4)
        design = {"maxPatients": 4, "accrualRatePerMonth": 2, "nullCutoff": 0.15, "altCutoff": 0.8}
        instance = {"m_design": design, "cache": {}}
        result = {}
        original.invoke(
            "Simulator",
            "SimulateOneTrial",
            [
                instance,
                100.0,
                Address(result, "hypothesis"),
                Address(result, "patients"),
                Address(result, "error"),
            ],
        )
        output.append(
            {
                "exponential_values": exponentials,
                "probabilities": probabilities,
                "result": result,
                "ledger": original.ledger,
                "calls": original.calls,
                "history": original.history,
            }
        )
    native = NativeControl(dll, [], [], 4)
    native.boundary_mode = True
    design = {"maxPatients": 4, "altMedianTTEMonths": 5.5, "nullCutoff": 0.15, "altCutoff": 0.8}
    boundaries = {}
    native.invoke(
        "Simulator",
        "GetBoundariesInDays",
        [
            {"m_design": design, "cache": {}},
            Address(boundaries, "inferiority"),
            Address(boundaries, "superiority"),
            Address(boundaries, "error"),
        ],
    )
    return {
        "dll_sha256": DLL_SHA256,
        "cases": output,
        "synthetic_bf_boundary_rule": "BF=exposure_days/(events+1)",
        "integer_boundaries": boundaries,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(references(args.dll), indent=2) + "\n"
    if args.check:
        if DESTINATION.read_text() != text:
            raise ValueError("original managed reference changed")
        print("Verified three original managed calendar/controller cases")
    else:
        DESTINATION.write_text(text)
        print("Wrote three original managed calendar/controller cases")


if __name__ == "__main__":
    main()
