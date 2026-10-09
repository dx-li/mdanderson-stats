"""Bounded interpretation of original managed Multc Lean model I/O control.

Research-only dnfile/dncil decode a checksum-verified local EXE. Original
instruction order, branches, indexing and model field assignments execute;
.NET objects, filesystem, conversion/formatting, clock and UI services are
substituted. This verifies the schema/default selection, not CLR formatting,
GUI execution or the numerical DLL. Reference generation imports no production
codec; the optional interoperability check supplies Python exports to the
original managed restore control.
"""

import argparse
import copy
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import dnfile
from dncil.cil.body.reader import read_method_body_from_bytes

EXE_SHA256 = "a9d8b5b8e3e0f6d2d30e93c550cda8253c00b0777dd6e5a3a77322eba553c69d"
CONFIG_SHA256 = "4c6fdc133739a4dd935212df4392e3194e13bd0eb7e56a88a50ec82ce1b4c58d"
ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "P_Res_Tox",
    "P_Res_NoTox",
    "P_NoRes_Tox",
    "P_NoRes_NoTox",
    "meanInterArrivalTime",
    "window",
)


class Address:
    def __init__(self, mapping, key):
        self.mapping, self.key = mapping, key

    def value(self):
        return self.mapping[self.key]


class Interpreter:
    def __init__(self, exe, settings):
        if hashlib.sha256(exe.read_bytes()).hexdigest() != EXE_SHA256:
            raise ValueError("original EXE checksum mismatch")
        self.pe = dnfile.dnPE(str(exe))
        self.settings = settings
        self.methods = {
            str(index.row.Name): index.row
            for t in self.pe.net.mdtables.TypeDef
            if str(t.TypeName) == "FormMultcLeanDesktop"
            for index in t.MethodList
        }
        self.output = io.StringIO()
        self.input_text = ""

    def row(self, token):
        return self.pe.net.mdtables.tables[token.table].rows[token.rid - 1]

    def field(self, token):
        return str(self.row(token).Name)

    def invoke(self, method, form, text=""):
        body = read_method_body_from_bytes(self.pe.get_data(self.methods[method].Rva, 0x20000))
        instructions = body.instructions
        positions = {ins.offset: i for i, ins in enumerate(instructions)}
        stack, locals_, args = [], {}, [form]
        self.input_text = text
        self.output = io.StringIO()
        pc = 0
        for _ in range(100000):
            ins = instructions[pc]
            op, val = ins.opcode.name, ins.operand
            pc += 1
            if op == "ldnull":
                stack.append(None)
            elif op == "ldstr":
                stack.append(self.pe.net.user_strings.get(val.rid).value)
            elif op.startswith("ldc.i4."):
                stack.append(
                    -1
                    if op.endswith("m1")
                    else int(val)
                    if op.endswith("s")
                    else int(op.rsplit(".", 1)[1])
                )
            elif op in ("ldc.i4", "ldc.r8"):
                stack.append(val)
            elif op.startswith("ldarg."):
                stack.append(args[val.index if op.endswith("s") else int(op.rsplit(".", 1)[1])])
            elif op in ("ldloca.s", "ldloca"):
                stack.append(Address(locals_, val.index))
            elif op.startswith(("ldloc.", "stloc.")):
                index = val.index if op.endswith("s") else int(op.rsplit(".", 1)[1])
                if op.startswith("ld"):
                    stack.append(locals_.get(index))
                else:
                    locals_[index] = stack.pop()
            elif op == "ldfld":
                stack.append(stack.pop()[self.field(val)])
            elif op == "ldflda":
                stack.append(Address(stack.pop(), self.field(val)))
            elif op == "stfld":
                value, obj = stack.pop(), stack.pop()
                obj[self.field(val)] = value
            elif op in ("box", "castclass"):
                pass
            elif op == "dup":
                stack.append(stack[-1])
            elif op == "pop":
                stack.pop()
            elif op == "add":
                right, left = stack.pop(), stack.pop()
                stack.append(left + right)
            elif op == "newarr":
                stack.append([None] * stack.pop())
            elif op == "ldelem":
                index, array = stack.pop(), stack.pop()
                stack.append(array[index])
            elif op in ("br", "br.s", "leave", "leave.s"):
                if op.startswith("leave"):
                    stack.clear()
                pc = positions[val]
            elif op in ("brtrue", "brtrue.s", "brfalse", "brfalse.s"):
                condition = bool(stack.pop())
                if condition == op.startswith("brtrue"):
                    pc = positions[val]
            elif op in ("bgt", "blt"):
                right, left = stack.pop(), stack.pop()
                if left > right if op == "bgt" else left < right:
                    pc = positions[val]
            elif op in ("call", "callvirt", "newobj"):
                row = self.row(val)
                signature = row.Signature.value
                if signature[0] & 0x10:
                    raise ValueError("generic calls are outside this bounded interpreter")
                count, return_type = signature[1:3]
                if count >= 128:
                    raise ValueError("large signatures are unsupported")
                params = stack[-count:] if count else []
                if count:
                    del stack[-count:]
                obj = None if op == "newobj" or not signature[0] & 0x20 else stack.pop()
                name = str(row.Name)
                owner = (
                    str(getattr(row.Class.row, "TypeName", ""))
                    if val.table == 0x0A
                    else "FormOrSettings"
                )
                result = self.service(name, owner, obj, params, op == "newobj")
                if op == "newobj" or return_type != 1:
                    stack.append(result)
            elif op == "ret":
                return stack.pop() if stack else None
            else:
                raise ValueError(f"unsupported original instruction: {op}")
        raise RuntimeError("managed instruction budget exceeded")

    def service(self, name, owner, obj, params, constructor):
        if constructor:
            if owner == "StreamReader":
                return iter(self.input_text.splitlines())
            if owner == "StreamWriter":
                return self.output
            if owner == "String":
                return chr(params[0]) * params[1]
            if owner in ("MProbabilityParameters", "MScenarioParameters"):
                return {}
            raise ValueError(f"unsupported constructor: {owner}")
        if name == "get_UserAppDataPath":
            return "/virtual"
        if name == "Exists":
            return True
        if name == "get_InvariantInfo":
            return "invariant"
        if name == "get_Default":
            return self.settings
        if name == "get_Now":
            return "controlled-clock"
        if name == "VersionInfo":
            return "controlled-version"
        if name in ("DisplayModelParametersInControls", "Clear", "SetError", "Flush", "Close"):
            return None
        if name == "ReadLine":
            return next(obj, None)
        if name == "get_Chars":
            return ord(obj[params[0]])
        if name == "CompareTo":
            return (obj > params[0]) - (obj < params[0])
        if name == "ToCharArray":
            return list(obj)
        if name == "Split":
            return obj.split(params[0][0], params[1] - 1)
        if name == "ToDouble":
            return float(params[0])
        if name == "ToInt32":
            return int(params[0])
        if name == "ToBoolean":
            return params[0].lower() == "true"
        if name == "Concat":
            return "".join(params)
        if name == "ToString":
            value = obj.value() if isinstance(obj, Address) else obj
            if isinstance(value, bool):
                return str(value)
            return format(value, ".15g") if isinstance(value, float) else str(value)
        if name in ("Write", "WriteLine"):
            value = str(params[0])
            if len(params) == 2:
                value = value.replace("{0:G}", str(params[1])).replace("{0}", str(params[1]))
            obj.write(value + ("\r\n" if name == "WriteLine" else ""))
            return None
        if name == "get_Count":
            return len(obj)
        if name == "get_Item":
            return obj[params[0]]
        if name == "SetValue":
            obj[params[1]] = params[0]
            return None
        if name.startswith("get_"):
            return obj[name[4:]]
        if name.startswith("set_"):
            obj[name[4:]] = params[0]
            return None
        raise ValueError(f"unsupported service: {owner}.{name}")


def reference(exe, config):
    if hashlib.sha256(config.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("original configuration checksum mismatch")
    settings = {}
    for setting in ET.parse(config).findall("./applicationSettings/*/setting"):
        name, value = setting.attrib["name"], setting.findtext("value")
        settings[name] = (
            value == "True"
            if name.startswith("bo")
            else int(value)
            if name.startswith("i")
            else float(value)
        )
    interpreter = Interpreter(exe, settings)
    default = {
        "m_responseParams": {},
        "m_toxicityParams": {},
        "m_errorProvider": {},
        "m_rtfStopBoundOutput": {},
    }
    for name in (
        "RespBetaA",
        "RespBetaB",
        "ExpRespPriorBetaA",
        "ExpRespPriorBetaB",
        "ToxBetaA",
        "ToxBetaB",
        "ExpToxPriorBetaA",
        "ExpToxPriorBetaB",
    ):
        default["m_dtb" + name] = {}
    interpreter.invoke("ResetModelParametersInControl", default)
    meaningful = (
        "m_responseParams",
        "m_toxicityParams",
        "m_maxPatients",
        "m_minPatients",
        "m_cohortSize",
    )
    default = {key: default[key] for key in meaningful}
    cases = []
    for index in range(8):
        form = copy.deepcopy(default)
        if index % 2:
            form["m_responseParams"].update(use_standard_constant=True, standard_constant=0.45)
        if index % 3 == 1:
            form["m_toxicityParams"].update(use_standard_constant=True, standard_constant=0.2)
        if index >= 4:
            form.update(m_maxPatients=24, m_minPatients=6, m_cohortSize=3)
            form["m_responseParams"]["delta"] = 0.1
            form["m_toxicityParams"]["delta"] = -0.05
        scenarios = [
            [0.1, 0.2, 0.3, 0.4, 0, 0],
            [0.25, 0.25, 0.25, 0.25, 0.2, 2],
            [0, 0.5, 0.25, 0.25, 0, 2],
        ][: index % 4]
        form["m_dataScenarioParams"] = {
            "Rows": [dict(zip(FIELDS, values, strict=True)) for values in scenarios]
        }
        for i, field in enumerate(FIELDS):
            form[f"m_dataGridTextBoxColumn{i}"] = {"HeaderText": field}
        interpreter.invoke("SaveModelToFile", form)
        text = interpreter.output.getvalue()
        restored = {}
        assert interpreter.invoke("RestoreModelFromFile", restored, text) == 1
        cases.append({"case": f"model_{index}", "text": text, "restored": restored})
    ctor = read_method_body_from_bytes(
        interpreter.pe.get_data(interpreter.methods[".ctor"].Rva, 0x1000)
    )
    repetitions = None
    for i, ins in enumerate(ctor.instructions):
        if ins.opcode.name == "stfld" and interpreter.field(ins.operand) == "m_defaultRepetitions":
            previous = ctor.instructions[i - 1]
            assert previous.opcode.name == "ldc.i4"
            repetitions = previous.operand
    assert repetitions is not None
    return {
        "exe_sha256": EXE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "defaults": default,
        "default_repetitions": repetitions,
        "scope": (
            "Original managed model I/O/default-selection control; substituted CLR/IO/UI services"
        ),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-codec", action="store_true")
    args = parser.parse_args()
    result = reference(args.exe, args.config)
    destination = ROOT / "tests/fixtures/multc-lean-managed-model.json"
    if args.check:
        assert result == json.loads(destination.read_text())
        print("Verified eight original managed model I/O cases and configured defaults")
    else:
        destination.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {destination.name}")
    if args.check_codec:
        from mdanderson_stats import parse_multc_lean_model

        interpreter = Interpreter(args.exe, {})
        for case in result["cases"]:
            exported = parse_multc_lean_model(case["text"]).to_model_text()
            restored = {}
            assert interpreter.invoke("RestoreModelFromFile", restored, exported) == 1
            assert restored == case["restored"]
        # Check more than the original writer's short general-format examples.
        model = parse_multc_lean_model(result["cases"][0]["text"])
        from dataclasses import replace

        precise = replace(
            model,
            response=replace(model.response, experimental_a=0.12345678901234568),
        )
        restored = {}
        assert interpreter.invoke("RestoreModelFromFile", restored, precise.to_model_text()) == 1
        assert restored["m_responseParams"]["experimental_a"] == precise.response.experimental_a
        print("Verified nine Python exports through original managed restore control")


if __name__ == "__main__":
    main()
