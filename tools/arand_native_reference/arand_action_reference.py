"""Original SetAction IL with supplied string comparison and one status cell.

The constructor separately initializes both action cells to 2; this probe checks
recognized, absent and unrecognized strings with explicit initial cell values.
"""

import json

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p


def run(action, initial=2):
    ins = CilMethodBody(CilMethodBodyReaderBytes(p.get_data(0x6F5A4, 10000))).instructions
    positions = {x.offset: i for i, x in enumerate(ins)}
    field_rvas = {r.Field.row_index: r.Rva for r in p.net.mdtables.FieldRva.rows}
    stack = []
    result = initial
    pc = 0
    for _ in range(100):
        x = ins[pc]
        pc += 1
        op, v = x.mnemonic, x.operand
        if op == "ldarg.1":
            stack.append(action)
        elif op == "ldarg.2":
            stack.append("status_cell")
        elif op == "ldsflda":
            stack.append(p.get_data(field_rvas[v.rid], 100).split(b"\0")[0].decode("ascii"))
        elif op.startswith("ldc.i4."):
            stack.append(int(op.rsplit(".", 1)[1]))
        elif op == "call":
            name = str(p.net.mdtables.tables[v.table].rows[v.rid - 1].Name)
            if name.endswith(".compare"):
                b, a = stack.pop(), stack.pop()
                stack.append(int(a != b))
            elif name.endswith("._Tidy"):
                del stack[-3:]
            else:
                raise ValueError(name)
        elif op in ("brtrue.s", "brfalse.s"):
            condition = bool(stack.pop())
            if condition == (op == "brtrue.s"):
                pc = positions[v]
        elif op == "leave.s":
            pc = positions[v]
        elif op == "stind.i4":
            result = stack.pop()
            assert stack.pop() == "status_cell"
        elif op == "ret":
            return {"action": action, "initial": initial, "status": result}
        else:
            raise ValueError((x.offset, op))
    raise RuntimeError("instruction budget exhausted")


if __name__ == "__main__":
    for action in ("open", "tmp_close", "close_early", "close", "", "unknown", "OPEN"):
        print(json.dumps(run(action), sort_keys=True))
    print(json.dumps(run("", initial=1), sort_keys=True))
