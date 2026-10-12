import dnfile
from arand_common import library_path
from dncil.cil.body import CilMethodBody as CilMethodBody
from dncil.cil.body.reader import CilMethodBodyReaderBytes as CilMethodBodyReaderBytes

p = dnfile.dnPE(library_path())


def name(s):
    s = str(s)
    if "<" in s:
        s = s[: s.index("<")] + "<...>" + s[s.rfind(">") + 1 :]
    return s


def resolve(op):
    if hasattr(op, "table"):
        tb = p.net.mdtables.tables.get(op.table)
        if tb:
            r = tb.rows[op.rid - 1]
            return name(getattr(r, "Name", getattr(r, "TypeName", str(op))))
    return str(op)
