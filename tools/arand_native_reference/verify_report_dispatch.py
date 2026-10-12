"""Resolve the noncovariate wrapper override and its base report candidate.

The metadata truncates long instantiated C++ method names to hash suffixes.
Consequently suffix-searching PrintSimulationSummary alone finds the covariate
instantiation. This probe resolves the actual noncovariate constructor/vtable.
"""

import json
import struct
from pathlib import Path

from arand_il import CilMethodBody, CilMethodBodyReaderBytes, p
from arand_report_index_reference import run


def body(rva):
    return CilMethodBody(CilMethodBodyReaderBytes(p.get_data(rva, 100000))).instructions


def main():
    fields = {r.Field.row_index: r.Rva for r in p.net.mdtables.FieldRva.rows}
    constructor = body(0x17B48)
    assert constructor[1].mnemonic == "ldsflda" and constructor[1].operand.rid == 0x6F6
    table = fields[0x6F6]
    thunk = struct.unpack("<I", p.get_data(table + 12, 4))[0] - p.OPTIONAL_HEADER.ImageBase
    assert thunk == 0x57B60
    assert p.get_data(thunk, 2) == b"\xff\x25"
    assert any(r.Rva == 0x57B68 for r in p.net.mdtables.MethodDef)
    # Wrapper StartSimulation invokes the scenario's vtable offset 12.
    wrapper = {i.offset: i for i in body(0xBE19C)}
    assert wrapper[455].operand == 12 and wrapper[459].mnemonic == "calli"
    xml = {i.offset: i for i in body(0x23FA8)}
    assert xml[410].mnemonic == "ldsflda" and xml[410].operand.rid == 0x524
    derived_table = fields[0x524]
    derived_entry = (
        struct.unpack("<I", p.get_data(derived_table + 12, 4))[0] - p.OPTIONAL_HEADER.ImageBase
    )
    assert derived_entry == 0x18D3E0
    assert p.get_data(derived_table + 28, 4) == p.get_data(table + 28, 4)
    comparisons = []
    for label, old, new, offsets in [
        ("report", 0x60A14, 0x57B68, {1194: (1560, 720)}),
        ("accumulation", 0x62894, 0x59AC0, {26: (1552, 712), 58: (1552, 712), 88: (1552, 712)}),
    ]:
        token_differences = []
        for x, y in zip(body(old), body(new), strict=True):
            assert (x.offset, x.mnemonic) == (y.offset, y.mnemonic)
            if str(x.operand) != str(y.operand):
                if x.offset in offsets:
                    assert (x.operand, y.operand) == offsets[x.offset]
                else:
                    assert x.mnemonic in ("call", "ldftn", "ldsflda")
                    token_differences.append(x.offset)
        comparisons.append(
            dict(
                name=label,
                prior_rva=hex(old),
                noncovariate_rva=hex(new),
                instructions=len(body(new)),
                class_layout_substitutions=offsets,
                metadata_token_substitution_offsets=token_differences,
            )
        )
    # Execute the same bounded percentile slice at the actual noncovariate RVA.
    for n in (0, 1, 20, 39, 40, 99, 100, 101, 120, 199, 200, 1000):
        assert run(n, rva=0x57B68) == run(n)
    out = dict(
        constructor_rva="0x17b48",
        vtable_field="0x040006f6",
        vtable_rva=hex(table),
        report_slot=12,
        report_thunk_rva=hex(thunk),
        base_report_method_rva="0x57b68",
        actual_wrapper_entry_rva=hex(derived_entry),
        derived_vtable_field="0x04000524",
        derived_vtable_rva=hex(derived_table),
        derived_vtable_assignment_xml_offset=410,
        accumulation_entry_shared=True,
        wrapper_start_rva="0xbe19c",
        wrapper_call_offset=459,
        comparisons=comparisons,
        limitations=[
            "Bounded percentile execution and dispatch/layout inspection; "
            "base candidate only; derived x86 override is not fully executed."
        ],
    )
    Path("research/fixtures/arand-native/report-dispatch.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print("Derived wrapper override found at 0x18d3e0; 12 BASE-report percentile probes agree")


if __name__ == "__main__":
    main()
