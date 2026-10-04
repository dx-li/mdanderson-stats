# ASYPOW calculation workflow source crosswalk

The workflow wrapper composes the existing Python calculation kernels; it
does not add a new information or divergence formula.

| Native/source contract | Existing Python implementation | Workflow route |
| --- | --- | --- |
| LR significance, power, sample-size calculations for linear nulls | `asypow.py:AsymptoticPower`, `asypow_information` | `lr_information` and typed `lr_*` routes |
| Independent-group, regression, ordinal, multinomial and general-design information | `asypow_groups.py`, `asypow_regression.py`, `asypow_ordinal.py`, `asypow_multinomial.py`, `asypow_design.py` | `lr_groups`, `lr_regression`, `lr_ordinal`, `lr_ordinal_regression`, `lr_multinomial`, `lr_design` |
| SMO null fitting and target calculations | `asypow_smo*.py`, `asypow_generic.py` | named `asypow_smo_*` routes |
| Native calculation object stores model/method/design and all three target values | `research/raw/ASYPOW/original/asypow/S/binomial.kgp.s:65-78,150-194` | request selects exactly two targets; wrapper calculates and reports the third |
| Native printer reports model inputs and result columns to the console | `research/raw/ASYPOW/original/asypow/S/print.asypow.s:18-45,63-80,85-203` | deterministic bounded `AsyPowCalculation.report()` |

The printer source does not specify a durable report-file schema. The Python
report is therefore a convenience representation, not a native file-format
or byte-output compatibility claim. The generic SMO callback's closure state
cannot be reconstructed from a qualified callable name. No ASYPOW numerical
method is omitted from the workflow registry: all existing LR information
builders, generic LR, and available SMO constructors are mapped. Source
anchors are from the cached local source tree; no new source was retrieved.
