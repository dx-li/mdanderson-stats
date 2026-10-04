# Saved BARD BF-BOIN studies and reports

`BARDStudySpecification` captures one complete, replayable BF-BOIN scenario:
the design, per-dose toxicity and response truth, categorical response-profile
model, stage-two eligibility and selection settings, both caller-supplied true
OBD labels, simulation controls, and an explicit seed. Save and reload it with
the specification's versioned JSON methods. The specification validates its
inputs and bounded patient-work allowance before simulation.

Pass one to twenty specifications to `bard_design_report`. The report validates
all specifications and their aggregate patient-work budget before starting any
trial. It then runs scenarios serially, using each saved seed independently;
reordering or adding a scenario does not change another scenario's stream.
The resulting immutable `BARDDesignReport` can render a self-contained HTML
document or write one atomically:

First create `bard-example-inputs.json` using the example in the
[saved-study guide](bard-study.md). Then run the one-scenario report:

```python
from mdanderson_stats import BARDStudySpecification, bard_design_report

study = BARDStudySpecification.read_json("bard-example-inputs.json")
report = bard_design_report([study], max_patient_work=20_000_000)
report.write_html("bard-study.html")
```

The report shows the stage-one cutoffs and human-readable protocol settings,
plus the complete versioned input JSON for replay. Its operating-characteristic
tables include per-dose selection and no-selection probabilities with MCSE,
unconditional and selected-only correct-OBD probabilities with their
denominators, sample-size and duration means/SDs/MCSEs, and allocation
imbalance for every modeled factor. No-pair, safety-rejected-pair, and
empty-arm exclusions are stated beside their metric denominators.

The BARD guide describes editable/uploaded scenarios, two OC tables, and
downloadable HTML/Word protocol templates (Guide, Section 2(a)–(b) and Section
3, printed pages 12–15). This community report saves Python inputs and results;
it does not claim to reproduce those native templates, output columns, defaults,
or random-number stream. It reports BF-BOIN operating characteristics. The
separate stochastic BF-BLRM operating-characteristic workflow remains outside
this report. See the [source audit](../research/bard-report-audit.md) for the
source-defined metrics and Python conventions.
