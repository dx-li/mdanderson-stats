# Community distribution licensing path

This is an engineering provenance review of this repository's retained notices,
not new permission, a relicense, or legal clearance. No notices or permissions
were changed. `LICENSE.md` licenses original contributions under MIT and expressly
excludes a blanket MIT claim for adapted material. The complete current wheel
retains commercial-use restrictions and must not be advertised as uniformly
open source or unrestricted for commercial use.

## Explicitly retained restricted adaptations

The following minimum set is established by `THIRD_PARTY_NOTICES.md`, not merely
by finding a restrictive license in a reference-only original archive:

| Family | Delivered file scope identified by notices | Retained condition |
| --- | --- | --- |
| Multc99 | `multc99.py`, `multc99_boundaries.py`, `multc99_calibration.py`, `multc99_randomized.py`, `multc99_report.py`, `multc99_trial.py` | Original README limits source extraction/adaptation to noncommercial use; commercial use requires written permission. |
| RANGE2 / KWRANGE | Optional legacy grouping in `ranges.py` | LEGALITIES requires written permission for incorporation into a commercial package sold to others. |
| MULTI | `_multi_lexer.py`, `multi_input.py`, `multi_session.py`, `multi_report.py`, `multiplicity.py`, `schweder.py`, `pvalue_models.py`, `nonparametric.py`, `nonparametric_testing.py`, `beta_mixture.py`, `beta_mixture_fit.py`, `beta_mixture_ml.py`, `beta_mixture_bootstrap.py`, `beta_mixture_selection.py`, `beta_mixture_testing.py`, `schweder_output.py` | Original section-II terms allow noncommercial source use and require written permission for commercial source use. |
| BLiP | `blip.py` | Its README permits noncommercial use/distribution with source citation; notice says the adaptation retains those terms. |
| SOGS | `sogs.py`, `sogs_simulation.py`, `sogs_reporting.py` | Source reuse is noncommercial; written permission required for commercial source use. |

The Multc99 family scope is conservatively included as a whole because its notice
applies to the `multc99` modules. This is not a claim that every line is copied.
Disabling a runtime option does not remove restricted material from the wheel.

A [static import review](restricted-adaptation-imports-2026-10-12.json) identifies
27 modules in those families and one additional internal import-dependent module,
`blip_custom.py`. Root `__init__.py` eagerly exports affected APIs. These are
packaging dependencies, not a conclusion that licenses automatically propagate
to every dependent line. Dynamic dependencies and the ambiguous families below
remain outside that bounded analysis.

## Additional review boundaries

Several notices distinguish independent mathematical implementations from
restricted original routines used only for numerical reference. For example,
SINGLE and SEQBIN expressly say original code is not incorporated. Their original
archive restrictions alone do not establish restrictions on independently
authored Python expressions of the mathematics. FLECS90 explicitly retains the
source's public-domain declaration; GPL/LGPL adaptations have open-source terms
with their own compliance obligations. These categories must remain separate.

KSB1CI (`kstage_binomial.py`), CTA (`cta*.py`), and RANLIST (`ranlist*.py`) need
particular provenance review before inclusion in a proposed unrestricted subset:
their notices describe independent formulas but also preserve conditions or say
the retained notice applies. This report does not silently resolve that ambiguity
in either direction. Other package-specific terms and any noncopyright rights
also need their recorded scope checked. The five-family table is a minimum set,
not an exhaustive license certification for the other hundreds of modules.

The new `mtpi_legacy_selection.py` uses independently authored equations and
SciPy's isotonic solver. The reference runner executes user-supplied original R
only for comparison; neither original functions nor binary assets are delivered.
This is distinct from translating and bundling source under changed terms.

## Actionable path without removing restrictions

1. Build a file-level provenance manifest: original contribution, adapted source,
   reference-only source, notice/license, evidence URL/hash, and transitive imports.
   Resolve ambiguous notices before certifying a distributable subset.
2. Prepare a genuinely separate open-source core distribution. Exclude restricted
   implementation files at build time, remove their eager exports, and audit
   downstream imports, tests and examples. Keep any legacy compatibility package
   separately labeled under its actual terms. Current optional dependencies do
   not accomplish this separation.
3. For each excluded application, recover a published mathematical specification
   and independently implement the required functionality with documented
   provenance and numerical references. Do not relabel existing adaptations as
   independent or assume a rewrite automatically removes obligations. Retain
   legacy artifacts for permitted local comparison only.
4. Where independent specification is insufficient, maintain the limitation;
   permission from the rights holder is an alternative only through a separately
   authorized process. No author contact or permission request has been made.
5. Validate the actual built archives, imports and installed examples after any
   separation, then review the resulting license claim before public release.

Splitting the package changes the public API and distribution scope. This is a
consequential project decision to resolve before packaging a new release, not a
reason to stop source research or independently authored implementation now.


## Concrete split proposal (not activated)

Keep an audited `mdanderson-stats` core and a separate `mdanderson-stats-legacy`
distribution using the **distinct** `mdanderson_stats_legacy` namespace. An
optional lazy root forwarding layer can preserve selected historical imports
when the add-on is present; it must not make core-only import fail.

Minimum restricted set: 27 notice-named modules plus `blip_custom`, affecting
83 root exports: Multc99 (23), MULTI (37), BLiP (9), SOGS (11), and ranges (3).
Isolating only the restricted `legacy=True` range branches preserves default
range APIs and reduces unavailable whole exports to 80. This is an import/API
impact estimate, not a legal conclusion about independent code.

Ambiguous CTA*, RANLIST* and kstage_binomial exclusions would add 17 seed and
35 dependent modules (52 total), affecting RANDLIB, SPPCR, KSBIN1/2, binomial
sample-size, Fisher design and diagnostic population workflows. Reduce that
impact by independently specifying neutral L'Ecuyer–Côté helpers currently in
ranlist_random, resolving the KStageBinomial recurrence provenance, and auditing
extsig's CTA Fisher and diagnostic_population's CTA diagnostic dependencies.
Dependency reachability does not itself establish license inheritance.

Compatibility review must cover eager `__init__`, `__all__`/star imports,
historical direct submodule imports, moved class pickle identities and version
compatibility. Exclude restricted files from BOTH wheel and sdist; optional
dependencies alone do not separate code. Verify core-only and add-on installs,
including uninstall behavior. Never have two wheels own the same file path.
GPL/LGPL, unclear EVENTCHART/SORTF90 grants and mixed Misclib provenance need
separate review. No blanket unrestricted license claim or package split has
been activated; this remains a consequential distribution decision.
