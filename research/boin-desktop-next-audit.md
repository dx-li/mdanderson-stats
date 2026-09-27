# Next catalog reconciliation: BOIN desktop

Entry 99 remains pending. Its [official page](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/99)
is readable and identifies version 1.1.0, modified December 22, 2021. It covers
single-agent BOIN, late-toxicity TITE-BOIN, combination single-MTD and combination
MTD-contour designs, simulation, English/Chinese protocol generation, 3+3 and
Rolling 6 comparisons, and a weighted standardized-follow-up calculator.

The repository already implements many of these method families under separate
web-app catalog entries, including `boin.py`, `tite_boin.py`, combination and
waterfall helpers, 3+3 simulation, Rolling 6 rules and protocol text. Reconcile
the desktop specification against those APIs and their existing native references
before implementing anything duplicate or changing entry 99's coverage status.
The BOIN documentation still calls combination/time-to-event variants unaudited;
that statement predates their subsequent ports and needs a scoped correction
during the desktop audit.

The archive is recorded as `BOIN_V1.1.0.zip`. Its embedded help and native
workflow differences have not been inspected in this pass. The author's
[software page](https://odin.mdacc.tmc.edu/~yyuan/index_code.html) provides links
to the R tutorial and single-agent/combination protocol templates. Next inspect
those contracts, identify actual missing calculations/workflows and delegate
the next coherent implementation to Luna. Do not mark complete from shared
method names alone or rerun large studies already supported by checked-in
references.

The author's linked [BOIN 2.4 manual](https://odin.mdacc.tmc.edu/~yyuan/Software/BOIN/BOIN2.4_manual.pdf)
and [tutorial](https://odin.mdacc.tmc.edu/~yyuan/Software/BOIN/BOIN2.4_tutorial.pdf)
were readable on September 27. They describe the historical R interfaces,
including combination/waterfall selection, rather than the desktop 1.1.0
embedded help. The two protocol-template links returned retrieval errors.
Existing `rolling_six` conduct, replay and simulation APIs already use the shared
timing scenario functions; the remaining comparison gap concerns coordinated
summaries/report generation, not a missing Rolling Six simulator. Any new
comparison should reuse those implementations and report their explicit Python
scheduling conventions rather than imply native desktop scheduling parity.
