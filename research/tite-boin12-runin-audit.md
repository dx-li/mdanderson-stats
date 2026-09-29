# TITE-BOIN12 optional 3+3 run-in

The recovered `RunIn3+3.txt` states that the option is available only when the
toxicity limit is 0.25. At exactly 3 or 6 patients treated at the current dose,
it triggers de-escalation when at least 2 DLTs have been observed. At all other
current-dose sample sizes, it says conduct remains coherent but specifies no
additional 3+3 transition rule. This implementation therefore adds only the
stated de-escalation override; it does not invent classic 3+3 escalation,
expansion, or persistent run-in states.

Python conduct policy: the existing pending-information gate runs first, so a
decision is suspended when that gate is exceeded. The posterior and global
safety/admissibility check then run; the run-in override is applied before the
precision stop and ordinary BOIN12 neighbor rules. DLT count uses observed
toxicity events among all patients assigned to the current dose. The override
chooses only the immediately lower dose, and only when that dose is admissible
and not already eliminated. At the lowest dose, or when the lower neighbor is
unavailable, conduct stops without assigning another dose. These ordering and
unavailable-neighbor conventions are Python choices because the recovered note
does not specify them. The source-supported threshold and sample sizes are
preserved exactly.


The lowest-dose `stop_safety` action is caused by the run-in DLT rule. It does
not itself set a posterior-admissible dose's elimination flag; the returned
mask continues to represent the existing prior/posterior admissibility rules.
This distinction avoids inventing an additional permanent elimination rule.

Twenty focused conduct and independent-reference tests pass, with 130.38 MiB
worker peak RSS and zero swaps. Targeted mypy, Ruff and formatting checks pass
on the committed change. Both integrated guide examples execute. Root also
checks that the 2/6-DLT override precedes an enabled precision stop and that a
lowest-dose run-in safety stop can retain a posterior-admissible elimination
mask. The root process peaks at 122.73 MiB with zero swaps. No full suite or
new CI checks were added; the remaining data-augmentation, categorical and
calendar workflows are unchanged.
