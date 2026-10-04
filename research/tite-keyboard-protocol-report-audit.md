# TITE-Keyboard protocol-report source crosswalk

The cached application snapshot `research/raw/TITE-KEYBOARD/app.html` exposes a
Decision Table and Design Flow Chart (lines 378–425), an Operating
Characteristics pane (lines 612–626), scenario entry by typing or CSV upload,
trial count and seed controls (lines 481–559), and downloads for HTML, Word,
Figure 1, and Table 1 (lines 675–712). It exposes a default suspension rule of
more than 50% pending at the current dose and an adjustable limit up to 65%
(lines 367–378). Simulation setup exposes uniform, Weibull, and log-logistic
conditional event timing, with the late-half probability for parametric timing
(lines 440–478). The cached page contains output placeholders, not generated OC
column definitions.

The Python report composes existing `KeyboardDesign`,
`tite_keyboard_boundaries`, and `simulate_tite_keyboard` behavior. The decision
flow follows the existing trial/controller implementation: enrolled-count
safety and lowest-dose stopping; unavailable-dose handling; pending-fraction
suspension; posterior-key move; the two-ascertained-outcome escalation gate;
precision stopping; and final complete-data MTD selection. Its boundary table
separates posterior transitions from enrolled-count safety thresholds.

Each scenario is simulated from its captured truth vector, plan, timing inputs,
and child seed derived in scenario order from the recorded root seed. Uniform
event masses omitted by the caller are passed as `None` to preserve the
simulator's default random stream, while the report records the effective equal
thirds. Scenario histories are discarded after calculation; only immutable
summary arrays and stop-reason frequencies remain. Adaptive timing is an
explicit Python extension. Its settings and aggregate work budget are captured,
and the simulator's derived outcome/sampler seeds and compact diagnostics are
reported. Adaptive timing and caller-supplied trimester analysis masses are
mutually exclusive.

The app page establishes that an OC table exists, but its empty HTML output
placeholder does not establish the native report schema or every column
definition. The paper does explicitly
define two trial-level allocation risks in Section 4.1: fewer than six patients
allocated to the MTD, and more than half of patients treated above the MTD.
The Python simulator now reports these when a caller supplies the true MTD
index. It counts patients above that dose against each trial's actual enrolled
total, including early-stopped trials. This provides the paper's defined
community metrics without claiming the app's hidden true-MTD convention.
Without an explicit true-MTD index, the report continues to give selection
probabilities (including no MTD), enrollment/DLT means, duration, suspension
time, and stop-reason frequencies without assigning unverified estimands.
Together these summaries and the two paper-defined risks cover the identified
community simulation calculations. Exact native random-stream parity and
HTML/Word/Figure 1/Table 1 file parity remain presentation/engine differences.
The app's exact OC column layout and remaining native conventions are not
recoverable from the cached placeholder page.
