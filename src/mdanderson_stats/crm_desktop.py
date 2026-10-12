"""CRM Suite 1.0.0 saved HTML studies and Python simulation reports.

The native XML payload is preserved, including inactive parameter slots and
unknown fields. See docs/crm-desktop.md for supported execution settings.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import escape

import numpy as np

from .crm_simulation import CRMSimulation, simulate_crm
from .dacrm_priors import dacrm_trimester_prior

_GUID = "A18959FB-1E57-4CE0-AAB1-B038A7B78423"
_MAX_DOCUMENT = 2_000_000


def _text(node: ET.Element, path: str) -> str:
    value = node.findtext(path)
    if value is None:
        raise ValueError(f"native study is missing {path}")
    return value.strip()


def _number(node: ET.Element, path: str) -> float:
    value = float(_text(node, path))
    if not np.isfinite(value):
        raise ValueError(f"native study {path} must be finite")
    return value


def _integer(node: ET.Element, path: str, low: int, high: int) -> int:
    value = _number(node, path)
    if value != int(value) or not low <= value <= high:
        raise ValueError(f"native study {path} must be an integer in [{low},{high}]")
    return int(value)


@dataclass(frozen=True)
class CRMDesktopStudy:
    """Validated immutable native study payload; patient dates remain literal text."""

    xml: str
    skeletons: tuple[tuple[float, ...], ...]
    scenario_names: tuple[str, ...]
    scenarios: tuple[tuple[float, ...], ...]
    dose_labels: tuple[str, ...]
    data_augmentation: bool
    patient_count: int

    def to_html(self) -> str:
        """Export preserved settings in the native HTML/XML envelope."""
        return _envelope("<h1>CRM Suite study</h1><p>No Python report generated.</p>", self.xml)


def _envelope(body: str, xml: str) -> str:
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>CRM study</title>'
        "<style>body{font:16px sans-serif;max-width:1100px;margin:2em auto}"
        "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.4em}</style>"
        f"</head><body>{body}</body></html>\n<!--\nbegin{_GUID}\n{xml}\nend{_GUID}\n-->"
    )


def read_crm_desktop_html(document: str) -> CRMDesktopStudy:
    """Read a CRM Suite 1.0.0 HTML string without executing its HTML content.

    Original report markup is discarded. All XML fields, including inactive
    skeleton probabilities, patient date text and unknown settings, are retained
    semantically. External entities/DTDs and oversized documents are rejected.
    """
    if not isinstance(document, str) or len(document) > _MAX_DOCUMENT:
        raise ValueError("document must be a string of at most 2000000 characters")
    begin, end = f"begin{_GUID}", f"end{_GUID}"
    if document.count(begin) != 1 or document.count(end) != 1:
        raise ValueError("expected exactly one native CRM Suite XML payload")
    xml = document.split(begin, 1)[1].split(end, 1)[0].strip()
    if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
        raise ValueError("XML DTDs and entities are not supported")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise ValueError("invalid native CRM XML") from error
    if root.tag != "CRMSimApp" or root.get("ProgID") != _GUID or root.get("FileVersion") != "1.0.0":
        raise ValueError("only CRM Suite CRMSimApp FileVersion 1.0.0 is supported")
    data = root.find("CRMSimulatorData")
    model = root.find("CRMSimulatorData/Model")
    if data is None or model is None:
        raise ValueError("native study is missing simulation/model settings")
    for node in (root, data, model):
        tags = [child.tag for child in node]
        if len(tags) != len(set(tags)):
            raise ValueError("duplicate native settings are ambiguous")
    doses = _integer(model, "NumDoses", 2, 20)
    count = _integer(model, "NumSkeletons", 1, 5)
    values = [float(_text(v, ".")) for v in model.findall("Probabilities/double")]
    if not count * doses <= len(values) <= 100 or not np.isfinite(values).all():
        raise ValueError("invalid native skeleton probability slots")
    skeletons = tuple(tuple(values[i * doses : (i + 1) * doses]) for i in range(count))
    active = np.asarray(skeletons)
    if np.any((active <= 0) | (active >= 1)) or np.any(np.diff(active, axis=1) < 0):
        raise ValueError("native skeletons must be nondecreasing probabilities in (0,1)")
    flag_root = "DAandSkelsUsed_ForSerializationOnly/"
    enabled = _text(model, flag_root + "DataAugmentationIsEnabled")
    if enabled not in {"true", "false"}:
        raise ValueError("invalid data-augmentation flag")
    flag_name = (
        "SkeletonsBeingUsedWithDA_ForSerializationOnly"
        if enabled == "true"
        else "SkeletonsBeingUsedWithOutDA_ForSerializationOnly"
    )
    flags = [_text(v, ".") for v in model.findall(flag_root + flag_name + "/boolean")]
    if len(flags) != count or any(v not in {"true", "false"} for v in flags):
        raise ValueError("invalid active skeleton flags")
    skeletons = tuple(row for row, flag in zip(skeletons, flags, strict=True) if flag == "true")
    if not skeletons or (enabled == "true" and len(skeletons) != 1):
        raise ValueError("native study must select skeletons, exactly one for DA")
    names, scenarios = [], []
    for scenario in data.findall("Scenarios/CRMScenario"):
        names.append(_text(scenario, "Name"))
        probabilities = tuple(
            float(_text(v, ".")) for v in scenario.findall("Probs/double")[:doses]
        )
        if (
            len(probabilities) != doses
            or not np.isfinite(probabilities).all()
            or any(not 0 <= p <= 1 for p in probabilities)
        ):
            raise ValueError("invalid native scenario probabilities")
        scenarios.append(probabilities)
    if not 1 <= len(scenarios) <= 100:
        raise ValueError("native study requires 1..100 scenarios")
    labels = tuple(_text(v, ".") for v in root.findall("DoseDescriptions/string")[:doses])
    if len(labels) != doses:
        raise ValueError("native study is missing dose descriptions")
    patients = root.find("Patients")
    patient_count = 0 if patients is None else len(patients)
    if patient_count > 200:
        raise ValueError("native study exceeds 200 patients")
    # Reserialization removes comments and escapes XML text before embedding it
    # inside an HTML comment. Original HTML/script content is never copied.
    normalized = ET.tostring(root, encoding="unicode")
    return CRMDesktopStudy(
        normalized,
        skeletons,
        tuple(names),
        tuple(scenarios),
        labels,
        enabled == "true",
        patient_count,
    )


@dataclass(frozen=True)
class CRMDesktopRun:
    """A saved study's scenarios simulated under the documented Python engine."""

    study: CRMDesktopStudy
    simulations: tuple[CRMSimulation, ...]
    draws: int
    warmup: int
    chains: int

    def to_html(self) -> str:
        """Render all scenario summaries and preserve native study settings."""
        body = [
            "<h1>CRM Suite study: Python simulation</h1>",
            "<p>Native desktop calendar profile; independent Python inference and random streams. "
            "Native adaptive MCMC and random-seed identity are not claimed.</p>",
        ]
        if self.study.data_augmentation:
            body.append(
                f"<p>DA sampler: {self.draws} retained draws, "
                f"{self.warmup} warmup, {self.chains} chains.</p>"
            )
        for name, truth, result in zip(
            self.study.scenario_names, self.study.scenarios, self.simulations, strict=True
        ):
            body.append(
                f"<h2>{escape(name)}</h2><table><tr><th>Dose</th><th>True toxicity</th>"
                "<th>Selection</th><th>Selection MCSE</th><th>Mean patients</th>"
                "<th>Mean toxicities</th></tr>"
            )
            for i, label in enumerate(self.study.dose_labels):
                cells = [escape(label)] + [
                    f"{v:.6g}"
                    for v in [
                        truth[i],
                        result.selection_probability[i],
                        result.selection_mcse[i],
                        result.mean_patients[i],
                        result.mean_toxicities[i],
                    ]
                ]
                body.append("<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
            body.append(
                "</table>" + f"<p>Trials: {len(result.selected_dose)}; "
                f"no selection: {result.no_selection_probability:.6g}; "
                f"early stop: {result.early_stop_probability:.6g}; "
                f"mean total toxicities: {np.sum(result.mean_toxicities):.6g}; "
                f"mean duration: {np.mean(result.duration):.6g} days "
                f"({np.mean(result.duration) / 30.4375:.6g} native-report months).</p>"
            )
        return _envelope("\n".join(body), self.study.xml)


def simulate_crm_desktop_study(
    study: CRMDesktopStudy,
    *,
    trials: int,
    seed: int,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 2,
    max_arrival_gaps: int = 2000,
    max_total_evaluations: int = 20_000_000,
) -> CRMDesktopRun:
    """Simulate every saved scenario with explicit workload and random seed.

    Stored native adaptive-MCMC tuning is preserved but not silently interpreted
    as Python sampler tuning. Patient records are retained; this is a prospective
    design simulation, not a reconstruction of an already-conducted trial.
    """
    if not isinstance(study, CRMDesktopStudy):
        raise TypeError("study must be a CRMDesktopStudy")
    # Revalidate publicly constructible dataclasses before scientific execution.
    study = read_crm_desktop_html(study.to_html())
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 10_000:
        raise ValueError("trials must be an integer in [1,10000]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer in [0,2147483647]")
    if (
        isinstance(max_total_evaluations, bool)
        or not isinstance(max_total_evaluations, int)
        or not 1 <= max_total_evaluations <= 200_000_000
    ):
        raise ValueError("max_total_evaluations must be an integer in [1,200000000]")
    root = ET.fromstring(study.xml)
    data = root.find("CRMSimulatorData")
    model = root.find("CRMSimulatorData/Model")
    assert data is not None and model is not None
    for key, expected in [
        ("PriorsAre_P", "1"),
        ("DoseEscalationRule", "eStrict"),
        ("MinPatientsToDeclareMTD", "3"),
    ]:
        if _text(model, key) != expected:
            raise ValueError(f"unsupported native {key}; refusing to substitute another policy")
    size = _integer(model, "CohortSize", 1, 4)
    patients = _integer(model, "MaxPatients", 1, 200)
    if patients % size:
        raise ValueError("native MaxPatients must be divisible by CohortSize")
    duration = float(_integer(model, "EvalTime", 1, 2_147_483_647))
    gap = _number(data, "MeanInterArrivalTimeSpan")
    if gap <= 0 or duration <= 0:
        raise ValueError("native duration and mean arrival gap must be positive")
    prior = None
    minimum = None
    if study.data_augmentation:
        trimester = [
            float(_text(v, "."))
            for v in model.findall("ToxicityOccurrencePriorProbabilities/double")
        ]
        prior = dacrm_trimester_prior(
            duration, trimester, dispersion=2, endpoint_policy="native_desktop"
        )
        minimum = _integer(model, "MinimumNumberRequiredToEscalate", 0, 200)
    # Keep the saved workload consistent with this report; other XML fields,
    # including unused probability slots and patient text, remain unchanged.
    for key, value in [("NumRepetitions", trials), ("Seed", seed)]:
        node = data.find(key)
        if node is None:
            raise ValueError(f"native study is missing {key}")
        node.text = str(value)
    study = read_crm_desktop_html(_envelope("", ET.tostring(root, encoding="unicode")))
    sequences = np.random.SeedSequence(seed).spawn(2 * len(study.scenarios))
    results = []
    remaining = max_total_evaluations
    for i, truth in enumerate(study.scenarios):
        if remaining <= 0:
            raise RuntimeError("desktop study exhausted max_total_evaluations")
        result = simulate_crm(
            study.skeletons,
            truth,
            duration,
            1 / gap,
            target=_number(model, "TargetToxicity"),
            cohorts=patients // size,
            cohort_size=size,
            trials=trials,
            starting_dose=_integer(model, "StartingDoseIndex", 0, len(truth) - 1),
            method="dacrm" if prior is not None else "bmacrm",
            da_prior=prior,
            minimum_observed=minimum,
            safety_cutoff=_number(model, "UpperLimitOnProbToxAtLowestDose"),
            late_probability=1 - _number(data, "FractToxFirstHalfT"),
            scheduling="native_desktop",
            max_arrival_gaps=max_arrival_gaps,
            rng=np.random.default_rng(sequences[2 * i]),
            sampler_rng=np.random.default_rng(sequences[2 * i + 1]) if prior is not None else None,
            max_total_evaluations=remaining,
            draws=draws,
            warmup=warmup,
            chains=chains,
        )
        results.append(result)
        remaining -= result.evaluations
    return CRMDesktopRun(study, tuple(results), draws, warmup, chains)
