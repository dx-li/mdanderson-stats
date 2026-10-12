"""Versioned Python WFMM studies, qualified execution and saved summaries."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from .hierarchical_binomial import summarize_chains
from .wfmm_basis import WFMMBasis, _setting, wfmm_basis, wfmm_transform
from .wfmm_compression import TiePolicy, WFMMCompression, wfmm_compress_coefficients
from .wfmm_covariance import WFMMCovarianceSummary, wfmm_summarize_covariance
from .wfmm_initialization import WFMMInitialization, initialize_wfmm
from .wfmm_model import WFMMCoefficientFit, _strata, fit_wfmm_coefficients
from .wfmm_pca import wfmm_pca_basis
from .wfmm_posterior import WFMMPosteriorSummary, wfmm_summarize
from .wfmm_prediction import wfmm_predict_coefficients
from .wfmm_selection import wfmm_pass_filter, wfmm_restore_coefficients

_MAX_JSON_BYTES = 16_000_000


def _matrix(value, name, rows=None, columns=None):
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.dtype.kind not in "iuf"
        or raw.size == 0
        or raw.size > 2_000_000
        or (rows is not None and raw.shape[0] != rows)
        or (columns is not None and raw.shape[1] != columns)
        or not np.all(np.isfinite(raw))
    ):
        raise ValueError(f"{name} must be a bounded finite real matrix with matching dimensions")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} is not representable as float64")
    return _freeze(converted)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_value(x) for x in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class WFMMStudy:
    """Saved observed curves and explicit controls for the stable Python workflow.

    The JSON schema is not a native MAT-file format. Initializer quality holds
    stop execution before MCMC; there is no automatic compatibility fallback.
    Predictions here are fixed-effect population curves, optionally with new
    residual noise. Existing/new random-level prediction remains available via
    the returned coefficient fit and wfmm_predict_coefficients.
    """

    curves: ArrayLike
    fixed_design: ArrayLike
    random_design: ArrayLike | None = None
    random_strata: ArrayLike | None = None
    residual_strata: ArrayLike | None = None
    transform: str = "wavelet"
    analysis_matrix: ArrayLike | None = None
    synthesis_matrix: ArrayLike | None = None
    levels: int = 1
    filter_length: int = 2
    alpha_pc: float = 0.99
    pc_partition_base: float = 10.0
    highpass: int = 0
    lowpass: int = 0
    alpha_energy: float = 1.0
    compression_votes: int = 0
    compression_ties: TiePolicy = "reject_split"
    draws: int = 1000
    warmup: int = 500
    chains: int = 2
    seed: int = 0
    min_variance: float = 1e-6
    delta_omega: float = 1e-4
    proposal_multiplier: float = 1.5
    moment_iterations: int = 100
    optimizer_iterations: int = 200
    empirical_bayes_iterations: int = 1000
    prediction_design: ArrayLike | None = None
    prediction_residual: bool = False
    prediction_strata: ArrayLike | None = None

    def __post_init__(self):
        y = _matrix(self.curves, "curves")
        n, t = y.shape
        if not 2 <= n <= 500 or not 2 <= t <= 4096:
            raise ValueError("study curve dimensions exceed supported bounds")
        x = _matrix(self.fixed_design, "fixed_design", rows=n)
        if not 1 <= x.shape[1] <= min(n, 100):
            raise ValueError("fixed design has an unsupported column count")
        z = (
            None
            if self.random_design is None
            else _matrix(self.random_design, "random_design", rows=n)
        )
        if z is not None and z.shape[1] > 500:
            raise ValueError("random design exceeds 500 columns")
        if z is None and self.random_strata is not None:
            raise ValueError("random_strata requires random_design")
        object.__setattr__(self, "curves", y)
        object.__setattr__(self, "fixed_design", x)
        object.__setattr__(self, "random_design", z)
        if z is not None:
            labels = _strata(self.random_strata, z.shape[1], "random_strata")
            object.__setattr__(self, "random_strata", tuple(int(i) for i in labels))
        labels = _strata(self.residual_strata, n, "residual_strata")
        object.__setattr__(self, "residual_strata", tuple(int(i) for i in labels))
        if self.transform not in ("identity", "wavelet", "custom", "PC", "PCw", "wPC"):
            raise ValueError("unsupported study transform")
        if self.transform == "custom":
            for name in ("analysis_matrix", "synthesis_matrix"):
                value = _matrix(getattr(self, name), name, rows=t, columns=t)
                object.__setattr__(self, name, value)
        elif self.analysis_matrix is not None or self.synthesis_matrix is not None:
            raise ValueError("matrix pairs require transform=custom")
        for name, low, high in (
            ("levels", 1, 12),
            ("filter_length", 2, 20),
            ("highpass", 0, 12),
            ("lowpass", 0, 12),
            ("compression_votes", 0, n),
            ("draws", 8, 5000),
            ("warmup", 0, 5000),
            ("chains", 2, 4),
            ("seed", 0, 2**63 - 1),
            ("moment_iterations", 1, 5000),
            ("optimizer_iterations", 1, 5000),
            ("empirical_bayes_iterations", 1, 5000),
        ):
            object.__setattr__(self, name, _setting(getattr(self, name), name, low, high))
        if self.transform in ("wavelet", "PCw", "wPC") and t % (2**self.levels):
            raise ValueError("2**levels must divide the number of time points")
        if self.filter_length % 2:
            raise ValueError("filter_length must be even")
        if self.transform != "wavelet" and (self.highpass or self.lowpass):
            raise ValueError("saved pass flags currently require the wavelet transform")
        if self.compression_ties not in ("reject_split", "original_index"):
            raise ValueError("unsupported compression tie policy")
        for name, lo, hi, strict in (
            ("alpha_pc", 0, 1, False),
            ("alpha_energy", 0, 1, True),
            ("pc_partition_base", 1, 1e100, True),
            ("min_variance", 0, 1e100, True),
            ("delta_omega", 0, 1e100, True),
            ("proposal_multiplier", 0, 1e100, True),
        ):
            raw = np.asarray(getattr(self, name))
            if (
                raw.ndim != 0
                or raw.dtype.kind not in "iuf"
                or not np.isfinite(raw)
                or float(raw) > hi
                or (float(raw) <= lo if strict else float(raw) < lo)
            ):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, float(raw))
        if type(self.prediction_residual) is not bool:
            raise ValueError("prediction_residual must be boolean")
        if self.prediction_design is None:
            if self.prediction_strata is not None or self.prediction_residual:
                raise ValueError("prediction controls require prediction_design")
        else:
            design = _matrix(self.prediction_design, "prediction_design", columns=x.shape[1])
            if design.shape[0] > 500:
                raise ValueError("prediction design exceeds 500 rows")
            object.__setattr__(self, "prediction_design", design)
            if self.prediction_residual:
                lab = np.asarray(self.prediction_strata)
                groups = max(labels) + 1
                if (
                    lab.shape != (design.shape[0],)
                    or lab.dtype.kind not in "iu"
                    or np.any(lab < 0)
                    or np.any(lab >= groups)
                ):
                    raise ValueError("prediction_strata must index fitted residual strata")
                object.__setattr__(self, "prediction_strata", tuple(int(i) for i in lab))
            elif self.prediction_strata is not None:
                raise ValueError("prediction_strata is only used with residual prediction")

    def to_json(self) -> str:
        data = {f.name: _json_value(getattr(self, f.name)) for f in fields(self)}
        data.update(schema="mdanderson_stats.wfmm.study", version=1)
        text = json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        if len(text.encode()) > _MAX_JSON_BYTES:
            raise ValueError("saved study exceeds 16 MB")
        return text


def read_wfmm_study(text: str) -> WFMMStudy:
    """Read complete version1 JSON with duplicate/unknown-field rejection."""
    if not isinstance(text, str) or len(text.encode()) > _MAX_JSON_BYTES:
        raise ValueError("study JSON must be text no larger than 16 MB")

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON field")
            out[key] = value
        return out

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        data = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
        if (
            not isinstance(data, dict)
            or data.pop("schema", None) != "mdanderson_stats.wfmm.study"
            or type(data.get("version")) is not int
            or data.pop("version") != 1
            or set(data) != {f.name for f in fields(WFMMStudy)}
        ):
            raise ValueError("unsupported WFMM study schema/version/fields")
        return WFMMStudy(**data)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid WFMM study JSON") from exc


@dataclass(frozen=True)
class WFMMStudyResult:
    study: WFMMStudy
    basis: WFMMBasis
    compression: WFMMCompression
    initialization: WFMMInitialization
    fit: WFMMCoefficientFit | None
    summary: WFMMPosteriorSummary | None
    covariance: WFMMCovarianceSummary | None
    prediction: WFMMPosteriorSummary | None

    def report_json(self) -> str:
        """Save summaries and diagnostics; null diagnostic values are explicitly unavailable."""
        data = dict(
            schema="mdanderson_stats.wfmm.report",
            version=1,
            study_sha256=hashlib.sha256(self.study.to_json().encode()).hexdigest(),
            status="held" if self.fit is None else "sampled",
            initializer_holds=self.initialization.holds,
            rng="NumPy PCG64; SeedSequence.spawn(2): fit, prediction",
            seed=self.study.seed,
            numpy_version=np.__version__,
            retained_indices=self.compression.selection.retained_indices,
            compression_energy_fraction=self.compression.energy_fraction,
            posterior_precision="not_certified",
            clinical_validity="not_established",
            null_diagnostic_meaning="unavailable, including constant-chain R-hat",
        )
        if self.fit is not None:
            fit = self.fit
            variance = np.concatenate((fit.random_variances, fit.residual_variances), axis=2)
            variance_summary = summarize_chains(variance)
            data.update(
                coefficient_rhat=fit.summary.split_rhat,
                coefficient_mcse=fit.summary.batch_mean_mcse,
                variance_rhat=variance_summary.split_rhat,
                variance_mcse=variance_summary.batch_mean_mcse,
                variance_acceptance=fit.variance_acceptance,
                variance_proposals=fit.variance_proposals,
                likelihood_evaluations=fit.likelihood_evaluations,
            )
            for name, summary in (("curves", self.summary), ("prediction", self.prediction)):
                if summary is not None:
                    data[name] = dict(
                        mean=summary.mean,
                        sd=summary.standard_deviation,
                        quantiles=summary.quantiles,
                        quantile_probabilities=summary.quantile_probabilities,
                        simultaneous_lower=summary.simultaneous_lower,
                        simultaneous_upper=summary.simultaneous_upper,
                        simbas=summary.simbas_probability,
                    )
            if self.covariance is not None:
                data["variance_functions"] = dict(
                    mean=self.covariance.variance_function_mean,
                    sd=self.covariance.variance_function_standard_deviation,
                )
        text = json.dumps(_json_value(data), sort_keys=True, allow_nan=False) + "\n"
        if len(text.encode()) > _MAX_JSON_BYTES:
            raise ValueError("report exceeds 16 MB")
        return text

    def curve_csv(self) -> str:
        if self.summary is None:
            raise ValueError("held studies have no posterior curve report")
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            ("effect", "time_index", "mean", "sd", "simultaneous_lower", "simultaneous_upper")
        )
        for effect, row in enumerate(self.summary.mean):
            for time, value in enumerate(row):
                writer.writerow(
                    (
                        effect,
                        time,
                        float(value),
                        float(self.summary.standard_deviation[effect, time]),
                        float(self.summary.simultaneous_lower[effect, time]),
                        float(self.summary.simultaneous_upper[effect, time]),
                    )
                )
        return output.getvalue()


def run_wfmm_study(study: WFMMStudy) -> WFMMStudyResult:
    """Transform, initialize, sample and summarize, stopping on initializer holds."""
    if not isinstance(study, WFMMStudy):
        raise ValueError("study must be a WFMMStudy")
    t = np.asarray(study.curves).shape[1]
    if study.transform == "identity":
        basis = wfmm_basis(t, transform="identity")
    elif study.transform == "custom":
        basis = wfmm_basis(
            t,
            transform="custom",
            analysis_matrix=study.analysis_matrix,
            synthesis_matrix=study.synthesis_matrix,
        )
    elif study.transform == "wavelet":
        basis = wfmm_basis(t, levels=study.levels, filter_length=study.filter_length)
    else:
        wave = (
            None
            if study.transform == "PC"
            else wfmm_basis(t, levels=study.levels, filter_length=study.filter_length)
        )
        basis = wfmm_pca_basis(
            study.curves,
            kind=study.transform,
            alpha=study.alpha_pc,
            partition_base=study.pc_partition_base,
            wavelet_basis=wave,
        ).basis
    transformed = wfmm_transform(study.curves, basis).coefficients
    eligible = None
    if study.transform == "wavelet":
        eligible = wfmm_pass_filter(
            transformed, basis, highpass=study.highpass, lowpass=study.lowpass
        ).retained_indices
    compression = wfmm_compress_coefficients(
        transformed,
        basis,
        alpha=study.alpha_energy,
        t=study.compression_votes,
        tie_policy=study.compression_ties,
        eligible_indices=eligible,
    )
    selection = compression.selection
    initialization = initialize_wfmm(
        selection.coefficients,
        study.fixed_design,
        study.random_design,
        random_strata=study.random_strata,
        residual_strata=study.residual_strata,
        coefficient_partition=selection.coefficient_partition,
        min_variance=study.min_variance,
        delta_omega=study.delta_omega,
        proposal_multiplier=study.proposal_multiplier,
        moment_iterations=study.moment_iterations,
        optimizer_iterations=study.optimizer_iterations,
        empirical_bayes_iterations=study.empirical_bayes_iterations,
    )
    if not initialization.ready_for_sampling:
        return WFMMStudyResult(study, basis, compression, initialization, None, None, None, None)
    # Zero restoration and curve summaries have bounds separate from sampler K.
    restored_cells = (
        study.chains
        * study.draws
        * t
        * max(
            np.asarray(study.fixed_design).shape[1],
            initialization.conditional.proposal_sd.shape[0],
            0 if study.prediction_design is None else np.asarray(study.prediction_design).shape[0],
        )
    )
    if restored_cells > 2_000_000:
        raise ValueError("study restored draws exceed the 2000000-cell limit")
    state = initialization.conditional
    h = state.random_variance.shape[0]
    seeds = np.random.SeedSequence(study.seed).spawn(2)
    fit = fit_wfmm_coefficients(
        selection.coefficients,
        study.fixed_design,
        study.random_design,
        prior=initialization.prior,
        random_variance=state.random_variance,
        residual_variance=state.residual_variance,
        random_strata=study.random_strata,
        residual_strata=study.residual_strata,
        coefficient_partition=selection.coefficient_partition,
        coefficient_scale=selection.coefficient_scale,
        proposal_sd=(state.proposal_sd[:h], state.proposal_sd[h:]),
        draws=study.draws,
        warmup=study.warmup,
        chains=study.chains,
        sample_random_effects=study.random_design is not None,
        rng=np.random.Generator(np.random.PCG64(seeds[0])),
    )
    summary = wfmm_summarize(wfmm_restore_coefficients(fit.coefficients, selection), basis)
    variances = np.concatenate((fit.random_variances, fit.residual_variances), axis=2)
    covariance = wfmm_summarize_covariance(wfmm_restore_coefficients(variances, selection), basis)
    prediction = None
    if study.prediction_design is not None:
        predicted = wfmm_predict_coefficients(
            fit,
            study.prediction_design,
            include_residual=study.prediction_residual,
            residual_strata=study.prediction_strata,
            rng=np.random.Generator(np.random.PCG64(seeds[1])),
        )
        prediction = wfmm_summarize(wfmm_restore_coefficients(predicted, selection), basis)
    return WFMMStudyResult(
        study, basis, compression, initialization, fit, summary, covariance, prediction
    )
