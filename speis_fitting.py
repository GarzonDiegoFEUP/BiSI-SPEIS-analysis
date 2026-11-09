"""Auditable SPEIS grouping, R-(R||CPE) fitting, and linear analysis.

No input points are silently filtered, no spectra are merged, and no circuit
capacitance is asserted to be a depletion capacitance. Voltages are the input
device bias, not a reference-electrode or work-function corrected potential.
All capacitances are total capacitance in F, resistances in ohm, and frequencies
in Hz. Covariances are local, residual-scaled approximations, not confidence
guarantees; correlations between measurements are not modelled.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.constants import elementary_charge, epsilon_0
from scipy.optimize import least_squares


@dataclass
class Spectrum:
    """One contiguous sweep, with all original rows and audit metadata."""

    spectrum_id: int
    data: pd.DataFrame
    cycle: object
    voltage_v: float
    voltage_spread_v: float
    frequency_min_hz: float
    frequency_max_hz: float
    n_distinct_frequencies: int
    flags: list[str] = field(default_factory=list)


def group_spectra(eis: pd.DataFrame, *, reset_ratio: float = 2.0,
                  direction: str = "descending", voltage_spread_limit: float = 0.01,
                  same_bias_tolerance: float = 0.001,
                  expected_points: Optional[int] = None,
                  expected_frequency_range: Optional[tuple[float, float]] = None,
                  frequency_tolerance: float = 0.05) -> list[Spectrum]:
    """Split contiguous rows on cycle changes or frequency resets, never on bias.

    Rows are sorted by a unique, finite numeric ``_row`` if supplied, otherwise
    positional row numbers are added. Invalid rows remain in their sweep and are
    flagged. Completeness is only testable with caller-supplied expected points
    and/or (minimum, maximum) frequency limits; otherwise it is unverified.
    ``reset_ratio`` detects a reversal by that factor (choose sweep direction).
    Close median biases are flagged, not combined. Tolerances are in V or relative
    frequency, respectively. All flags require user review, not automatic removal.
    """
    required = ["freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm", "<Ewe>/V"]
    missing = [c for c in required if c not in eis]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not np.isfinite(reset_ratio) or reset_ratio <= 1:
        raise ValueError("reset_ratio must be finite and > 1")
    if direction not in ("ascending", "descending"):
        raise ValueError("direction must be ascending or descending")
    if (not np.isfinite(voltage_spread_limit) or voltage_spread_limit < 0 or
            not np.isfinite(same_bias_tolerance) or same_bias_tolerance < 0 or
            not np.isfinite(frequency_tolerance) or not 0 <= frequency_tolerance < 1):
        raise ValueError("Invalid grouping tolerance")
    if expected_points is not None and (not isinstance(expected_points, int) or expected_points < 1):
        raise ValueError("expected_points must be a positive integer")
    if expected_frequency_range is not None:
        lo, hi = expected_frequency_range
        if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
            raise ValueError("expected_frequency_range must be positive (min, max)")
    df = eis.copy()
    if "_row" not in df:
        df["_row"] = np.arange(len(df))
    rows = pd.to_numeric(df["_row"], errors="coerce")
    if rows.isna().any() or not np.isfinite(rows).all() or rows.duplicated().any():
        raise ValueError("_row must contain unique finite numeric values")
    df = df.assign(_sort_row=rows).sort_values("_sort_row", kind="stable").drop(columns="_sort_row").reset_index(drop=True)
    if df.empty:
        return []
    freq = pd.to_numeric(df["freq/Hz"], errors="coerce").to_numpy(dtype=float)
    cycle = df["cycle number"].to_numpy() if "cycle number" in df else None
    starts = [0]
    previous_f = None
    for i, f in enumerate(freq):
        changed = i > 0 and cycle is not None and not (
            (pd.isna(cycle[i]) and pd.isna(cycle[i - 1])) or
            (not pd.isna(cycle[i]) and not pd.isna(cycle[i - 1]) and cycle[i] == cycle[i - 1]))
        valid_f = np.isfinite(f) and f > 0
        reset = (previous_f is not None and valid_f and
                 (f / previous_f >= reset_ratio if direction == "descending"
                  else previous_f / f >= reset_ratio))
        if i and (changed or reset):
            starts.append(i)
            previous_f = None
        if valid_f:
            previous_f = f
    starts.append(len(df))
    spectra = []
    for number, (start, end) in enumerate(zip(starts[:-1], starts[1:]), 1):
        part = df.iloc[start:end].copy()
        f = pd.to_numeric(part["freq/Hz"], errors="coerce").to_numpy(dtype=float)
        re = pd.to_numeric(part["Re(Z)/Ohm"], errors="coerce").to_numpy(dtype=float)
        im = pd.to_numeric(part["-Im(Z)/Ohm"], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(part["<Ewe>/V"], errors="coerce").to_numpy(dtype=float)
        good_f = f[np.isfinite(f) & (f > 0)]
        good_v = v[np.isfinite(v)]
        flags = []
        if len(good_f) != len(f) or not np.isfinite(re).all() or not np.isfinite(im).all():
            flags.append("invalid_impedance_or_frequency")
        if len(good_v) != len(v):
            flags.append("invalid_voltage")
        median = float(np.median(good_v)) if len(good_v) else np.nan
        spread = float(np.ptp(good_v)) if len(good_v) else np.nan
        if np.isfinite(spread) and spread > voltage_spread_limit:
            flags.append("voltage_spread")
        n_distinct = len(np.unique(good_f))
        if n_distinct < 6:
            flags.append("too_few_distinct_frequencies")
        if len(good_f) > 1:
            differences = np.diff(good_f)
            if np.any(differences > 0 if direction == "descending" else differences < 0):
                flags.append("frequency_reversal")
        fmin = float(np.min(good_f)) if len(good_f) else np.nan
        fmax = float(np.max(good_f)) if len(good_f) else np.nan
        if expected_points is None and expected_frequency_range is None:
            flags.append("completeness_unverified")
        if expected_points is not None and len(part) < expected_points:
            flags.append("incomplete_sweep")
        if expected_frequency_range is not None and (not len(good_f) or
                fmin > expected_frequency_range[0] * (1 + frequency_tolerance) or
                fmax < expected_frequency_range[1] * (1 - frequency_tolerance)):
            if "incomplete_sweep" not in flags:
                flags.append("incomplete_sweep")
        c = part["cycle number"].iloc[0] if cycle is not None else None
        if cycle is not None and pd.isna(c):
            flags.append("missing_cycle")
        spectra.append(Spectrum(number, part, c, median, spread, fmin, fmax, n_distinct, flags))
    for i, spectrum in enumerate(spectra):
        if np.isfinite(spectrum.voltage_v) and any(
                i != j and np.isfinite(other.voltage_v) and
                abs(spectrum.voltage_v - other.voltage_v) <= same_bias_tolerance
                for j, other in enumerate(spectra)):
            spectrum.flags.append("possible_repeated_bias")
    return spectra


def impedance_rcpe(frequency_hz, rs: float, rp: float, q: float, alpha: float):
    """Return Rs + (Rp || CPE) as complex ohms, avoiding large powers."""
    f = np.asarray(frequency_hz, dtype=float)
    if not np.isfinite(f).all() or np.any(f <= 0):
        raise ValueError("Frequencies must be finite and positive")
    if not np.isfinite([rs, rp, q, alpha]).all() or rs <= 0 or rp <= 0 or q <= 0 or not 0 < alpha <= 1:
        raise ValueError("Require finite Rs, Rp, Q > 0 and 0 < alpha <= 1")
    t = np.log(rp) + np.log(q) + alpha * (np.log(2 * np.pi) + np.log(f))
    phase = np.exp(0.5j * np.pi * alpha)
    x = np.exp(-np.abs(t))
    branch = np.empty_like(x, dtype=complex)
    positive = t >= 0
    branch[positive] = rp * (x[positive] / (x[positive] + phase))
    branch[~positive] = rp / (1 + x[~positive] * phase)
    return rs + branch


@dataclass
class FitResult:
    success: bool
    parameters: Optional[np.ndarray] = None  # Rs, Rp, Q, alpha
    covariance: Optional[np.ndarray] = None  # same physical parameter order
    capacitance_f: Optional[float] = None
    capacitance_std_f: Optional[float] = None
    raw_rmse_ohm: Optional[float] = None  # complex magnitude RMS
    relative_rmse: Optional[float] = None  # modulus-weighted complex RMS
    modulus_floor_ohm: Optional[float] = None
    jacobian_condition: Optional[float] = None
    frequency_range_hz: Optional[tuple[float, float]] = None
    used_rows: tuple = ()
    excluded_rows: tuple = ()
    flags: list[str] = field(default_factory=list)
    message: str = ""
    weighting: str = "modulus"


def effective_capacitance(rp: float, q: float, alpha: float,
                          covariance: Optional[np.ndarray] = None) -> tuple[float, Optional[float]]:
    """Compute branch effective C and optional delta-method sigma from 4x4 covariance.

    Covariance parameter order is (Rs, Rp, Q, alpha). No uncertainty is returned
    without a finite symmetric positive-semidefinite covariance.
    """
    if not np.isfinite([rp, q, alpha]).all() or rp <= 0 or q <= 0 or not 0 < alpha <= 1:
        raise ValueError("Require finite Rp, Q > 0 and 0 < alpha <= 1")
    logc = (np.log(q) + (1 - alpha) * np.log(rp)) / alpha
    if not np.log(np.finfo(float).tiny) <= logc <= np.log(np.finfo(float).max):
        raise ValueError("Effective capacitance is outside representable range")
    c = float(np.exp(logc))
    if covariance is None:
        return c, None
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (4, 4) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T, rtol=1e-10, atol=0):
        raise ValueError("covariance must be a finite symmetric 4x4 matrix")
    # Check PSD in correlation-scaled coordinates, not in mixed physical units.
    diagonal = np.diag(cov)
    if np.any(diagonal < 0) or np.any((diagonal == 0) & np.any(cov != 0, axis=1)):
        raise ValueError("covariance must be positive semidefinite")
    scale = np.sqrt(np.where(diagonal > 0, diagonal, 1))
    corr = cov / np.outer(scale, scale)
    if np.linalg.eigvalsh(corr).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    grad = np.array([0, (1 - alpha) / (alpha * rp), 1 / (alpha * q),
                     -(np.log(q) + np.log(rp)) / alpha**2])
    variance = float(grad @ cov @ grad)
    if not np.isfinite(variance):
        raise ValueError("Capacitance uncertainty is outside representable range")
    std = float(c * np.sqrt(max(variance, 0)))
    if not np.isfinite(std):
        raise ValueError("Capacitance uncertainty is outside representable range")
    return c, std


def fit_spectrum(spectrum: Spectrum, *, mask=None,
                 bounds=((1e-12, 1e-12, 1e-30, 0.05), (1e15, 1e15, 1e5, 1.0)),
                 modulus_floor_fraction: float = 1e-12, max_relative_rmse: float = 0.1,
                 max_condition: float = 1e8, bound_tolerance: float = 1e-4,
                 max_nfev: int = 5000) -> FitResult:
    """Fit selected rows with bounded, deterministic multi-start CNLS.

    ``mask`` is a positional boolean array: excluded rows are recorded. Invalid
    selected data fail rather than being dropped. Residuals are real and imaginary
    differences divided by max(|measured Z|, floor_fraction * median |Z|).
    Covariance uses the residual-scaled Jacobian in log(Rs,Rp,Q), alpha coordinates
    and is suppressed for active/near bounds or poor conditioning. Thresholds are
    diagnostics, not permission to interpret a circuit as depletion capacitance.
    """
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    if (lower.shape != (4,) or upper.shape != (4,) or not np.isfinite(lower).all() or
            not np.isfinite(upper).all() or np.any(lower <= 0) or np.any(lower >= upper) or upper[3] > 1):
        raise ValueError("Invalid positive physical bounds; alpha upper bound must be <= 1")
    if (not np.isfinite(modulus_floor_fraction) or modulus_floor_fraction <= 0 or
            not np.isfinite(max_relative_rmse) or max_relative_rmse < 0 or
            not np.isfinite(max_condition) or max_condition <= 1 or
            not np.isfinite(bound_tolerance) or not 0 <= bound_tolerance < 0.5 or max_nfev < 1):
        raise ValueError("Invalid fitting configuration")
    df = spectrum.data
    if mask is None:
        selected = np.ones(len(df), dtype=bool)
    else:
        selected = np.asarray(mask)
        if selected.shape != (len(df),) or selected.dtype.kind != "b":
            raise ValueError("mask must be a positional boolean array matching the spectrum")
    used = tuple(df.loc[selected, "_row"])
    excluded = tuple(df.loc[~selected, "_row"])
    flags = list(spectrum.flags)
    if excluded:
        flags.append("user_excluded_points")
    result = FitResult(False, used_rows=used, excluded_rows=excluded, flags=flags)
    part = df.loc[selected]
    f = pd.to_numeric(part["freq/Hz"], errors="coerce").to_numpy(dtype=float)
    re = pd.to_numeric(part["Re(Z)/Ohm"], errors="coerce").to_numpy(dtype=float)
    im = pd.to_numeric(part["-Im(Z)/Ohm"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(f).all() or np.any(f <= 0) or not np.isfinite(re).all() or not np.isfinite(im).all():
        result.flags.append("invalid_selected_data")
        result.message = "Selected rows contain invalid frequencies or impedance; provide an explicit mask."
        return result
    if len(np.unique(f)) < 6:
        result.flags.append("too_few_distinct_frequencies")
        result.message = "At least six distinct frequencies are required."
        return result
    z = re - 1j * im
    magnitude = np.abs(z)
    median = float(np.median(magnitude))
    if not np.isfinite(magnitude).all() or not np.isfinite(median) or median <= 0:
        result.flags.append("invalid_weight_scale")
        result.message = "The impedance modulus median must be finite and positive."
        return result
    floor = max(modulus_floor_fraction * median, np.finfo(float).tiny)
    if not np.isfinite(floor):
        result.flags.append("invalid_weight_scale")
        result.message = "The modulus weight floor is outside representable range."
        return result
    result.modulus_floor_ohm = floor
    weights = np.maximum(magnitude, floor)
    result.frequency_range_hz = (float(f.min()), float(f.max()))
    lb = np.r_[np.log(lower[:3]), lower[3]]
    ub = np.r_[np.log(upper[:3]), upper[3]]

    def residual(p):
        prediction = impedance_rcpe(f, *np.r_[np.exp(p[:3]), p[3]])
        diff = (prediction - z) / weights
        return np.r_[diff.real, diff.imag]

    rs0 = max(float(re[np.argmax(f)]), lower[0])
    rp0 = max(float(np.ptp(re)), median * 0.1, lower[1])
    peak_f = float(f[np.argmax(im)])
    best = None
    for a0 in (0.6, 0.85, 0.98):
        a = float(np.clip(a0, lower[3], upper[3]))
        for factor in (0.3, 1.0, 3.0):
            # Compute Q guesses in log space to avoid overflowing powers.
            log_rp = np.log(rp0) + np.log(factor)
            guess = np.array([np.log(rs0), log_rp,
                              -log_rp - a * (np.log(2 * np.pi) + np.log(peak_f)), a])
            guess = np.clip(guess, lb + (ub - lb) * 1e-8, ub - (ub - lb) * 1e-8)
            try:
                fit = least_squares(residual, guess, bounds=(lb, ub), max_nfev=max_nfev)
            except (ValueError, FloatingPointError, OverflowError):
                continue
            if fit.success and np.isfinite(fit.cost) and (best is None or fit.cost < best.cost):
                best = fit
    if best is None:
        result.flags.append("fit_failed")
        result.message = "All optimization starts failed or did not converge."
        return result
    result.success = True
    result.message = best.message
    result.parameters = np.r_[np.exp(best.x[:3]), best.x[3]]
    diff = impedance_rcpe(f, *result.parameters) - z
    result.raw_rmse_ohm = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
    result.relative_rmse = float(np.sqrt(np.mean(np.abs(diff / weights) ** 2)))
    if result.relative_rmse > max_relative_rmse:
        result.flags.append("poor_relative_fit")
    near_bound = np.any((best.x - lb) / (ub - lb) <= bound_tolerance) or np.any((ub - best.x) / (ub - lb) <= bound_tolerance)
    if near_bound or np.any(best.active_mask):
        result.flags.append("parameter_at_bound")
    singular = np.linalg.svd(best.jac, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
    result.jacobian_condition = condition
    if not np.isfinite(condition) or condition > max_condition or np.linalg.matrix_rank(best.jac) < 4:
        result.flags.append("ill_conditioned")
    if "parameter_at_bound" not in result.flags and "ill_conditioned" not in result.flags:
        # SVD avoids squaring the condition number in a normal-equation inverse.
        _, s, vt = np.linalg.svd(best.jac, full_matrices=False)
        cov_log = ((vt.T / s**2) @ vt) * (2 * best.cost / (2 * len(f) - 4))
        transform = np.diag(np.r_[result.parameters[:3], 1.0])
        cov = transform @ cov_log @ transform
        if np.isfinite(cov).all():
            result.covariance = (cov + cov.T) / 2
    if result.covariance is None:
        result.flags.append("uncertainty_unavailable")
    try:
        result.capacitance_f, result.capacitance_std_f = effective_capacitance(
            result.parameters[1], result.parameters[2], result.parameters[3], result.covariance)
    except ValueError:
        result.flags.append("capacitance_or_uncertainty_unavailable")
        try:
            result.capacitance_f, _ = effective_capacitance(*result.parameters[1:])
        except ValueError:
            pass
    return result


def inverse_capacitance_squared(capacitance_f, std_f=None):
    """Return 1/C² in F^-2 and, if supplied, its delta-method standard error.

    ``std_f`` is pointwise sigma in F and must match the capacitance shape. This
    describes marginal uncertainties only; it assumes no inter-point covariance.
    It does not imply the downstream unweighted regression uses these errors.
    """
    c = np.asarray(capacitance_f, dtype=float)
    if not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("Capacitance must be finite and positive")
    with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
        y = (1 / c) ** 2
    if not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("Inverse squared capacitance outside representable range")
    if std_f is None:
        return y, None
    std = np.asarray(std_f, dtype=float)
    if std.shape != c.shape or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError("Standard errors must be finite, nonnegative, and match capacitance")
    with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
        ystd = (2 * (std / c)) * y
    if not np.isfinite(ystd).all():
        raise ValueError("Inverse capacitance uncertainty outside representable range")
    return y, ystd


@dataclass
class LinearResult:
    slope: float  # F^-2 V^-1
    intercept: float  # F^-2
    covariance: np.ndarray  # slope, intercept; residual-scaled unweighted OLS
    r_squared: float  # NaN if constant y
    rmse: float  # F^-2
    x_intercept_v: Optional[float]
    x_intercept_std_v: Optional[float]
    n_points: int
    voltage_range_v: tuple[float, float]
    flags: list[str] = field(default_factory=list)
    weighting: str = "unweighted"


def linear_regression(voltage_v, inverse_capacitance_squared, *, min_abs_slope: float = 0.0) -> LinearResult:
    """Unweighted OLS in SI units; do not identify its intercept as flat-band voltage.

    Requires >=4 points and distinct voltages. All inputs must be finite. Uses
    centered/scaled coordinates and residual-estimated full slope/intercept
    covariance. ``min_abs_slope`` has units F^-2/V; numerical resolution also
    limits near-zero slopes. Constant response has undefined R² and intercept.
    Repeated biases remain independent points and are flagged, never averaged.
    """
    x = np.asarray(voltage_v, dtype=float)
    y = np.asarray(inverse_capacitance_squared, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 4 or len(np.unique(x)) < 4:
        raise ValueError("At least four points at distinct voltages are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Regression inputs must be finite; filter explicitly")
    if not np.isfinite(min_abs_slope) or min_abs_slope < 0:
        raise ValueError("min_abs_slope must be finite and nonnegative")
    center = float(np.mean(x))
    span = float(np.ptp(x))
    xs = (x - center) / span
    yscale = float(np.max(np.abs(y)))
    if yscale == 0:
        yscale = 1.0
    ys = y / yscale
    design = np.column_stack((xs, np.ones(len(x))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, ys, rcond=None)
    if rank != 2:
        raise ValueError("Voltage values are numerically indistinguishable")
    residual = ys - design @ coefficients
    slope = float(coefficients[0] * (yscale / span))
    intercept = float((coefficients[1] - coefficients[0] * center / span) * yscale)
    sigma2 = float(residual @ residual / (len(x) - 2))
    transform = np.array([[yscale / span, 0], [-yscale * (center / span), yscale]])
    cov = transform @ (sigma2 * np.linalg.inv(design.T @ design)) @ transform.T
    if not np.isfinite([slope, intercept]).all() or not np.isfinite(cov).all():
        raise ValueError("Regression results exceed numeric range")
    sst = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = float(1 - residual @ residual / sst) if sst > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residual**2)) * yscale)
    flags = []
    if len(np.unique(x)) != len(x):
        flags.append("repeated_bias")
    if not np.isfinite(r2):
        flags.append("r_squared_undefined")
    resolution = np.finfo(float).eps * yscale / span * 16
    xintercept = xstd = None
    if abs(slope) <= max(min_abs_slope, resolution):
        flags.append("slope_too_small")
    else:
        xintercept = -intercept / slope
        gradient = np.array([-xintercept / slope, -1 / slope])
        variance = float(gradient @ cov @ gradient)
        if np.isfinite(xintercept) and np.isfinite(variance):
            xstd = float(np.sqrt(max(variance, 0)))
            if not x.min() <= xintercept <= x.max():
                flags.append("intercept_extrapolated")
        else:
            xintercept = None
            flags.append("intercept_unavailable")
    return LinearResult(slope, intercept, cov, r2, rmse, xintercept, xstd,
                        len(x), (float(x.min()), float(x.max())), flags)


def candidate_ranges(voltage_v, capacitance_f, *, min_span_v: float,
                     min_points: int = 4, min_abs_slope: float = 0.0) -> pd.DataFrame:
    """Enumerate all contiguous windows after stable voltage sorting, without selection.

    Returns original positional indices, bounds, OLS slope, R², RMSE and flags.
    Each window requires >=min_points and >=4 distinct voltages, and the specified
    minimum span. Repeated biases are retained; windows are exploratory and do not
    establish physical validity. Inputs must be finite and capacitances positive.
    """
    x = np.asarray(voltage_v, dtype=float)
    c = np.asarray(capacitance_f, dtype=float)
    if x.ndim != 1 or x.shape != c.shape or not np.isfinite(x).all() or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("Voltage and positive capacitance must be finite matching arrays")
    if not np.isfinite(min_span_v) or min_span_v <= 0 or not isinstance(min_points, int) or min_points < 4:
        raise ValueError("min_span_v must be positive and min_points at least four")
    if not np.isfinite(min_abs_slope) or min_abs_slope < 0:
        raise ValueError("min_abs_slope must be finite and nonnegative")
    y, _ = inverse_capacitance_squared(c)
    order = np.argsort(x, kind="stable")
    records = []
    for start in range(len(x)):
        for end in range(start + min_points, len(x) + 1):
            idx = order[start:end]
            if x[idx[-1]] - x[idx[0]] < min_span_v or len(np.unique(x[idx])) < 4:
                continue
            fit = linear_regression(x[idx], y[idx], min_abs_slope=min_abs_slope)
            records.append({"indices": tuple(int(i) for i in idx), "n_points": len(idx),
                            "voltage_min_v": fit.voltage_range_v[0], "voltage_max_v": fit.voltage_range_v[1],
                            "span_v": fit.voltage_range_v[1] - fit.voltage_range_v[0],
                            "slope": fit.slope, "intercept": fit.intercept,
                            "r_squared": fit.r_squared, "rmse": fit.rmse, "flags": tuple(fit.flags)})
    return pd.DataFrame.from_records(records, columns=["indices", "n_points", "voltage_min_v", "voltage_max_v",
                                                       "span_v", "slope", "intercept", "r_squared", "rmse", "flags"])


def carrier_density(regression: LinearResult, *, area_m2: float, relative_permittivity: float,
                    confirmed: bool, min_abs_slope: float = 0.0) -> dict:
    """Conditional total-capacitance Mott–Schottky density in m^-3 and cm^-3.

    Caller must explicitly confirm the range and relevant depletion/model/area
    assumptions. No p/n, interface, flat-band, reference, or thermal inference is
    made from slope sign. Uncertainty includes slope only, not area or permittivity.
    ``area_m2`` is the active area, not area-normalized capacitance.
    """
    if confirmed is not True:
        raise ValueError("Explicit confirmation of range and physical assumptions is required")
    if not np.isfinite([area_m2, relative_permittivity, min_abs_slope]).all() or area_m2 <= 0 or relative_permittivity <= 0 or min_abs_slope < 0:
        raise ValueError("Area and relative permittivity must be positive; slope threshold nonnegative")
    slope = regression.slope
    if not np.isfinite(slope) or abs(slope) <= min_abs_slope or "slope_too_small" in regression.flags:
        raise ValueError("Slope is zero or too small for a meaningful density")
    log_density = (np.log(2) - np.log(elementary_charge) - np.log(epsilon_0) -
                   np.log(relative_permittivity) - 2 * np.log(area_m2) - np.log(abs(slope)))
    if not np.log(np.finfo(float).tiny) <= log_density <= np.log(np.finfo(float).max):
        raise ValueError("Density outside representable range")
    with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
        density = float(np.exp(log_density))
        slope_std = float(np.sqrt(regression.covariance[0, 0]))
        density_std = float(density * slope_std / abs(slope))
    if not np.isfinite(density) or density <= 0 or not np.isfinite(density_std):
        raise ValueError("Density or uncertainty outside representable range")
    return {"density_m3": density, "density_cm3": density * 1e-6,
            "std_m3": density_std, "std_cm3": density_std * 1e-6,
            "slope_sign": int(np.sign(slope)), "area_m2": area_m2,
            "relative_permittivity": relative_permittivity,
            "voltage_range_v": regression.voltage_range_v,
            "assumption": "conditional effective density; slope uncertainty only"}