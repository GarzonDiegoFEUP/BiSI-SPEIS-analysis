"""Auditable two-RCPE Voigt fitting; branch labels are frequency order only.

Weights, impedance sign and single-branch fit are shared with speis_fitting.
Fit diagnostics are heuristics, not evidence that either branch is depletion.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from speis_fitting import effective_capacitance, fit_spectrum, impedance_rcpe


def load_eis_csv(path, chunksize=50000):
    """Stream CSV, preserving all positive-frequency rows and row identities.

    Zero frequencies typically represent non-EIS acquisition rows. Invalid and
    negative frequencies cannot be grouped reliably; counts and a bounded sample
    of original rows are returned for audit, rather than silently discarded.
    _row is the zero-based data record ordinal, excluding the CSV header.
    """
    cols = ['freq/Hz', 'Re(Z)/Ohm', '-Im(Z)/Ohm', '<Ewe>/V', 'cycle number']
    parts, samples = [], []
    counts = dict(total=0, positive_frequency=0, zero_frequency=0,
                  negative_frequency=0, invalid_frequency=0, invalid_positive_impedance=0,
                  invalid_positive_voltage=0)
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        chunk['_row'] = np.arange(counts['total'], counts['total'] + len(chunk))
        counts['total'] += len(chunk)
        f = pd.to_numeric(chunk['freq/Hz'], errors='coerce').to_numpy(float)
        finite = np.isfinite(f)
        positive = finite & (f > 0)
        negative = finite & (f < 0)
        invalid = ~finite
        counts['positive_frequency'] += int(positive.sum())
        counts['zero_frequency'] += int((finite & (f == 0)).sum())
        counts['negative_frequency'] += int(negative.sum())
        counts['invalid_frequency'] += int(invalid.sum())
        part = chunk.loc[positive].copy()
        re = pd.to_numeric(part['Re(Z)/Ohm'], errors='coerce').to_numpy(float)
        im = pd.to_numeric(part['-Im(Z)/Ohm'], errors='coerce').to_numpy(float)
        v = pd.to_numeric(part['<Ewe>/V'], errors='coerce').to_numpy(float)
        counts['invalid_positive_impedance'] += int((~(np.isfinite(re) & np.isfinite(im))).sum())
        counts['invalid_positive_voltage'] += int((~np.isfinite(v)).sum())
        parts.append(part)
        if sum(len(s) for s in samples) < 20:
            samples.append(chunk.loc[negative | invalid].head(20 - sum(len(s) for s in samples)))
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols + ['_row'])
    audit = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=cols + ['_row'])
    return frame, counts, audit


def fit_mask(spectrum, fmin=None, fmax=None):
    """Identical explicit selection for both models; all exclusions remain auditable."""
    if fmin is not None and (not np.isfinite(fmin) or fmin <= 0):
        raise ValueError('fmin must be positive')
    if fmax is not None and (not np.isfinite(fmax) or fmax <= 0):
        raise ValueError('fmax must be positive')
    if fmin is not None and fmax is not None and fmin > fmax:
        raise ValueError('frequency limits reversed')
    d = spectrum.data
    f, re, im = (pd.to_numeric(d[c], errors='coerce').to_numpy(float)
                 for c in ('freq/Hz', 'Re(Z)/Ohm', '-Im(Z)/Ohm'))
    mask = np.isfinite(f) & (f > 0) & np.isfinite(re) & np.isfinite(im) & (im >= 0)
    if fmin is not None:
        mask &= f >= fmin
    if fmax is not None:
        mask &= f <= fmax
    return mask


def impedance_voigt(frequency_hz, rs, r1, q1, a1, r2, q2, a2):
    """Rs + two parallel R-CPE branches, with the BioLogic negative-Im sign."""
    return impedance_rcpe(frequency_hz, rs, r1, q1, a1) + impedance_rcpe(frequency_hz, rs, r2, q2, a2) - rs


def peak_frequency(r, q, alpha):
    if not np.isfinite([r, q, alpha]).all() or r <= 0 or q <= 0 or not 0 < alpha <= 1:
        raise ValueError('Invalid branch parameters')
    with np.errstate(over='ignore', under='ignore'):
        return float(np.exp(-(np.log(r) + np.log(q)) / alpha - np.log(2 * np.pi)))


def order_branches(parameters, covariance=None):
    """Canonicalize by increasing peak frequency, including covariance permutation."""
    p = np.asarray(parameters, dtype=float)
    if p.shape != (7,):
        raise ValueError('Expected seven parameters')
    peaks = [peak_frequency(*p[1:4]), peak_frequency(*p[4:7])]
    order = np.array([0, 4, 5, 6, 1, 2, 3]) if peaks[0] > peaks[1] else np.arange(7)
    return p[order], None if covariance is None else np.asarray(covariance)[np.ix_(order, order)]


@dataclass
class VoigtResult:
    success: bool = False
    parameters: object = None
    covariance: object = None
    capacitances: tuple = (None, None)
    capacitance_stds: tuple = (None, None)
    peaks_hz: tuple = (None, None)
    relative_rmse: float = np.nan
    raw_rmse_ohm: float = np.nan
    weighted_rss: float = np.nan
    jacobian_condition: float = np.nan
    used_rows: tuple = ()
    excluded_rows: tuple = ()
    flags: list = field(default_factory=list)
    message: str = ''
    starts_converged: int = 0
    near_optimal_starts: int = 0


DEFAULT_BOUNDS = ((1e-12, 1e-12, 1e-30, .05, 1e-12, 1e-30, .05),
                  (1e15, 1e15, 1e5, 1., 1e15, 1e5, 1.))


def fit_voigt(spectrum, *, mask=None, bounds=DEFAULT_BOUNDS, n_starts=12,
              max_nfev=2500, modulus_floor_fraction=1e-12, max_relative_rmse=.1,
              max_condition=1e8, bound_tolerance=1e-4, near_cost_fraction=.01,
              ambiguity_log_ratio=np.log(3), min_peak_decades=.5):
    """Deterministic dispersed bounded CNLS; log-positive coordinates.

    Near-optimal starts differing by a factor of three (or 0.15 in alpha) flag
    ambiguity, after allowing label exchange. These thresholds are heuristics.
    """
    lo, hi = (np.asarray(x, float) for x in bounds)
    if (lo.shape != (7,) or hi.shape != (7,) or not np.isfinite(lo).all() or
            not np.isfinite(hi).all() or np.any(lo <= 0) or np.any(lo >= hi) or
            hi[3] > 1 or hi[6] > 1):
        raise ValueError('Invalid physical bounds')
    if (not isinstance(n_starts, int) or n_starts < 1 or max_nfev < 1 or
        not np.isfinite([modulus_floor_fraction, max_relative_rmse, max_condition,
                         bound_tolerance, near_cost_fraction, ambiguity_log_ratio, min_peak_decades]).all() or
        modulus_floor_fraction <= 0 or max_relative_rmse < 0 or max_condition <= 1 or
        not 0 <= bound_tolerance < .5 or near_cost_fraction < 0 or
        ambiguity_log_ratio <= 0 or min_peak_decades < 0):
        raise ValueError('Invalid fitting configuration')
    d = spectrum.data
    selected = fit_mask(spectrum) if mask is None else np.asarray(mask)
    if selected.shape != (len(d),) or selected.dtype.kind != 'b':
        raise ValueError('mask must be a positional boolean array')
    result = VoigtResult(used_rows=tuple(d.loc[selected, '_row']),
                         excluded_rows=tuple(d.loc[~selected, '_row']), flags=list(spectrum.flags))
    if result.excluded_rows:
        result.flags.append('excluded_points')
    part = d.loc[selected]
    f, re, im = (pd.to_numeric(part[c], errors='coerce').to_numpy(float)
                 for c in ('freq/Hz', 'Re(Z)/Ohm', '-Im(Z)/Ohm'))
    if not (np.isfinite(f).all() and np.isfinite(re).all() and np.isfinite(im).all()) or np.any(f <= 0):
        result.flags.append('invalid_selected_data')
        result.message = 'Selected data must be finite with positive frequencies.'
        return result
    if len(np.unique(f)) < 6 or 2 * len(f) <= 7:
        result.flags.append('too_few_distinct_frequencies')
        return result
    z = re - 1j * im
    mag = np.abs(z)
    median = float(np.median(mag))
    if not np.isfinite(mag).all() or not np.isfinite(median) or median <= 0:
        result.flags.append('invalid_weight_scale')
        return result
    weights = np.maximum(mag, max(modulus_floor_fraction * median, np.finfo(float).tiny))
    logidx = [0, 1, 2, 4, 5]
    lb, ub = lo.copy(), hi.copy()
    lb[logidx], ub[logidx] = np.log(lo[logidx]), np.log(hi[logidx])

    def unpack(x):
        p = x.copy()
        p[logidx] = np.exp(p[logidx])
        return p

    def residual(x):
        delta = (impedance_voigt(f, *unpack(x)) - z) / weights
        return np.r_[delta.real, delta.imag]

    rs0 = max(re[np.argmax(f)], median * .01, lo[0])
    r0 = max(np.ptp(re), median * .5, lo[1])
    f_low, f_high = np.min(f), np.max(f)
    starts = []
    # Frequencies span the measured window; some guesses straddle its edges.
    grid = ((.15, .75), (.35, .9), (.05, .5), (.25, .6), (.0, 1.), (.4, .65))
    for i in range(n_starts):
        t1, t2 = grid[i % len(grid)]
        a = (.65, .85, .97)[(i // len(grid) + i) % 3]
        frac = (.25, .5, .75)[i % 3]
        r1, r2 = r0 * frac, r0 * (1 - frac)
        lf1 = np.log(f_low) * (1 - t1) + np.log(f_high) * t1
        lf2 = np.log(f_low) * (1 - t2) + np.log(f_high) * t2
        guess = np.array([np.log(rs0), np.log(r1), -np.log(r1) - a * (np.log(2*np.pi)+lf1), a,
                          np.log(r2), -np.log(r2) - a * (np.log(2*np.pi)+lf2), a])
        guess = np.clip(guess, lb + (ub-lb)*1e-8, ub - (ub-lb)*1e-8)
        try:
            fit = least_squares(residual, guess, bounds=(lb, ub), max_nfev=max_nfev)
            if fit.success and np.isfinite(fit.cost) and np.isfinite(fit.x).all():
                starts.append(fit)
        except (ValueError, FloatingPointError, OverflowError):
            continue
    result.starts_converged = len(starts)
    if not starts:
        result.flags.append('fit_failed')
        result.message = 'All starts failed or did not converge.'
        return result
    best = min(starts, key=lambda x: x.cost)
    p, _ = order_branches(unpack(best.x))
    result.parameters = p
    result.success = True
    result.message = str(best.message)
    result.weighted_rss = float(2 * best.cost)
    delta = impedance_voigt(f, *p) - z
    result.raw_rmse_ohm = float(np.sqrt(np.mean(np.abs(delta)**2)))
    result.relative_rmse = float(np.sqrt(np.mean(np.abs(delta / weights)**2)))
    if result.relative_rmse > max_relative_rmse:
        result.flags.append('poor_relative_fit')
    near = [x for x in starts if x.cost <= best.cost * (1 + near_cost_fraction) + 1e-16]
    result.near_optimal_starts = len(near)
    if len(starts) < 2:
        result.flags.append('multistart_unverified')
    for candidate in near:
        cp, _ = order_branches(unpack(candidate.x))
        if (np.max(np.abs(np.log(cp[logidx] / p[logidx]))) > ambiguity_log_ratio or
                np.max(np.abs(cp[[3, 6]] - p[[3, 6]])) > .15):
            result.flags.append('near_optimal_parameter_ambiguity')
            break
    peaks = (peak_frequency(*p[1:4]), peak_frequency(*p[4:7]))
    result.peaks_hz = peaks
    if not all(np.isfinite(v) and f_low <= v <= f_high for v in peaks):
        result.flags.append('peak_outside_measured_band')
    if min(peaks) <= 0 or not np.isfinite(peaks).all() or np.log10(max(peaks) / min(peaks)) < min_peak_decades:
        result.flags.append('overlapping_peaks')
    if np.any((best.x-lb)/(ub-lb) <= bound_tolerance) or np.any((ub-best.x)/(ub-lb) <= bound_tolerance) or np.any(best.active_mask):
        result.flags.append('parameter_at_bound')
    s = np.linalg.svd(best.jac, compute_uv=False)
    result.jacobian_condition = float(s[0]/s[-1]) if s[-1] > 0 else np.inf
    if not np.isfinite(result.jacobian_condition) or result.jacobian_condition > max_condition or np.linalg.matrix_rank(best.jac) < 7:
        result.flags.append('ill_conditioned')
    unsafe = {'parameter_at_bound', 'ill_conditioned', 'near_optimal_parameter_ambiguity',
              'overlapping_peaks', 'peak_outside_measured_band', 'multistart_unverified', 'poor_relative_fit'}
    if not unsafe.intersection(result.flags):
        _, s, vt = np.linalg.svd(best.jac, full_matrices=False)
        cov_x = ((vt.T / s**2) @ vt) * (2 * best.cost / (2 * len(f) - 7))
        rawp = unpack(best.x)
        transform = np.diag(np.where(np.isin(np.arange(7), logidx), rawp, 1.))
        cov = transform @ cov_x @ transform
        if np.isfinite(cov).all():
            _, result.covariance = order_branches(rawp, (cov + cov.T)/2)
    if result.covariance is None:
        result.flags.append('uncertainty_unavailable')
    caps, stds = [], []
    for start in (1, 4):
        indices = [0, start, start+1, start+2]
        branch_cov = None if result.covariance is None else result.covariance[np.ix_(indices, indices)]
        try:
            c, std = effective_capacitance(*p[start:start+3], branch_cov)
        except ValueError:
            try:
                c, _ = effective_capacitance(*p[start:start+3])
            except ValueError:
                c = None
            std = None
            result.flags.append('capacitance_or_uncertainty_unavailable')
        caps.append(c)
        stds.append(std)
    result.capacitances, result.capacitance_stds = tuple(caps), tuple(stds)
    return result


def aicc(weighted_rss, n_observations, n_parameters):
    """Gaussian iid residual AICc up to common constants; weights are data-dependent."""
    if n_observations <= n_parameters + 1 or not np.isfinite(weighted_rss) or weighted_rss < 0:
        return np.nan
    if weighted_rss == 0:
        return -np.inf
    n, k = n_observations, n_parameters
    return float(n * np.log(weighted_rss/n) + 2*k + 2*k*(k+1)/(n-k-1))


def compare_models(spectrum, *, mask=None, modulus_floor_fraction=1e-12, **voigt_options):
    """Fit on exactly the same rows and modulus weights, returning both results."""
    mask = fit_mask(spectrum) if mask is None else np.asarray(mask)
    one = fit_spectrum(spectrum, mask=mask, modulus_floor_fraction=modulus_floor_fraction)
    two = fit_voigt(spectrum, mask=mask, modulus_floor_fraction=modulus_floor_fraction, **voigt_options)
    n = len(one.used_rows)
    assert one.used_rows == two.used_rows and one.excluded_rows == two.excluded_rows
    one_rss = n * one.relative_rmse**2 if one.success else np.nan
    comparison = {'n_points': n, 'single_k': 4, 'voigt_k': 7,
                  'single_weighted_rss': one_rss, 'voigt_weighted_rss': two.weighted_rss,
                  'single_relative_rmse': one.relative_rmse if one.success else np.nan,
                  'voigt_relative_rmse': two.relative_rmse,
                  'single_aicc': aicc(one_rss, 2*n, 4), 'voigt_aicc': aicc(two.weighted_rss, 2*n, 7)}
    return one, two, comparison


def crossing_flags(results):
    """Flag possible label exchange across adjacent voltage-sorted spectra.

    A lower cross-assignment distance in log peak frequency is only a warning;
    peak-order labels cannot establish physical branch continuity.
    """
    ordered = sorted(((s.voltage_v, s.spectrum_id, r) for s, r in results
                      if r.success and np.isfinite(s.voltage_v) and
                      np.isfinite(r.peaks_hz).all() and min(r.peaks_hz) > 0))
    flagged = set()
    for (_, id1, r1), (_, id2, r2) in zip(ordered, ordered[1:]):
        a, b = np.log(r1.peaks_hz), np.log(r2.peaks_hz)
        same = np.linalg.norm(a-b)
        swapped = np.linalg.norm(a-b[::-1])
        # Also flag near-coincident peaks in either neighboring sweep.
        if swapped <= same * 1.1 or min(np.diff(a)[0], np.diff(b)[0]) < np.log(3):
            flagged.update((id1, id2))
    return flagged