"""Auditable exact-frequency SPEIS cuts; no physical identification is implied."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from speis_fitting import group_spectra, inverse_capacitance_squared, linear_regression

FREQ = "freq/Hz"
REAL = "Re(Z)/Ohm"
NEG_IM = "-Im(Z)/Ohm"
VOLTAGE = "<Ewe>/V"
PHASE = "Phase(Z)/deg"


def load_spectra(path, *, chunksize=100_000, **group_options):
    """Scan all rows; preserve positive-frequency points and suspicious ungroupable rows.

    A zero-frequency, zero-impedance row is considered a non-EIS sentinel; other
    invalid-frequency rows are retained separately because their sweep is unknown.
    ``_row`` is the zero-based data-row position in the CSV, excluding the header.
    """
    path = Path(path)
    headers = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    required = [FREQ, REAL, NEG_IM, VOLTAGE]
    missing = [c for c in required if c not in headers]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    cols = required + [c for c in ["cycle number", PHASE, "time/s"] if c in headers]
    points, ungroupable = [], []
    total = sentinels = 0
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize, encoding="utf-8-sig"):
        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        chunk["_row"] = np.arange(total, total + len(chunk))
        total += len(chunk)
        f, re, im = (chunk[c].to_numpy(dtype=float) for c in (FREQ, REAL, NEG_IM))
        positive = np.isfinite(f) & (f > 0)
        sentinel = (f == 0) & (re == 0) & (im == 0)
        sentinels += int(sentinel.sum())
        points.append(chunk.loc[positive])
        ungroupable.append(chunk.loc[~positive & ~sentinel])
    empty = pd.DataFrame(columns=cols + ["_row"])
    eis = pd.concat(points, ignore_index=True) if points else empty.copy()
    bad = pd.concat(ungroupable, ignore_index=True) if ungroupable else empty.copy()
    spectra = group_spectra(eis, **group_options)
    invalid = (~np.isfinite(eis[REAL]) | ~np.isfinite(eis[NEG_IM]) | ~np.isfinite(eis[VOLTAGE]))
    return spectra, bad, {"total_rows": total, "zero_frequency_zero_impedance_sentinels": sentinels,
                          "positive_frequency_rows": len(eis), "positive_rows_invalid_impedance_or_voltage": int(invalid.sum()),
                          "ungroupable_frequency_rows": len(bad)}


@dataclass(frozen=True)
class PhaseGate:
    low_hz: float = 50_000.0
    high_hz: float = 100_000.0
    target_deg: float = -85.0
    max_deviation_deg: float = 5.0
    max_range_deg: float = 10.0
    max_neighbor_change_deg: float = 5.0

    def __post_init__(self):
        vals = [self.low_hz, self.high_hz, self.target_deg, self.max_deviation_deg,
                self.max_range_deg, self.max_neighbor_change_deg]
        if not np.isfinite(vals).all() or not 0 < self.low_hz <= self.high_hz or min(vals[3:]) < 0:
            raise ValueError("Invalid frequency band or phase thresholds")


def _angle_difference(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _point(row):
    re, im, v = (float(row[c]) for c in (REAL, NEG_IM, VOLTAGE))
    if not np.isfinite([re, im, v]).all() or (re == 0 and im == 0):
        return np.nan
    return float(np.degrees(np.arctan2(-im, re)))


def audit_frequencies(spectra, gate=PhaseGate(), *, voltage_offset_v=-0.2):
    """Audit the union of measured band frequencies and the exact intersection.

    Immediate neighbors are the next distinct measured frequencies on either
    side, even if invalid or duplicated; at least one must be unique, valid and
    within the configured phase change. The gate is heuristic, not a proof.
    Returns candidate and per-spectrum evidence tables; no point is selected.
    """
    if not np.isfinite(voltage_offset_v):
        raise ValueError("voltage offset must be finite")
    if len({s.spectrum_id for s in spectra}) != len(spectra):
        raise ValueError("spectrum IDs must be unique")
    frequencies = []
    prepared = []
    for spectrum in spectra:
        df = spectrum.data.copy()
        for col in (FREQ, REAL, NEG_IM, VOLTAGE):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        f = df[FREQ].to_numpy(dtype=float)
        freq = set(f[np.isfinite(f) & (f > 0)])
        frequencies.append(freq)
        prepared.append(df)
    common = set.intersection(*frequencies) if frequencies else set()
    union = set.union(*frequencies) if frequencies else set()
    candidates = sorted(f for f in union if gate.low_hz <= f <= gate.high_hz)
    detail, summary = [], []
    for f in candidates:
        rows = []
        for s, df, fs in zip(spectra, prepared, frequencies):
            matches = df.loc[df[FREQ] == f]
            rec = {"frequency_hz": f, "spectrum_id": s.spectrum_id, "cycle": s.cycle,
                   "bias_measured_v": s.voltage_v, "bias_corrected_v": s.voltage_v + voltage_offset_v,
                   "spectrum_flags": ", ".join(s.flags), "count": len(matches), "source_row": None,
                   "point_bias_v": np.nan, "phase_deg": np.nan, "instrument_phase_deg": np.nan,
                   "instrument_phase_difference_deg": np.nan, "lower_hz": np.nan, "lower_change_deg": np.nan,
                   "upper_hz": np.nan, "upper_change_deg": np.nan, "neighbor_pass": False}
            if len(matches) == 1:
                row = matches.iloc[0]
                phase = _point(row)
                rec.update(source_row=row["_row"], point_bias_v=row[VOLTAGE], phase_deg=phase)
                if PHASE in df and np.isfinite(row[PHASE]) and np.isfinite(phase):
                    rec["instrument_phase_deg"] = row[PHASE]
                    rec["instrument_phase_difference_deg"] = _angle_difference(phase, row[PHASE])
                for label, nearby in (("lower", [x for x in fs if x < f]), ("upper", [x for x in fs if x > f])):
                    if nearby:
                        nf = max(nearby) if label == "lower" else min(nearby)
                        rec[label + "_hz"] = nf
                        neighbor = df.loc[df[FREQ] == nf]
                        if len(neighbor) == 1 and np.isfinite(phase):
                            nphase = _point(neighbor.iloc[0])
                            if np.isfinite(nphase):
                                rec[label + "_change_deg"] = _angle_difference(phase, nphase)
                changes = [rec["lower_change_deg"], rec["upper_change_deg"]]
                rec["neighbor_pass"] = any(np.isfinite(x) and x <= gate.max_neighbor_change_deg for x in changes)
            rows.append(rec)
        detail.extend(rows)
        phases = np.array([r["phase_deg"] for r in rows])
        good = np.isfinite(phases)
        deviations = np.array([_angle_difference(x, gate.target_deg) for x in phases[good]])
        all_valid = len(rows) > 0 and all(r["count"] == 1 for r in rows) and good.all()
        phase_range = float(np.ptp(phases)) if all_valid else np.nan
        median = float(np.median(deviations)) if len(deviations) else np.nan
        maximum = float(np.max(deviations)) if len(deviations) else np.nan
        passed = bool(all_valid and maximum <= gate.max_deviation_deg and phase_range <= gate.max_range_deg
                      and all(r["neighbor_pass"] for r in rows))
        summary.append({"frequency_hz": f, "exact_common": f in common, "missing_sweeps": sum(r["count"] == 0 for r in rows),
                        "duplicate_sweeps": sum(r["count"] > 1 for r in rows), "invalid_sweeps": sum(r["count"] == 1 and not np.isfinite(r["phase_deg"]) for r in rows),
                        "median_deviation_deg": median, "max_deviation_deg": maximum, "phase_range_deg": phase_range,
                        "neighbor_fail_sweeps": sum(not r["neighbor_pass"] for r in rows), "passes": passed})
    cols = ["frequency_hz", "exact_common", "missing_sweeps", "duplicate_sweeps", "invalid_sweeps", "median_deviation_deg", "max_deviation_deg", "phase_range_deg", "neighbor_fail_sweeps", "passes"]
    result = pd.DataFrame(summary, columns=cols).sort_values(["passes", "median_deviation_deg", "phase_range_deg", "frequency_hz"], ascending=[False, True, True, True], na_position="last").reset_index(drop=True)
    result.attrs["phase_gate"] = gate
    result.attrs["spectrum_ids"] = tuple(s.spectrum_id for s in spectra)
    return result, pd.DataFrame(detail), sorted(common)


def extract_cut(spectra, candidates, frequency_hz, *, confirmed=False, voltage_offset_v=-0.2,
                series_rs_ohm=None, rs_source=None, rs_uncertainty_ohm=None):
    """Extract a passing exact frequency only with explicit confirmation; report masks."""
    if confirmed is not True:
        raise ValueError("Explicit frequency confirmation required")
    if not np.isfinite(voltage_offset_v):
        raise ValueError("voltage offset must be finite")
    gate = candidates.attrs.get("phase_gate")
    if not isinstance(gate, PhaseGate) or candidates.attrs.get("spectrum_ids") != tuple(s.spectrum_id for s in spectra):
        raise ValueError("Audit does not match the selected spectra; rerun the audit")
    choice = candidates.loc[(candidates.frequency_hz == frequency_hz) & candidates.passes]
    if len(choice) != 1:
        raise ValueError("Frequency is not a unique passing exact candidate")
    current, _, _ = audit_frequencies(spectra, gate, voltage_offset_v=voltage_offset_v)
    if not ((current.frequency_hz == frequency_hz) & current.passes).any():
        raise ValueError("Data changed since the audit; rerun the audit")
    if series_rs_ohm is not None:
        if not np.isfinite(series_rs_ohm) or series_rs_ohm < 0 or not isinstance(rs_source, str) or not rs_source.strip():
            raise ValueError("Rs must be finite, nonnegative, and have a documented source")
        if rs_uncertainty_ohm is not None and (not np.isfinite(rs_uncertainty_ohm) or rs_uncertainty_ohm < 0):
            raise ValueError("Rs uncertainty must be finite and nonnegative")
    records = []
    for s in spectra:
        found = s.data.loc[pd.to_numeric(s.data[FREQ], errors="coerce") == frequency_hz]
        if len(found) != 1:
            raise ValueError("Selected spectra changed since audit; rerun the audit")
        row = found.iloc[0]
        re, negim, v = (float(row[c]) for c in (REAL, NEG_IM, VOLTAGE))
        z = complex(re, -negim)
        valid = np.isfinite([re, negim, v, s.voltage_v]).all() and abs(z) > 0 and negim > 0
        cp = cs = cp_rs = np.nan
        if valid:
            omega = 2 * np.pi * frequency_hz
            cp = (1 / z).imag / omega
            cs = 1 / (omega * negim)
            if series_rs_ohm is not None:
                corrected = z - series_rs_ohm
                cp_rs = (1 / corrected).imag / omega
            valid = np.isfinite([cp, cs]).all() and cp > 0 and cs > 0
        records.append({"spectrum_id": s.spectrum_id, "cycle": s.cycle, "source_row": row["_row"],
                        "frequency_hz": frequency_hz, "bias_measured_v": s.voltage_v,
                        "bias_corrected_v": s.voltage_v + voltage_offset_v, "point_bias_v": v,
                        "phase_deg": np.degrees(np.angle(z)), "Re_ohm": re, "neg_Im_ohm": negim,
                        "Cp_F": cp, "Cs_F": cs, "Cp_Rs_corrected_F": cp_rs,
                        "eligible": bool(valid), "point_flags": "" if valid else "invalid_or_noncapacitive",
                        "spectrum_flags": ", ".join(s.flags), "Rs_ohm": series_rs_ohm,
                        "Rs_source": rs_source, "Rs_uncertainty_ohm": rs_uncertainty_ohm})
    return pd.DataFrame(records)


def regress_cut(cut, bounds_v, *, confirmed=False, capacitance_column="Cp_F"):
    """Explicit inclusive corrected-bias mask and unweighted OLS; return mask and fit."""
    if confirmed is not True:
        raise ValueError("Explicit regression confirmation required")
    lo, hi = bounds_v
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("Provide finite increasing corrected-bias bounds")
    x = cut["bias_corrected_v"].to_numpy(dtype=float)
    c = cut[capacitance_column].to_numpy(dtype=float)
    mask = cut["eligible"].to_numpy(dtype=bool) & np.isfinite(x) & np.isfinite(c) & (c > 0) & (x >= lo) & (x <= hi)
    y, _ = inverse_capacitance_squared(c[mask])
    return mask, linear_regression(x[mask], y)