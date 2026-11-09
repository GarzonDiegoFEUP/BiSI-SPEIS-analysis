"""Deterministic tests for the standalone SPEIS analysis API."""

import unittest

import numpy as np
import pandas as pd
from scipy.constants import elementary_charge, epsilon_0

from speis_fitting import (candidate_ranges, carrier_density, effective_capacitance,
                           fit_spectrum, group_spectra, impedance_rcpe,
                           inverse_capacitance_squared, linear_regression)


def spectrum_frame(frequencies, voltage=0.1, cycle=1, parameters=(20., 500., 2e-5, 0.83)):
    f = np.asarray(frequencies, dtype=float)
    z = impedance_rcpe(f, *parameters)
    return pd.DataFrame({"freq/Hz": f, "Re(Z)/Ohm": z.real, "-Im(Z)/Ohm": -z.imag,
                         "<Ewe>/V": voltage, "cycle number": cycle})


class GroupTests(unittest.TestCase):
    def test_resets_cycles_order_and_repeated_bias(self):
        f = np.geomspace(1e4, 1, 8)
        a = spectrum_frame(f, 0.1, 1)
        b = spectrum_frame(f, 0.1005, 1)
        c = spectrum_frame(f, 0.3, 2)
        frame = pd.concat([a, b, c], ignore_index=True)
        frame["_row"] = np.arange(len(frame)) + 42
        frame = frame.sample(frac=1, random_state=4)
        groups = group_spectra(frame, expected_points=8, expected_frequency_range=(1, 1e4))
        self.assertEqual([len(s.data) for s in groups], [8, 8, 8])
        self.assertEqual(groups[0].data["_row"].tolist(), list(range(42, 50)))
        self.assertIn("possible_repeated_bias", groups[0].flags)
        self.assertIn("possible_repeated_bias", groups[1].flags)
        self.assertNotIn("possible_repeated_bias", groups[2].flags)
        self.assertNotIn("incomplete_sweep", groups[0].flags)

    def test_cycle_change_without_reset_and_incomplete(self):
        frame = spectrum_frame(np.geomspace(100, 1, 8))
        frame.loc[4:, "cycle number"] = 2
        frame.loc[0, "<Ewe>/V"] = 0.2
        groups = group_spectra(frame, expected_points=8, voltage_spread_limit=0.05)
        self.assertEqual([len(s.data) for s in groups], [4, 4])
        self.assertIn("voltage_spread", groups[0].flags)
        self.assertIn("incomplete_sweep", groups[0].flags)
        self.assertIn("too_few_distinct_frequencies", groups[0].flags)

    def test_invalid_retained_and_explicit_exclusion(self):
        frame = spectrum_frame(np.geomspace(1e4, 1, 9))
        frame.loc[2, "Re(Z)/Ohm"] = np.nan
        group = group_spectra(frame)[0]
        self.assertEqual(len(group.data), 9)
        self.assertIn("invalid_impedance_or_frequency", group.flags)
        self.assertIn("completeness_unverified", group.flags)
        failed = fit_spectrum(group)
        self.assertFalse(failed.success)
        mask = np.ones(9, dtype=bool)
        mask[2] = False
        fit = fit_spectrum(group, mask=mask)
        self.assertTrue(fit.success)
        self.assertEqual(fit.excluded_rows, (2,))
        self.assertIn("user_excluded_points", fit.flags)

    def test_ascending_and_minor_reversal(self):
        frame = spectrum_frame([1, 2, 4, 3.5, 8, 16, 32, 1, 2, 4, 8, 16, 32])
        groups = group_spectra(frame, direction="ascending")
        self.assertEqual([len(s.data) for s in groups], [7, 6])
        self.assertIn("frequency_reversal", groups[0].flags)

    def test_duplicate_row_rejected(self):
        frame = spectrum_frame(np.geomspace(100, 1, 6))
        frame["_row"] = 1
        with self.assertRaises(ValueError):
            group_spectra(frame)


class FitTests(unittest.TestCase):
    def test_noisy_rcpe_recovery_and_uncertainty(self):
        f = np.geomspace(1e5, 0.1, 70)
        expected = np.array([20., 500., 2e-5, 0.83])
        frame = spectrum_frame(f, parameters=expected)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.04, (len(f), 2))
        frame["Re(Z)/Ohm"] += noise[:, 0]
        frame["-Im(Z)/Ohm"] += noise[:, 1]
        result = fit_spectrum(group_spectra(frame)[0])
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.parameters, expected, rtol=0.015)
        self.assertLess(result.relative_rmse, 0.003)
        self.assertGreater(result.modulus_floor_ohm, 0)
        self.assertIsNotNone(result.covariance)
        self.assertGreater(result.capacitance_std_f, 0)
        self.assertEqual(len(result.used_rows), len(f))
        self.assertLess(impedance_rcpe([1], *expected).imag[0], 0)

    def test_ideal_capacitor_and_bounds(self):
        f = np.geomspace(1e5, 0.1, 60)
        params = (10., 200., 4e-5, 1.)
        result = fit_spectrum(group_spectra(spectrum_frame(f, parameters=params))[0])
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.parameters, params, rtol=0.002)
        self.assertLessEqual(result.parameters[3], 1)
        self.assertAlmostEqual(result.capacitance_f, params[2], delta=1e-7)
        self.assertIn("parameter_at_bound", result.flags)
        self.assertIsNone(result.covariance)

    def test_six_distinct_frequencies_minimum(self):
        frame = spectrum_frame([100, 100, 50, 20, 10, 5])
        fit = fit_spectrum(group_spectra(frame)[0])
        self.assertFalse(fit.success)
        self.assertIn("too_few_distinct_frequencies", fit.flags)

    def test_poor_fit_and_explicit_threshold(self):
        frame = spectrum_frame(np.geomspace(1e4, 1, 20))
        frame["-Im(Z)/Ohm"] *= -1
        fit = fit_spectrum(group_spectra(frame)[0], max_relative_rmse=0.001)
        self.assertTrue(fit.success)
        self.assertIn("poor_relative_fit", fit.flags)

    def test_ill_conditioned_suppresses_uncertainty(self):
        frame = spectrum_frame(np.geomspace(1e4, 1, 25))
        fit = fit_spectrum(group_spectra(frame)[0], max_condition=1.01)
        self.assertTrue(fit.success)
        self.assertIn("ill_conditioned", fit.flags)
        self.assertIn("uncertainty_unavailable", fit.flags)
        self.assertIsNone(fit.covariance)

    def test_extreme_model_does_not_overflow(self):
        z = impedance_rcpe([1e-300, 1e300], 1, 1e100, 1e100, 1)
        self.assertTrue(np.isfinite(z).all())
        self.assertAlmostEqual(z[0].real / 1e100, 1)
        self.assertAlmostEqual(z[1].real, 1)

    def test_capacitance_covariance_and_validation(self):
        rp, q, a = 500., 2e-5, 0.8
        cov = np.diag([1., 25., 1e-14, 1e-4])
        c, std = effective_capacitance(rp, q, a, cov)
        self.assertAlmostEqual(c, q**(1/a) * rp**((1-a)/a))
        self.assertGreater(std, 0)
        self.assertAlmostEqual(effective_capacitance(rp, q, 1)[0], q)
        cov[2, 2] = -1
        with self.assertRaises(ValueError):
            effective_capacitance(rp, q, a, cov)


class RegressionTests(unittest.TestCase):
    def test_inverse_capacitance_uncertainty(self):
        y, sigma = inverse_capacitance_squared([2e-6, 4e-6], [1e-8, 2e-8])
        np.testing.assert_allclose(y, [1 / (2e-6)**2, 1 / (4e-6)**2])
        np.testing.assert_allclose(sigma / y, [0.01, 0.01])
        with self.assertRaises(ValueError):
            inverse_capacitance_squared([1e-6], [-1])

    def test_positive_negative_slopes_density_and_area(self):
        x = np.linspace(-0.2, 0.5, 8)
        area = 9e-6  # 3 mm x 3 mm
        epsilon_r = 38.
        density = 1e22
        slope = 2 / (elementary_charge * epsilon_0 * epsilon_r * area**2 * density)
        for sign in (-1, 1):
            y = 5e16 + sign * slope * x
            regression = linear_regression(x, y)
            estimate = carrier_density(regression, area_m2=area, relative_permittivity=epsilon_r, confirmed=True)
            self.assertAlmostEqual(regression.slope / (sign * slope), 1, places=10)
            self.assertAlmostEqual(estimate["density_m3"] / density, 1, places=10)
            self.assertAlmostEqual(estimate["density_cm3"] / (density * 1e-6), 1, places=10)
            self.assertEqual(estimate["slope_sign"], sign)
            doubled_area = carrier_density(regression, area_m2=2 * area, relative_permittivity=epsilon_r, confirmed=True)
            self.assertAlmostEqual(doubled_area["density_m3"] / density, 0.25)

    def test_intercept_covariance_with_noise_and_extrapolation(self):
        x = np.array([0., .1, .2, .3, .4, .5])
        y = 2e14 * x + 3e14 + np.array([1., -2, 1, -1, 2, -1]) * 1e12
        fit = linear_regression(x, y)
        expected_cov = np.linalg.inv(np.column_stack((x, np.ones(len(x)))).T @
                                     np.column_stack((x, np.ones(len(x)))))
        expected_cov *= np.sum((y - (fit.slope*x + fit.intercept))**2) / (len(x)-2)
        np.testing.assert_allclose(fit.covariance, expected_cov, rtol=1e-10)
        g = np.array([fit.intercept / fit.slope**2, -1 / fit.slope])
        self.assertAlmostEqual(fit.x_intercept_std_v, np.sqrt(g @ expected_cov @ g))
        self.assertIn("intercept_extrapolated", fit.flags)

    def test_zero_and_small_slope(self):
        x = np.arange(5.)
        fit = linear_regression(x, np.ones(5) * 1e15)
        self.assertIn("slope_too_small", fit.flags)
        self.assertIn("r_squared_undefined", fit.flags)
        self.assertIsNone(fit.x_intercept_v)
        with self.assertRaises(ValueError):
            carrier_density(fit, area_m2=9e-6, relative_permittivity=38, confirmed=True)
        fit = linear_regression(x, 1e15 + x * 1e10, min_abs_slope=2e10)
        self.assertIn("slope_too_small", fit.flags)

    def test_no_inference_without_confirmation(self):
        fit = linear_regression(np.arange(4.), np.arange(4.) + 1)
        with self.assertRaises(ValueError):
            carrier_density(fit, area_m2=9e-6, relative_permittivity=38, confirmed=False)
        with self.assertRaises(ValueError):
            carrier_density(fit, area_m2=0, relative_permittivity=38, confirmed=True)

    def test_small_sample_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            linear_regression([1, 2, 3], [1, 2, 3])
        with self.assertRaises(ValueError):
            linear_regression([1, 1, 2, 3], [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            linear_regression([1, 2, 3, 4], [1, 2, np.nan, 4])

    def test_candidates_sorted_no_merging(self):
        x = np.array([0.3, 0.1, 0.2, 0.2, 0.4])
        c = 1 / np.sqrt(1e15 * (x + 2))
        candidates = candidate_ranges(x, c, min_span_v=0.25)
        self.assertEqual(len(candidates), 1)
        row = candidates.iloc[0]
        self.assertEqual(row["indices"], (1, 2, 3, 0, 4))
        self.assertEqual(row["n_points"], 5)
        self.assertIn("repeated_bias", row["flags"])
        self.assertAlmostEqual(row["r_squared"], 1)
        self.assertEqual(len(candidate_ranges(x, c, min_span_v=1)), 0)
        with self.assertRaises(ValueError):
            candidate_ranges(x, [1, 1, 0, 1, 1], min_span_v=0.1)


if __name__ == "__main__":
    unittest.main()