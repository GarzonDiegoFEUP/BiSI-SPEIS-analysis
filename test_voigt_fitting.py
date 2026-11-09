"""Synthetic and audit tests for the independent Voigt workflow."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from speis_fitting import group_spectra, linear_regression, carrier_density
from voigt_fitting import (aicc, compare_models, crossing_flags, fit_mask, fit_voigt,
                           impedance_voigt, load_eis_csv, order_branches, peak_frequency)


def make_spectrum(params=(12., 400., 2e-4, .87, 100., 2e-6, .9), n=75):
    f = np.geomspace(1e6, .01, n)
    z = impedance_voigt(f, *params)
    d = pd.DataFrame({'freq/Hz': f, 'Re(Z)/Ohm': z.real, '-Im(Z)/Ohm': -z.imag,
                      '<Ewe>/V': .3, 'cycle number': 1})
    return group_spectra(d)[0]


class VoigtTests(unittest.TestCase):
    def test_separated_recovery_and_fair_comparison(self):
        expected = np.array([12., 400., 2e-4, .87, 100., 2e-6, .9])
        s = make_spectrum(expected)
        s.data.loc[0, '-Im(Z)/Ohm'] = -1
        s.data.loc[1, 'Re(Z)/Ohm'] = np.nan
        mask = fit_mask(s, fmin=.02)
        one, two, comp = compare_models(s, mask=mask, n_starts=8)
        self.assertTrue(one.success)
        self.assertTrue(two.success, two.message)
        np.testing.assert_allclose(two.parameters, expected, rtol=.01)
        self.assertEqual(one.used_rows, two.used_rows)
        self.assertEqual(one.excluded_rows, two.excluded_rows)
        self.assertGreaterEqual(len(two.excluded_rows), 3)
        self.assertLess(two.relative_rmse, 1e-5)
        self.assertEqual(comp['n_points'], mask.sum())
        self.assertAlmostEqual(comp['single_weighted_rss'], mask.sum() * one.relative_rmse**2)
        self.assertLess(comp['voigt_aicc'], comp['single_aicc'])
        self.assertLess(impedance_voigt([100], *expected).imag[0], 0)
        self.assertTrue(all(c > 0 for c in two.capacitances))

    def test_swap_covariance_and_ideal_capacitance(self):
        p = np.array([10., 50., 1e-5, 1., 400., 1e-3, 1.])
        swap = p[[0, 4, 5, 6, 1, 2, 3]]
        f = np.geomspace(1e4, .01, 30)
        np.testing.assert_allclose(impedance_voigt(f, *p), impedance_voigt(f, *swap))
        cov = np.diag(np.arange(1., 8.))
        sorted_p, sorted_cov = order_branches(p, cov)
        np.testing.assert_allclose(sorted_p, swap)
        np.testing.assert_allclose(sorted_cov.diagonal(), [1, 5, 6, 7, 2, 3, 4])
        self.assertAlmostEqual(peak_frequency(400, 1e-3, 1), 1/(2*np.pi*.4))
        from speis_fitting import effective_capacitance
        self.assertAlmostEqual(effective_capacitance(400, 1e-3, 1)[0], 1e-3)

    def test_overlapping_out_of_band_and_condition(self):
        s = make_spectrum((10, 300, 1e-4, .85, 100, 3e-4, .85))
        r = fit_voigt(s, n_starts=6, max_condition=1.01, min_peak_decades=20)
        self.assertTrue(r.success)
        self.assertIn('ill_conditioned', r.flags)
        self.assertIn('overlapping_peaks', r.flags)
        self.assertIsNone(r.covariance)
        self.assertEqual(r.capacitance_stds, (None, None))
        out = make_spectrum((12, 400, 2e-4, .87, 100, 2e-6, .9))
        narrow = fit_mask(out, fmin=1000, fmax=1e6)
        r = fit_voigt(out, mask=narrow, n_starts=6)
        self.assertTrue(r.success)
        self.assertTrue({'peak_outside_measured_band', 'ill_conditioned', 'parameter_at_bound',
                         'near_optimal_parameter_ambiguity'} & set(r.flags))

    def test_ambiguous_near_optimal_starts(self):
        s = make_spectrum((10, 300, 1e-4, .85, 100, 3e-4, .85))
        r = fit_voigt(s, n_starts=12)
        self.assertTrue(r.success)
        self.assertGreater(r.near_optimal_starts, 1)
        self.assertIn('near_optimal_parameter_ambiguity', r.flags)
        self.assertIn('peak_outside_measured_band', r.flags)
        self.assertIsNone(r.covariance)

    def test_full_covariance_propagation_on_noisy_fit(self):
        s = make_spectrum()
        rng = np.random.default_rng(42)
        s.data['Re(Z)/Ohm'] += rng.normal(0, .01, len(s.data))
        s.data['-Im(Z)/Ohm'] += rng.normal(0, .01, len(s.data))
        r = fit_voigt(s, n_starts=8)
        self.assertTrue(r.success)
        self.assertIsNotNone(r.covariance)
        for i, start in enumerate((1, 4)):
            rp, q, alpha = r.parameters[start:start+3]
            c = r.capacitances[i]
            grad = np.zeros(7)
            grad[start:start+3] = [c*(1-alpha)/(alpha*rp), c/(alpha*q),
                                   -c*(np.log(q)+np.log(rp))/alpha**2]
            expected_std = np.sqrt(grad @ r.covariance @ grad)
            self.assertAlmostEqual(r.capacitance_stds[i] / expected_std, 1., places=10)

    def test_bounds_mask_failures_and_aicc(self):
        s = make_spectrum()
        r = fit_voigt(s, n_starts=3, max_condition=1.01)
        self.assertIn('ill_conditioned', r.flags)
        self.assertIsNone(r.covariance)
        self.assertTrue(np.all(r.parameters > 0))
        self.assertTrue(np.all(r.parameters[[3, 6]] <= 1))
        with self.assertRaises(ValueError):
            fit_mask(s, fmin=3, fmax=2)
        with self.assertRaises(ValueError):
            fit_voigt(s, mask=np.ones(len(s.data)))
        self.assertTrue(np.isnan(aicc(1., 8, 7)))
        self.assertEqual(aicc(0., 20, 7), -np.inf)
        self.assertFalse(fit_voigt(s, mask=np.zeros(len(s.data), dtype=bool)).success)

    def test_loader_audit_and_row_ids(self):
        frame = pd.DataFrame({'freq/Hz': [0, 10, np.nan, -1, 2, 0],
                              'Re(Z)/Ohm': [0, 1, 1, 1, np.nan, 0],
                              '-Im(Z)/Ohm': [0, 2, 1, 1, 2, 0],
                              '<Ewe>/V': [.1]*6, 'cycle number': [1]*6})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'data.csv'
            frame.to_csv(path, index=False)
            data, counts, audit = load_eis_csv(path, chunksize=2)
        self.assertEqual(data['_row'].tolist(), [1, 4])
        self.assertEqual(audit['_row'].tolist(), [2, 3])
        self.assertEqual(counts['zero_frequency'], 2)
        self.assertEqual(counts['invalid_positive_impedance'], 1)
        self.assertEqual(counts['total'], 6)

    def test_crossing_flag_and_offset_density(self):
        s = make_spectrum()
        r = fit_voigt(s, n_starts=4)
        r.peaks_hz = (1., 2.)
        t = make_spectrum()
        t.spectrum_id = 2
        t.voltage_v = .4
        self.assertEqual(crossing_flags([(s, r), (t, r)]), {1, 2})
        x = np.arange(6.) * .1
        y = 1e15 + x * 4e14 + np.array([1, -1, 0, 2, -2, 0])*1e12
        a = linear_regression(x, y)
        b = linear_regression(x - .2, y)
        self.assertAlmostEqual(a.slope/b.slope, 1.)
        self.assertAlmostEqual(b.x_intercept_v, a.x_intercept_v - .2)
        density = carrier_density(b, area_m2=9e-6, relative_permittivity=38, confirmed=True)
        self.assertAlmostEqual(density['density_cm3'], density['density_m3']*1e-6)
        with self.assertRaises(ValueError):
            carrier_density(b, area_m2=9e-6, relative_permittivity=38, confirmed=False)


if __name__ == '__main__':
    unittest.main()