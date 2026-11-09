"""Synthetic tests for the independent exact-frequency workflow."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from single_frequency import PhaseGate, audit_frequencies, extract_cut, load_spectra, regress_cut
from speis_fitting import group_spectra, carrier_density


def make_sweeps(freqs=(100000., 80000., 60000., 40000.), phases=None, count=4):
    phases = phases or [-84.] * len(freqs)
    parts = []
    for i in range(count):
        z = (100 + i * 5) * np.exp(1j * np.deg2rad(phases))
        parts.append(pd.DataFrame({'freq/Hz': freqs, 'Re(Z)/Ohm': z.real, '-Im(Z)/Ohm': -z.imag,
                                   '<Ewe>/V': i * .1, 'cycle number': i + 1,
                                   'Phase(Z)/deg': phases}))
    return group_spectra(pd.concat(parts, ignore_index=True))


class AuditTests(unittest.TestCase):
    def test_exact_common_rank_and_phase(self):
        sweeps = make_sweeps()
        table, details, common = audit_frequencies(sweeps)
        self.assertEqual(common, [40000., 60000., 80000., 100000.])
        self.assertTrue(table.passes.all())
        self.assertEqual(table.frequency_hz.tolist(), [60000., 80000., 100000.])
        self.assertTrue((details.instrument_phase_difference_deg < 1e-10).all())
        self.assertTrue(details.neighbor_pass.all())

    def test_missing_near_and_duplicates_never_match(self):
        sweeps = make_sweeps(count=2)
        sweeps[1].data['freq/Hz'] += .01
        table, _, common = audit_frequencies(sweeps)
        self.assertEqual(common, [])
        self.assertFalse(table.passes.any())
        self.assertTrue((table.missing_sweeps == 1).all())
        sweeps = make_sweeps(count=2)
        duplicate = sweeps[0].data.loc[sweeps[0].data['freq/Hz'] == 80000].copy()
        sweeps[0].data = pd.concat([sweeps[0].data, duplicate])
        table, _, common = audit_frequencies(sweeps)
        self.assertIn(80000., common)
        row = table.set_index('frequency_hz').loc[80000]
        self.assertEqual(row.duplicate_sweeps, 1)
        self.assertFalse(row.passes)

    def test_no_plateau_or_bad_point(self):
        sweeps = make_sweeps(phases=[-70., -85., -70., -60.])
        table, _, _ = audit_frequencies(sweeps)
        self.assertFalse(table.passes.any())
        self.assertEqual(table.set_index('frequency_hz').loc[80000., 'neighbor_fail_sweeps'], 4)
        sweeps[0].data.loc[sweeps[0].data['freq/Hz'] == 80000, 'Re(Z)/Ohm'] = np.nan
        table, _, _ = audit_frequencies(sweeps)
        self.assertEqual(table.set_index('frequency_hz').loc[80000., 'invalid_sweeps'], 1)

    def test_empty_and_configuration(self):
        table, _, common = audit_frequencies([])
        self.assertTrue(table.empty)
        self.assertEqual(common, [])
        with self.assertRaises(ValueError):
            PhaseGate(low_hz=-1)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.sweeps = make_sweeps()
        self.table, _, _ = audit_frequencies(self.sweeps)

    def test_confirmation_and_formulas(self):
        with self.assertRaises(ValueError):
            extract_cut(self.sweeps, self.table, 80000)
        with self.assertRaises(ValueError):
            extract_cut(self.sweeps, self.table, 80000.01, confirmed=True)
        cut = extract_cut(self.sweeps, self.table, 80000, confirmed=True,
                          series_rs_ohm=4, rs_source='independent test', rs_uncertainty_ohm=1)
        row = cut.iloc[0]
        z = complex(row.Re_ohm, -row.neg_Im_ohm)
        omega = 2*np.pi*80000
        self.assertAlmostEqual(row.Cp_F, (1/z).imag/omega)
        self.assertAlmostEqual(row.Cs_F, -1/(omega*z.imag))
        self.assertAlmostEqual(row.Cp_Rs_corrected_F, (1/(z-4)).imag/omega)
        self.assertTrue(cut.eligible.all())
        np.testing.assert_allclose(cut.bias_corrected_v, cut.bias_measured_v - .2)
        with self.assertRaises(ValueError):
            extract_cut(self.sweeps, self.table, 80000, confirmed=True, series_rs_ohm=5)

    def test_noncapacitive_and_zero_mask(self):
        self.sweeps[0].data.loc[self.sweeps[0].data['freq/Hz'] == 80000, '-Im(Z)/Ohm'] = -1
        self.sweeps[1].data.loc[self.sweeps[1].data['freq/Hz'] == 80000, ['Re(Z)/Ohm', '-Im(Z)/Ohm']] = 0
        # A stale audit cannot authorize changed or noncapacitive points.
        with self.assertRaisesRegex(ValueError, 'Data changed'):
            extract_cut(self.sweeps, self.table, 80000, confirmed=True)

    def test_regression_confirmation_range_offset_and_density(self):
        cut = extract_cut(self.sweeps, self.table, 80000, confirmed=True)
        with self.assertRaises(ValueError):
            regress_cut(cut, (-.2, .2))
        mask, fit = regress_cut(cut, (-.2, .2), confirmed=True)
        self.assertEqual(mask.sum(), 4)
        shifted = cut.copy()
        shifted['bias_corrected_v'] += .2
        _, fit2 = regress_cut(shifted, (0, .4), confirmed=True)
        self.assertAlmostEqual(fit.slope/fit2.slope, 1)
        self.assertAlmostEqual(fit2.x_intercept_v - fit.x_intercept_v, .2)
        density = carrier_density(fit, area_m2=9e-6, relative_permittivity=38, confirmed=True)
        self.assertAlmostEqual(density['density_cm3']/density['density_m3'], 1e-6)
        with self.assertRaises(ValueError):
            regress_cut(cut, (0, .01), confirmed=True)


class LoadingTests(unittest.TestCase):
    def test_chunking_and_invalid_audit(self):
        df = pd.DataFrame({'freq/Hz': [0, 100000, 80000, -1, 0], 'Re(Z)/Ohm': [0, 1, np.nan, 3, 5],
                           '-Im(Z)/Ohm': [0, 2, 1, 2, 2], '<Ewe>/V': [.1]*5, 'cycle number': [1]*5})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'data.csv'
            df.to_csv(path, index=False)
            sweeps, bad, stats = load_spectra(path, chunksize=2)
        self.assertEqual(stats['total_rows'], 5)
        self.assertEqual(stats['positive_frequency_rows'], 2)
        self.assertEqual(stats['positive_rows_invalid_impedance_or_voltage'], 1)
        self.assertEqual(bad['_row'].tolist(), [3, 4])
        self.assertEqual(sweeps[0].data['_row'].tolist(), [1, 2])


if __name__ == '__main__':
    unittest.main()