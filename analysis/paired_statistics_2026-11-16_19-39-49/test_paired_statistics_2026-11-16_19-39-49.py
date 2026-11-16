"""Fixed and exhaustive fixtures for the new paired-analysis implementation."""
import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np
from scipy import stats

SCRIPT = Path(__file__).with_name('analyze_paired_records_2026-11-16_19-39-49.py')
spec = importlib.util.spec_from_file_location('paired_completion', SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class PairedStatisticsTests(unittest.TestCase):
    def test_holm_known_fixture(self):
        np.testing.assert_allclose(analysis.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_holm_preserves_order_and_undefined(self):
        self.assertEqual(analysis.holm([None, 0.5, 0.01]), [None, 1.0, 0.03])

    def test_t_known_fixture(self):
        result = analysis.describe_differences(np.array([1., 2., 3., 4., 5.]))
        self.assertAlmostEqual(result['mean_difference'], 3.)
        self.assertAlmostEqual(result['sample_SD_difference'], np.sqrt(2.5))
        self.assertAlmostEqual(result['paired_t_statistic'], np.sqrt(18))
        self.assertAlmostEqual(result['paired_t_pvalue'], 0.013235599563682695)
        np.testing.assert_allclose(result['t95ci'], [1.036756838522439, 4.963243161477561])

    def test_t_reversal_changes_sign_preserves_p(self):
        a = analysis.describe_differences(np.array([-.1, .3, .4, .2, -.2]))
        b = analysis.describe_differences(-np.array([-.1, .3, .4, .2, -.2]))
        self.assertAlmostEqual(a['mean_difference'], -b['mean_difference'])
        self.assertAlmostEqual(a['paired_t_pvalue'], b['paired_t_pvalue'])
        np.testing.assert_allclose(a['t95ci'], [-b['t95ci'][1], -b['t95ci'][0]])

    def test_zero_variance_t_is_unavailable(self):
        result = analysis.describe_differences(np.ones(5))
        self.assertIsNone(result['paired_t_pvalue'])
        self.assertEqual(result['t95ci'], [1., 1.])
        self.assertEqual(result['exact_Wilcoxon']['pvalue'], 0.0625)

    def test_five_all_positive_exact_minimum(self):
        result = analysis.exact_signed_rank(np.arange(1, 6, dtype=float))
        self.assertEqual(result['statistic'], 0.)
        self.assertEqual(result['pvalue'], 0.0625)

    def test_all_zero_signed_rank(self):
        result = analysis.exact_signed_rank(np.zeros(5))
        self.assertEqual(result['pvalue'], 1.)
        self.assertEqual(result['nonzero_n'], 0)

    def test_exact_distribution_against_bruteforce_ties_and_zeros(self):
        for values in ([1., -2., 2., 0., -4.], [1., 1., -1., -1.], [1., -3., 4., -7., 8.], [0., 0., -2.]):
            x = np.asarray(values)
            x = x[x != 0]
            rank = stats.rankdata(np.abs(x))
            observed = min(float(rank[x > 0].sum()), float(rank[x < 0].sum()))
            outcomes = [sum(r for r, sign in zip(rank, signs) if sign) for signs in itertools.product([0, 1], repeat=len(x))]
            p = min(1., 2 * sum(v <= observed for v in outcomes) / len(outcomes))
            actual = analysis.exact_signed_rank(np.asarray(values))
            self.assertEqual(actual['statistic'], observed)
            self.assertEqual(actual['pvalue'], p)

    def test_exact_distribution_agrees_with_scipy_no_ties(self):
        values = np.array([.12, -.42, .39, .14, -.17])
        expected = stats.wilcoxon(values, zero_method='wilcox', alternative='two-sided', method='exact')
        actual = analysis.exact_signed_rank(values)
        self.assertEqual(actual['statistic'], expected.statistic)
        self.assertEqual(actual['pvalue'], expected.pvalue)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            analysis.describe_differences(np.array([1., np.nan]))
        with self.assertRaises(ValueError):
            analysis.exact_signed_rank(np.array([np.inf]))


if __name__ == '__main__':
    unittest.main()
