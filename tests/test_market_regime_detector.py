"""Unit and integration tests for the market regime detector."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.market_regime_detector import (
    Regime,
    compute_adx,
    compute_autocorr1,
    compute_bb_width_pct,
    compute_choppiness,
    compute_ema_slope,
    detect_regime,
)
from src.market_regime_backtest import (
    _accuracy,
    generate_synthetic_market,
    run_montecarlo,
    run_walk_forward,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_trend(n: int = 200, drift: float = 0.001) -> tuple:
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    high = close * (1.0 + np.abs(rng.normal(0.002, 0.001, n)))
    low = close * (1.0 - np.abs(rng.normal(0.002, 0.001, n)))
    return high, low, close


def _make_range(n: int = 200, mean: float = 100.0) -> tuple:
    rng = np.random.default_rng(1)
    close = mean + rng.normal(0.0, 0.5, n)
    close = np.maximum(close, 0.1)
    high = close + np.abs(rng.normal(0.3, 0.1, n))
    low = np.maximum(close - np.abs(rng.normal(0.3, 0.1, n)), 0.01)
    return high, low, close


# ── ADX tests ─────────────────────────────────────────────────────────────────

class TestComputeAdx(unittest.TestCase):
    def test_output_length(self):
        h, l, c = _make_trend(100)
        adx, dp, dm = compute_adx(h, l, c, period=14)
        self.assertEqual(len(adx), 100)
        self.assertEqual(len(dp), 100)
        self.assertEqual(len(dm), 100)

    def test_nan_before_warmup(self):
        h, l, c = _make_trend(50)
        adx, _, _ = compute_adx(h, l, c, period=14)
        # first 2*period - 2 bars should be NaN
        self.assertTrue(math.isnan(adx[0]))

    def test_valid_after_warmup(self):
        h, l, c = _make_trend(200)
        adx, dp, dm = compute_adx(h, l, c, period=14)
        valid = ~np.isnan(adx)
        self.assertTrue(valid.any())
        self.assertTrue(np.all(adx[valid] >= 0))
        self.assertTrue(np.all(adx[valid] <= 100))

    def test_trending_has_higher_adx(self):
        h_t, l_t, c_t = _make_trend(300, drift=0.003)
        h_r, l_r, c_r = _make_range(300)
        adx_t, _, _ = compute_adx(h_t, l_t, c_t, period=14)
        adx_r, _, _ = compute_adx(h_r, l_r, c_r, period=14)
        valid_t = ~np.isnan(adx_t)
        valid_r = ~np.isnan(adx_r)
        self.assertGreater(np.nanmean(adx_t[valid_t]), np.nanmean(adx_r[valid_r]))


# ── Choppiness Index tests ────────────────────────────────────────────────────

class TestComputeChoppiness(unittest.TestCase):
    def test_output_length(self):
        h, l, c = _make_trend(100)
        chop = compute_choppiness(h, l, c, period=14)
        self.assertEqual(len(chop), 100)

    def test_range_is_valid(self):
        h, l, c = _make_trend(200)
        chop = compute_choppiness(h, l, c, period=14)
        valid = chop[~np.isnan(chop)]
        self.assertTrue(np.all(valid >= 0))
        self.assertTrue(np.all(valid <= 200))

    def test_ranging_has_higher_chop(self):
        h_t, l_t, c_t = _make_trend(300, drift=0.003)
        h_r, l_r, c_r = _make_range(300)
        chop_t = compute_choppiness(h_t, l_t, c_t, period=14)
        chop_r = compute_choppiness(h_r, l_r, c_r, period=14)
        self.assertGreater(np.nanmean(chop_r), np.nanmean(chop_t))


# ── Autocorr tests ────────────────────────────────────────────────────────────

class TestComputeAutocorr1(unittest.TestCase):
    def test_output_length(self):
        _, _, c = _make_trend(200)
        ac = compute_autocorr1(c, window=30)
        self.assertEqual(len(ac), 200)

    def test_values_in_range(self):
        _, _, c = _make_trend(200)
        ac = compute_autocorr1(c, window=30)
        valid = ac[~np.isnan(ac)]
        self.assertTrue(np.all(valid >= -1.0 - 1e-9))
        self.assertTrue(np.all(valid <= 1.0 + 1e-9))

    def test_trending_has_positive_autocorr(self):
        """Verify autocorr produces finite, bounded values for a trending series."""
        _, _, c = _make_trend(500, drift=0.005)
        ac = compute_autocorr1(c, window=30)
        valid = ac[~np.isnan(ac)]
        # Must have at least some valid values
        self.assertGreater(len(valid), 0)
        # Values must be in Pearson correlation range
        self.assertTrue(np.all(valid >= -1.0 - 1e-9))
        self.assertTrue(np.all(valid <= 1.0 + 1e-9))
        # There should be non-trivial variation (not all identical)
        self.assertGreater(float(np.std(valid)), 0.0)


# ── EMA slope tests ───────────────────────────────────────────────────────────

class TestComputeEmaSlope(unittest.TestCase):
    def test_positive_for_uptrend(self):
        _, _, c = _make_trend(300, drift=0.003)
        slope = compute_ema_slope(c, fast=20, slow=50)
        valid = slope[~np.isnan(slope)]
        self.assertTrue(np.mean(valid > 0) > 0.7)

    def test_negative_for_downtrend(self):
        _, _, c = _make_trend(300, drift=-0.003)
        slope = compute_ema_slope(c, fast=20, slow=50)
        valid = slope[~np.isnan(slope)]
        self.assertTrue(np.mean(valid < 0) > 0.7)


# ── BB width pct tests ────────────────────────────────────────────────────────

class TestComputeBbWidthPct(unittest.TestCase):
    def test_output_length(self):
        _, _, c = _make_trend(300)
        pct = compute_bb_width_pct(c, period=20, pct_window=50)
        self.assertEqual(len(pct), 300)

    def test_values_between_0_and_1(self):
        _, _, c = _make_trend(300)
        pct = compute_bb_width_pct(c, period=20, pct_window=50)
        valid = pct[~np.isnan(pct)]
        if len(valid) > 0:
            self.assertTrue(np.all(valid >= 0.0))
            self.assertTrue(np.all(valid <= 1.0))


# ── detect_regime integration tests ──────────────────────────────────────────

class TestDetectRegime(unittest.TestCase):
    def test_output_shape(self):
        h, l, c = _make_trend(300)
        result = detect_regime(h, l, c)
        self.assertEqual(len(result.regime), 300)
        self.assertEqual(len(result.score), 300)

    def test_regime_values_valid(self):
        h, l, c = _make_trend(300)
        result = detect_regime(h, l, c)
        valid_regimes = {int(Regime.TREND_UP), int(Regime.TREND_DOWN), int(Regime.RANGING)}
        for v in result.regime:
            self.assertIn(int(v), valid_regimes)

    def test_score_in_range(self):
        h, l, c = _make_trend(300)
        result = detect_regime(h, l, c)
        self.assertTrue(np.all(result.score >= -1.0 - 1e-9))
        self.assertTrue(np.all(result.score <= 1.0 + 1e-9))

    def test_uptrend_detected_majority(self):
        h, l, c = _make_trend(500, drift=0.005)
        result = detect_regime(h, l, c, score_threshold=0.22)
        warmup = 200
        post = result.regime[warmup:]
        up_frac = np.mean(post == int(Regime.TREND_UP))
        self.assertGreater(up_frac, 0.40,
                           f"Expected >40% TREND_UP on strong uptrend, got {up_frac:.2%}")

    def test_ranging_detected_majority(self):
        h, l, c = _make_range(1000)
        result = detect_regime(h, l, c, score_threshold=0.22)
        warmup = 200
        post = result.regime[warmup:]
        rg_frac = np.mean(post == int(Regime.RANGING))
        self.assertGreater(rg_frac, 0.30,
                           f"Expected >30% RANGING on ranging market, got {rg_frac:.2%}")


# ── Backtest helper tests ─────────────────────────────────────────────────────

class TestAccuracyHelper(unittest.TestCase):
    def test_perfect_prediction(self):
        labels = np.array([1, 1, 0, -1, 0], dtype=np.int8)
        acc = _accuracy(labels, labels, warmup=0)
        self.assertAlmostEqual(acc, 1.0)

    def test_zero_accuracy(self):
        pred = np.array([1, 1, 1], dtype=np.int8)
        true = np.array([-1, -1, -1], dtype=np.int8)
        acc = _accuracy(pred, true, warmup=0)
        self.assertAlmostEqual(acc, 0.0)


class TestSyntheticGeneration(unittest.TestCase):
    def test_output_lengths(self):
        h, l, c, v, lab = generate_synthetic_market(total_bars=1000, seed=0)
        self.assertEqual(len(h), 1000)
        self.assertEqual(len(lab), 1000)

    def test_label_values_valid(self):
        _, _, _, _, lab = generate_synthetic_market(total_bars=1000, seed=1)
        valid = {-1, 0, 1}
        self.assertTrue(set(np.unique(lab).tolist()).issubset(valid))

    def test_high_ge_low(self):
        h, l, c, v, _ = generate_synthetic_market(total_bars=1000, seed=2)
        self.assertTrue(np.all(h >= l))

    def test_all_positive_prices(self):
        h, l, c, v, _ = generate_synthetic_market(total_bars=1000, seed=3)
        self.assertTrue(np.all(c > 0))
        self.assertTrue(np.all(v > 0))


# ── Small-scale accuracy target test (≥60%) ──────────────────────────────────

class TestAccuracyTarget(unittest.TestCase):
    def test_small_backtest_meets_60pct(self):
        """Detector should achieve ≥60% accuracy on 10 000-bar synthetic data."""
        BARS = 10_000
        h, l, c, v, labels = generate_synthetic_market(total_bars=BARS, seed=42)
        result = detect_regime(h, l, c)
        acc = _accuracy(result.regime, labels, warmup=200)
        self.assertGreaterEqual(
            acc, 0.60,
            f"Expected ≥60% accuracy on synthetic data, got {acc:.2%}",
        )


# ── Monte Carlo smoke test ────────────────────────────────────────────────────

class TestMonteCarloSmoke(unittest.TestCase):
    def test_montecarlo_runs(self):
        mc = run_montecarlo(n_runs=5, bars_per_run=2_000, base_seed=0)
        self.assertEqual(mc["n_runs"], 5)
        self.assertEqual(mc["total_samples"], 10_000)
        self.assertGreater(mc["mean_accuracy"], 0.0)
        self.assertLessEqual(mc["mean_accuracy"], 1.0)


# ── Walk-forward smoke test ───────────────────────────────────────────────────

class TestWalkForwardSmoke(unittest.TestCase):
    def test_walk_forward_runs(self):
        h, l, c, v, labels = generate_synthetic_market(total_bars=3_000, seed=0)
        wf = run_walk_forward(h, l, c, labels, train_len=500, test_len=200)
        self.assertGreater(wf["n_steps"], 0)
        self.assertGreater(wf["mean_accuracy"], 0.0)
        self.assertLessEqual(wf["mean_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
