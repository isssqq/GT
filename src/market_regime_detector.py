"""Advanced market regime detector: trending vs ranging classification.

Indicators:
  1. ADX / DI+DI-   – trend strength and direction
  2. Choppiness Index – range/trend separation
  3. Lag-1 autocorrelation of log-returns – Hurst proxy (persistence)
  4. EMA slope ratio (fast vs slow)
  5. Bollinger Band width percentile

Regime scoring:
  score ∈ [-1, 1]: positive → TREND_UP, negative → TREND_DOWN, ~0 → RANGING
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np
from scipy.signal import lfilter


class Regime(IntEnum):
    TREND_DOWN = -1
    RANGING = 0
    TREND_UP = 1


@dataclass(frozen=True)
class RegimeResult:
    regime: np.ndarray        # int8, Regime values
    score: np.ndarray         # float64 in [-1, 1]
    adx: np.ndarray
    choppiness: np.ndarray
    autocorr: np.ndarray      # lag-1 autocorrelation
    ema_slope: np.ndarray
    bb_width_pct: np.ndarray


# ── Thresholds (documented so Pine translation is obvious) ──────────────────
ADX_TREND: float = 25.0
ADX_RANGE: float = 20.0
CHOP_TREND: float = 38.2
CHOP_RANGE: float = 61.8
SCORE_THRESHOLD: float = 0.22


# ── Low-level helpers ────────────────────────────────────────────────────────

def _wilder_smooth(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothed MA (RMA) via scipy IIR – O(n)."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < period:
        return out
    # Find first window of `period` consecutive non-NaN values
    alpha = 1.0 / period
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    for start in range(n - period + 1):
        window = x[start: start + period]
        if not np.any(np.isnan(window)):
            y0 = float(np.mean(window))
            out[start + period - 1] = y0
            if n > start + period:
                zi = np.array([(1.0 - alpha) * y0])
                out[start + period:], _ = lfilter(b, a, x[start + period:], zi=zi)
            break
    return out


def _ema(x: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average via scipy IIR – O(n)."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < period:
        return out
    y0 = float(np.mean(x[:period]))
    out[period - 1] = y0
    if n > period:
        alpha = 2.0 / (period + 1)
        b = [alpha]
        a = [1.0, -(1.0 - alpha)]
        zi = np.array([(1.0 - alpha) * y0])
        out[period:], _ = lfilter(b, a, x[period:], zi=zi)
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    hl = high - low
    hc = np.abs(np.concatenate(([hl[0]], high[1:] - close[:-1])))
    lc = np.abs(np.concatenate(([0.0], low[1:] - close[:-1])))
    return np.maximum(hl, np.maximum(hc, lc))


# ── Indicator functions ──────────────────────────────────────────────────────

def compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (adx, di_plus, di_minus)."""
    n = len(high)
    tr = _true_range(high, low, close)

    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    dm_plus[1:] = np.where((up > down) & (up > 0), up, 0.0)
    dm_minus[1:] = np.where((down > up) & (down > 0), down, 0.0)

    atr = _wilder_smooth(tr, period)
    sdm_plus = _wilder_smooth(dm_plus, period)
    sdm_minus = _wilder_smooth(dm_minus, period)

    eps = 1e-12
    di_plus = 100.0 * sdm_plus / np.maximum(atr, eps)
    di_minus = 100.0 * sdm_minus / np.maximum(atr, eps)

    dx_num = np.abs(di_plus - di_minus)
    dx_den = di_plus + di_minus
    dx = np.where(dx_den > eps, 100.0 * dx_num / dx_den, 0.0)
    nan_mask = np.isnan(di_plus) | np.isnan(di_minus)
    dx[nan_mask] = np.nan

    adx = _wilder_smooth(dx, period)
    return adx, di_plus, di_minus


def compute_choppiness(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Choppiness Index ∈ [0, 100]. Low (<38.2) → trending, high (>61.8) → ranging."""
    from numpy.lib.stride_tricks import sliding_window_view

    tr = _true_range(high, low, close)
    n = len(tr)
    result = np.full(n, np.nan)
    if n < period:
        return result

    log_n = np.log10(float(period))
    # Rolling TR sum via cumsum – O(n)
    cum_tr = np.concatenate(([0.0], np.cumsum(tr)))
    tr_sum = cum_tr[period:] - cum_tr[:n - period + 1]  # shape (n-period+1,)

    # Rolling high/low via sliding window (no copy – only a view)
    hh = sliding_window_view(high, period).max(axis=1)  # (n-period+1,)
    ll = sliding_window_view(low, period).min(axis=1)

    denom = hh - ll
    valid = denom > 1e-12
    ci = np.full(n - period + 1, np.nan)
    ci[valid] = 100.0 * np.log10(np.maximum(tr_sum[valid] / denom[valid], 1e-12)) / log_n
    result[period - 1:] = ci
    return result


def compute_autocorr1(close: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling lag-1 autocorrelation of log-returns via cumsum – O(n).

    Values > 0 indicate trending (H > 0.5); < 0 indicate mean-reversion (H < 0.5).
    """
    n = len(close)
    result = np.full(n, np.nan)
    log_r = np.diff(np.log(np.maximum(close, 1e-12)))  # length n-1
    m = len(log_r)
    w = window - 1  # Pearson window on lagged pairs
    if m < window:
        return result

    x = log_r[:-1]   # length m-1
    y = log_r[1:]    # length m-1
    pairs = len(x)   # m-1 = n-2

    if pairs < w:
        return result

    # Cumulative sums for O(n) rolling Pearson correlation
    c0 = np.zeros(1)
    cum_x = np.concatenate((c0, np.cumsum(x)))
    cum_y = np.concatenate((c0, np.cumsum(y)))
    cum_xy = np.concatenate((c0, np.cumsum(x * y)))
    cum_x2 = np.concatenate((c0, np.cumsum(x * x)))
    cum_y2 = np.concatenate((c0, np.cumsum(y * y)))

    j = np.arange(w - 1, pairs)  # ending pair indices
    sx = cum_x[j + 1] - cum_x[j - w + 1]
    sy = cum_y[j + 1] - cum_y[j - w + 1]
    sxy = cum_xy[j + 1] - cum_xy[j - w + 1]
    sx2 = cum_x2[j + 1] - cum_x2[j - w + 1]
    sy2 = cum_y2[j + 1] - cum_y2[j - w + 1]

    cov_num = w * sxy - sx * sy
    var_x = np.maximum(w * sx2 - sx * sx, 0.0)
    var_y = np.maximum(w * sy2 - sy * sy, 0.0)
    denom = np.sqrt(var_x * var_y)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 1e-12, cov_num / np.maximum(denom, 1e-12), 0.0)

    # pair ending at j → last return index j+1 → bar index j+2
    start_bar = w + 1  # = (w-1) + 2
    result[start_bar: start_bar + len(corr)] = corr
    return result


def compute_ema_slope(close: np.ndarray, fast: int = 20, slow: int = 50) -> np.ndarray:
    """(fast_ema − slow_ema) / slow_ema, i.e. relative spread."""
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    eps = 1e-12
    result = np.full(len(close), np.nan)
    valid = ~np.isnan(fast_ema) & ~np.isnan(slow_ema) & (slow_ema > eps)
    result[valid] = (fast_ema[valid] - slow_ema[valid]) / slow_ema[valid]
    return result


def compute_bb_width_pct(close: np.ndarray, period: int = 20, pct_window: int = 100) -> np.ndarray:
    """Bollinger Band width percentile over `pct_window` bars.

    Near 1 → volatility expanding (trend-like); near 0 → compressed (range-like).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(close)
    bb_width = np.full(n, np.nan)
    result = np.full(n, np.nan)

    if n < period:
        return result

    wins = sliding_window_view(close, period)          # (n-period+1, period)
    means = wins.mean(axis=1)
    stds = wins.std(axis=1, ddof=1)
    bb_width[period - 1:] = 2.0 * stds / np.maximum(means, 1e-12)

    start = period - 1 + pct_window - 1               # first bar with full pct window
    if start >= n:
        return result

    bw_wins = sliding_window_view(bb_width[period - 1:], pct_window)  # (k, pct_window)
    cur = bb_width[start:]
    pct = np.sum(bw_wins <= cur[:, np.newaxis], axis=1) / float(pct_window)
    result[start:start + len(pct)] = pct
    return result


# ── Main detection function ──────────────────────────────────────────────────

def detect_regime(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    adx_period: int = 14,
    chop_period: int = 14,
    ema_fast: int = 20,
    ema_slow: int = 50,
    bb_period: int = 20,
    autocorr_window: int = 50,
    score_threshold: float = SCORE_THRESHOLD,
) -> RegimeResult:
    """Detect market regime using a multi-indicator scoring system.

    Each indicator contributes a component in [-1, 1]:
      • Trend-strength indicators (ADX, Choppiness, autocorrelation, BB width)
        are combined to a scalar in [0, 1].
      • Direction indicators (DI+/DI- spread, EMA slope sign) determine the sign.
      • final_score = strength × direction
      • |score| > score_threshold → TREND_UP/DOWN; else RANGING.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)

    adx, di_plus, di_minus = compute_adx(high, low, close, adx_period)
    chop = compute_choppiness(high, low, close, chop_period)
    autocorr = compute_autocorr1(close, autocorr_window)
    ema_slope = compute_ema_slope(close, ema_fast, ema_slow)
    bb_pct = compute_bb_width_pct(close, bb_period)

    # ── Trend-strength component (0 → ranging, 1 → strongly trending) ────────
    # ADX: map [15, 50] → [0, 1]  (generous lower bound to capture moderate trends)
    adx_s = np.clip((adx - 15.0) / 35.0, 0.0, 1.0)
    adx_s[np.isnan(adx)] = np.nan

    # Choppiness: map [CHOP_RANGE, CHOP_TREND] → [0, 1] (inverted)
    chop_s = np.clip((CHOP_RANGE - chop) / (CHOP_RANGE - CHOP_TREND), 0.0, 1.0)
    chop_s[np.isnan(chop)] = np.nan

    # Autocorr lag-1: |autocorr| mapped to [0, 1]
    autocorr_s = np.clip(np.abs(autocorr) * 2.0, 0.0, 1.0)
    autocorr_s[np.isnan(autocorr)] = np.nan

    # BB width percentile is already [0, 1]
    bb_s = np.where(np.isnan(bb_pct), np.nan, bb_pct)

    # Core strength requires BOTH ADX and Choppiness to agree (AND gate via min).
    # Secondary indicators (autocorr, BB) act as bonuses.
    adx_v = np.where(np.isnan(adx_s), 0.0, adx_s)
    chop_v = np.where(np.isnan(chop_s), 0.0, chop_s)
    ac_v = np.where(np.isnan(autocorr_s), 0.0, autocorr_s)
    bb_v = np.where(np.isnan(bb_s), 0.0, bb_s)

    core = np.minimum(adx_v, chop_v)                   # both must agree
    bonus = 0.15 * ac_v + 0.15 * bb_v
    strength = np.clip(0.70 * core + bonus, 0.0, 1.0)

    # ── Direction component ────────────────────────────────────────────────────
    # DI spread (normalised to [-1, 1])
    di_spread = di_plus - di_minus
    di_spread_norm = di_spread / np.maximum(np.abs(di_spread), 1e-12)
    di_spread_norm[np.isnan(di_spread)] = 0.0

    # EMA slope sign
    ema_sign = np.sign(np.where(np.isnan(ema_slope), 0.0, ema_slope))

    # Weighted direction: DI 60%, EMA 40%
    di_valid = (~np.isnan(di_plus)).astype(float)
    ema_valid = (~np.isnan(ema_slope)).astype(float)
    dir_total = 0.6 * di_valid + 0.4 * ema_valid
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = np.where(
            dir_total > 0,
            (0.6 * di_valid * di_spread_norm + 0.4 * ema_valid * ema_sign) / np.maximum(dir_total, 1e-12),
            0.0,
        )

    # ── Choppiness gate: suppress score when CI says clearly ranging ──────────
    # CI > 55 → multiply score by 0.25 (strong ranging suppression)
    # CI < 45 → full score
    # CI ∈ [45, 55] → linearly interpolate
    chop_gate = np.where(
        np.isnan(chop),
        1.0,
        np.clip((55.0 - chop) / 10.0, 0.25, 1.0),
    )

    # ── Final score and regime ────────────────────────────────────────────────
    score = np.clip(strength * direction * chop_gate, -1.0, 1.0)

    regime = np.zeros(n, dtype=np.int8)
    regime[score > score_threshold] = int(Regime.TREND_UP)
    regime[score < -score_threshold] = int(Regime.TREND_DOWN)

    return RegimeResult(
        regime=regime,
        score=score,
        adx=adx,
        choppiness=chop,
        autocorr=autocorr,
        ema_slope=ema_slope,
        bb_width_pct=bb_pct,
    )
