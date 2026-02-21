"""Minimal VWAP Auto Anchored Z-Score reference implementation.

Design notes:
- Auto anchor uses the most recent confirmed pivot (high/low/both).
- Anchored VWAP starts from that pivot bar.
- Z-score uses volume-weighted variance of typical price around anchored VWAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class VwapAutoAnchoredResult:
    anchor_index: int
    vwap: List[float]
    zscore: List[float]


def _find_pivots(values: Sequence[float], left: int, right: int, is_high: bool) -> List[int]:
    pivots: List[int] = []
    n = len(values)
    for i in range(left, n - right):
        center = values[i]
        ok = True
        for j in range(i - left, i + right + 1):
            if j == i:
                continue
            if is_high and values[j] >= center:
                ok = False
                break
            if not is_high and values[j] <= center:
                ok = False
                break
        if ok:
            pivots.append(i)
    return pivots


def find_auto_anchor(
    high: Sequence[float],
    low: Sequence[float],
    left: int = 5,
    right: int = 5,
    mode: str = "both",
) -> int:
    if len(high) != len(low):
        raise ValueError("high and low must have the same length")
    if len(high) == 0:
        raise ValueError("input series cannot be empty")

    candidates: List[int] = []
    if mode in ("high", "both"):
        candidates.extend(_find_pivots(high, left, right, is_high=True))
    if mode in ("low", "both"):
        candidates.extend(_find_pivots(low, left, right, is_high=False))

    if not candidates:
        return 0
    return max(candidates)


def compute_vwap_auto_anchored_zscore(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    left: int = 5,
    right: int = 5,
    mode: str = "both",
    eps: float = 1e-12,
) -> VwapAutoAnchoredResult:
    n = len(close)
    if not (n == len(high) == len(low) == len(volume)):
        raise ValueError("all input series must have the same length")
    if n == 0:
        raise ValueError("input series cannot be empty")

    anchor = find_auto_anchor(high, low, left=left, right=right, mode=mode)

    typical = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close)]

    vwap = [float("nan")] * n
    zscore = [float("nan")] * n

    w_sum = 0.0
    wx_sum = 0.0
    for i in range(anchor, n):
        w = max(volume[i], 0.0)
        x = typical[i]
        w_sum += w
        wx_sum += w * x
        mean = wx_sum / max(w_sum, eps)
        vwap[i] = mean

        var_num = 0.0
        ww = 0.0
        for j in range(anchor, i + 1):
            wj = max(volume[j], 0.0)
            dj = typical[j] - mean
            var_num += wj * dj * dj
            ww += wj
        std = sqrt(var_num / max(ww, eps))
        zscore[i] = (close[i] - mean) / max(std, eps)

    return VwapAutoAnchoredResult(anchor_index=anchor, vwap=vwap, zscore=zscore)


def _to_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]


if __name__ == "__main__":
    # Tiny smoke example.
    h = _to_list([10, 11, 13, 12, 11, 12, 13, 14, 13, 12, 11, 10, 11, 12])
    l = _to_list([9, 10, 11, 10, 9, 10, 11, 12, 11, 10, 9, 8, 9, 10])
    c = _to_list([9.5, 10.5, 12.5, 11, 10, 11, 12.5, 13.5, 12, 11, 10, 9, 10, 11])
    v = _to_list([100, 120, 150, 130, 110, 115, 160, 180, 140, 135, 120, 110, 105, 100])
    r = compute_vwap_auto_anchored_zscore(h, l, c, v, left=2, right=2, mode="both")
    print(f"anchor_index={r.anchor_index}, last_vwap={r.vwap[-1]:.4f}, last_z={r.zscore[-1]:.4f}")
