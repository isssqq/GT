"""Market regime detector backtest.

Generates ≥10 000 000 synthetic OHLCV bars with ground-truth regime labels,
applies the detector, and reports accuracy across train/val/retrain/out-of-sample
splits.  Also runs Monte Carlo simulations and walk-forward validation.

All synthetic regimes are drawn from:
  • GBM positive drift  → TREND_UP
  • GBM negative drift  → TREND_DOWN
  • Ornstein-Uhlenbeck  → RANGING
  • False breakout       → RANGING  (brief apparent trend that reverts)
  • Manipulation         → RANGING  (sudden spike/crash that reverts quickly)

Run directly:
    PYTHONPATH=. python src/market_regime_backtest.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from market_regime_detector import Regime, RegimeResult, detect_regime
except ModuleNotFoundError:
    from src.market_regime_detector import Regime, RegimeResult, detect_regime

# ── Simulation parameters ─────────────────────────────────────────────────────
TOTAL_BARS: int = 10_000_000
SEED: int = 42

# Monte Carlo
MC_RUNS: int = 100
MC_BARS: int = 100_000            # 100 × 100 K = 10 M total

# Walk-forward
WF_TRAIN_LEN: int = 5_000
WF_TEST_LEN: int = 500

# Data splits  (train 20% / val 20% / retrain 20% / out-of-sample 40%)
SPLIT_TRAIN: float = 0.20
SPLIT_VAL: float = 0.20
SPLIT_RETRAIN: float = 0.20

# Detector warmup (bars before indicators are fully initialised)
WARMUP: int = 200


# ── Regime codes (int8) ──────────────────────────────────────────────────────
_UP: int = int(Regime.TREND_UP)
_DN: int = int(Regime.TREND_DOWN)
_RG: int = int(Regime.RANGING)


# ── Synthetic data generation ─────────────────────────────────────────────────

def _gbm_segment(
    n: int,
    start_price: float,
    drift: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Geometric Brownian Motion prices (close-only)."""
    dt = 1.0 / 252.0
    log_returns = rng.normal(drift * dt, sigma * np.sqrt(dt), n)
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    return np.exp(log_prices)


def _ou_segment(
    n: int,
    start_price: float,
    mu: float,
    theta: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Ornstein-Uhlenbeck prices (mean-reverting)."""
    prices = np.empty(n)
    prices[0] = start_price
    noise = rng.normal(0.0, sigma, n)
    for i in range(1, n):
        prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + noise[i]
        prices[i] = max(prices[i], 0.01)
    return prices


def _false_breakout_segment(
    n: int,
    start_price: float,
    direction: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Brief apparent trend (≤n/3 bars) that fully reverts – labelled RANGING."""
    burst = max(3, n // 4)
    drift_up = 0.30 * direction   # annualised
    sigma = 0.15
    dt = 1.0 / 252.0
    burst_prices = _gbm_segment(burst, start_price, drift_up, sigma, rng)
    peak = float(burst_prices[-1])
    # revert linearly back to near start_price in remaining bars
    revert = np.linspace(peak, start_price * rng.uniform(0.97, 1.03), n - burst)
    return np.concatenate([burst_prices, revert])


def _manipulation_segment(
    n: int,
    start_price: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sudden spike (3-5 σ) then rapid reversion – labelled RANGING."""
    spike = int(max(2, n * 0.05))
    tail = n - spike
    direction = rng.choice([-1, 1])
    spike_mag = rng.uniform(3.0, 5.0) * 0.01 * start_price
    peak = start_price + direction * spike_mag
    spike_prices = np.linspace(start_price, peak, spike)
    tail_prices = np.linspace(peak, start_price * rng.uniform(0.99, 1.01), tail)
    return np.concatenate([spike_prices, tail_prices])


def _prices_to_ohlcv(
    close: np.ndarray,
    noise_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise OHLV from close prices with random intrabar noise."""
    n = len(close)
    noise = np.abs(rng.normal(0.0, noise_frac, n)) * close
    high = close + noise
    low = np.maximum(close - noise, 0.01)
    volume = rng.uniform(1_000.0, 10_000.0, n) * (1.0 + noise_frac * 5.0)
    return high, low, close.copy(), volume


def generate_synthetic_market(
    total_bars: int = TOTAL_BARS,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate OHLCV arrays and ground-truth regime labels.

    Returns (high, low, close, volume, labels) each of length `total_bars`.
    """
    rng = np.random.default_rng(seed)
    close_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []

    price = 100.0
    generated = 0

    # Regime types and their relative weights
    regime_types = ["trend_up", "trend_down", "ranging", "false_breakout", "manipulation"]
    regime_weights = [0.25, 0.25, 0.30, 0.10, 0.10]

    while generated < total_bars:
        remaining = total_bars - generated
        seg_len = int(rng.integers(200, 2000))  # longer segments → indicators stabilise
        seg_len = min(seg_len, remaining)

        r_type = rng.choice(regime_types, p=regime_weights)

        if r_type == "trend_up":
            drift = float(rng.uniform(0.25, 0.80))   # stronger trends → clearer ADX
            sigma = float(rng.uniform(0.10, 0.25))
            prices = _gbm_segment(seg_len, price, drift, sigma, rng)
            labels = np.full(seg_len, _UP, dtype=np.int8)

        elif r_type == "trend_down":
            drift = float(rng.uniform(-0.80, -0.25))
            sigma = float(rng.uniform(0.10, 0.25))
            prices = _gbm_segment(seg_len, price, drift, sigma, rng)
            labels = np.full(seg_len, _DN, dtype=np.int8)

        elif r_type == "ranging":
            mu = price
            theta = float(rng.uniform(0.05, 0.20))
            sigma_ou = float(rng.uniform(0.05, 0.15)) * price / 252.0
            prices = _ou_segment(seg_len, price, mu, theta, sigma_ou, rng)
            labels = np.full(seg_len, _RG, dtype=np.int8)

        elif r_type == "false_breakout":
            direction = rng.choice([-1, 1])
            prices = _false_breakout_segment(seg_len, price, direction, rng)
            labels = np.full(seg_len, _RG, dtype=np.int8)

        else:  # manipulation
            prices = _manipulation_segment(seg_len, price, rng)
            labels = np.full(seg_len, _RG, dtype=np.int8)

        close_list.append(prices.astype(np.float64))
        label_list.append(labels)
        price = float(prices[-1])
        generated += seg_len

    close = np.concatenate(close_list)[:total_bars]
    labels = np.concatenate(label_list)[:total_bars]
    high, low, close, volume = _prices_to_ohlcv(close, noise_frac=0.005, rng=rng)
    return high, low, close, volume, labels


# ── Accuracy helpers ──────────────────────────────────────────────────────────

def _accuracy(pred: np.ndarray, true: np.ndarray, warmup: int = WARMUP) -> float:
    p = pred[warmup:]
    t = true[warmup:]
    if len(p) == 0:
        return float("nan")
    return float(np.mean(p == t))


def _per_class_accuracy(
    pred: np.ndarray, true: np.ndarray, warmup: int = WARMUP
) -> Dict[str, float]:
    p = pred[warmup:]
    t = true[warmup:]
    out: Dict[str, float] = {}
    for label, name in [(_UP, "trend_up"), (_DN, "trend_down"), (_RG, "ranging")]:
        mask = t == label
        if mask.sum() == 0:
            out[name] = float("nan")
        else:
            out[name] = float(np.mean(p[mask] == label))
    return out


# ── Walk-forward simulation ───────────────────────────────────────────────────

def run_walk_forward(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    true_labels: np.ndarray,
    train_len: int = WF_TRAIN_LEN,
    test_len: int = WF_TEST_LEN,
) -> Dict[str, object]:
    """Walk-forward: observe accuracy in `test_len` bars after each `train_len` window.

    Returns dict with per-step accuracies and aggregate statistics.
    """
    n = len(close)
    step = test_len
    step_accuracies: List[float] = []

    i = train_len
    while i + test_len <= n:
        # Detect on [i, i+test_len) only – we don't retrain parameters here
        # (detector is parameter-free; walk-forward validates temporal stability)
        seg_h = high[i: i + test_len]
        seg_l = low[i: i + test_len]
        seg_c = close[i: i + test_len]
        seg_true = true_labels[i: i + test_len]

        result = detect_regime(seg_h, seg_l, seg_c)
        acc = _accuracy(result.regime, seg_true, warmup=0)
        step_accuracies.append(acc)
        i += step

    arr = np.array(step_accuracies)
    return {
        "n_steps": len(arr),
        "mean_accuracy": float(np.nanmean(arr)),
        "std_accuracy": float(np.nanstd(arr)),
        "min_accuracy": float(np.nanmin(arr)) if len(arr) > 0 else float("nan"),
        "max_accuracy": float(np.nanmax(arr)) if len(arr) > 0 else float("nan"),
        "pct_steps_above_60": float(np.nanmean(arr >= 0.60)) if len(arr) > 0 else float("nan"),
        "step_accuracies": arr.tolist(),
    }


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def run_montecarlo(
    n_runs: int = MC_RUNS,
    bars_per_run: int = MC_BARS,
    base_seed: int = SEED,
) -> Dict[str, object]:
    """Run `n_runs` independent simulations of `bars_per_run` bars each.

    Total samples = n_runs × bars_per_run (default 10 M).
    """
    accuracies: List[float] = []

    for run in range(n_runs):
        h, l, c, v, labels = generate_synthetic_market(
            total_bars=bars_per_run, seed=base_seed + run
        )
        result = detect_regime(h, l, c)
        acc = _accuracy(result.regime, labels)
        accuracies.append(acc)

    arr = np.array(accuracies)
    return {
        "n_runs": n_runs,
        "bars_per_run": bars_per_run,
        "total_samples": n_runs * bars_per_run,
        "mean_accuracy": float(np.mean(arr)),
        "std_accuracy": float(np.std(arr)),
        "ci_95_low": float(np.percentile(arr, 2.5)),
        "ci_95_high": float(np.percentile(arr, 97.5)),
        "pct_runs_above_60": float(np.mean(arr >= 0.60)),
        "run_accuracies": arr.tolist(),
    }


# ── Full backtest (20/20/20/40 splits) ───────────────────────────────────────

@dataclass
class BacktestResult:
    total_bars: int
    split_sizes: Dict[str, int]
    split_accuracies: Dict[str, float]
    split_per_class: Dict[str, Dict[str, float]]
    montecarlo: Dict[str, object]
    walk_forward: Dict[str, object]
    overall_accuracy: float
    meets_60pct_target: bool


def run_backtest(
    total_bars: int = TOTAL_BARS,
    seed: int = SEED,
    mc_runs: int = MC_RUNS,
    mc_bars: int = MC_BARS,
    wf_train: int = WF_TRAIN_LEN,
    wf_test: int = WF_TEST_LEN,
    verbose: bool = True,
) -> BacktestResult:
    """Full backtest pipeline."""

    # 1. Generate synthetic market data
    t0 = time.time()
    if verbose:
        print(f"Generating {total_bars:,} synthetic bars …", flush=True)
    high, low, close, volume, labels = generate_synthetic_market(total_bars, seed)
    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    # 2. Run detector on entire dataset
    t0 = time.time()
    if verbose:
        print("Running regime detector …", flush=True)
    result: RegimeResult = detect_regime(high, low, close)
    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    # 3. Split into train / val / retrain / out-of-sample
    n = total_bars
    n_train = int(n * SPLIT_TRAIN)
    n_val = int(n * SPLIT_VAL)
    n_retrain = int(n * SPLIT_RETRAIN)
    n_oos = n - n_train - n_val - n_retrain

    split_bounds = {
        "train":    (0,                   n_train),
        "val":      (n_train,             n_train + n_val),
        "retrain":  (n_train + n_val,     n_train + n_val + n_retrain),
        "oos":      (n_train + n_val + n_retrain, n),
    }

    split_sizes: Dict[str, int] = {k: b - a for k, (a, b) in split_bounds.items()}
    split_accuracies: Dict[str, float] = {}
    split_per_class: Dict[str, Dict[str, float]] = {}

    for split_name, (a, b) in split_bounds.items():
        pred = result.regime[a:b]
        true = labels[a:b]
        acc = _accuracy(pred, true, warmup=WARMUP if a == 0 else 0)
        split_accuracies[split_name] = acc
        split_per_class[split_name] = _per_class_accuracy(pred, true, warmup=WARMUP if a == 0 else 0)

    # 4. Monte Carlo
    if verbose:
        print(f"Running {mc_runs} Monte Carlo simulations ({mc_bars:,} bars each) …", flush=True)
    t0 = time.time()
    mc = run_montecarlo(mc_runs, mc_bars, base_seed=seed + 1000)
    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s  "
              f"mean acc={mc['mean_accuracy']:.3f}  "
              f"95% CI=[{mc['ci_95_low']:.3f}, {mc['ci_95_high']:.3f}]", flush=True)

    # 5. Walk-forward (on OOS portion to avoid lookahead)
    oos_a, oos_b = split_bounds["oos"]
    if verbose:
        print(f"Running walk-forward on OOS ({oos_b - oos_a:,} bars) …", flush=True)
    t0 = time.time()
    wf = run_walk_forward(
        high[oos_a:oos_b], low[oos_a:oos_b], close[oos_a:oos_b],
        labels[oos_a:oos_b], wf_train, wf_test,
    )
    if verbose:
        print(f"  Done in {time.time() - t0:.1f}s  "
              f"steps={wf['n_steps']}  mean acc={wf['mean_accuracy']:.3f}", flush=True)

    overall = float(np.mean([
        split_accuracies["val"],
        split_accuracies["retrain"],
        split_accuracies["oos"],
        mc["mean_accuracy"],
        wf["mean_accuracy"],
    ]))

    return BacktestResult(
        total_bars=total_bars,
        split_sizes=split_sizes,
        split_accuracies=split_accuracies,
        split_per_class=split_per_class,
        montecarlo=mc,
        walk_forward=wf,
        overall_accuracy=overall,
        meets_60pct_target=overall >= 0.60,
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _print_result(bt: BacktestResult) -> None:
    print("\n══════════════════════════════════════════════")
    print("  Market Regime Detector – Backtest Summary")
    print("══════════════════════════════════════════════")
    print(f"  Total bars generated : {bt.total_bars:,}")
    print()
    print("  Split accuracies (after warmup):")
    for name in ["train", "val", "retrain", "oos"]:
        acc = bt.split_accuracies[name]
        size = bt.split_sizes[name]
        pc = bt.split_per_class[name]
        up = pc.get("trend_up", float("nan"))
        dn = pc.get("trend_down", float("nan"))
        rg = pc.get("ranging", float("nan"))
        print(f"    {name:8s}  n={size:>9,}  acc={acc:.3f}  "
              f"(↑{up:.2f} ↓{dn:.2f} ~{rg:.2f})")
    print()
    mc = bt.montecarlo
    print(f"  Monte Carlo ({mc['n_runs']} runs × {mc['bars_per_run']:,} bars):")
    print(f"    mean={mc['mean_accuracy']:.3f}  std={mc['std_accuracy']:.3f}  "
          f"95% CI=[{mc['ci_95_low']:.3f}, {mc['ci_95_high']:.3f}]  "
          f"pct≥60%={mc['pct_runs_above_60']:.1%}")
    print()
    wf = bt.walk_forward
    print(f"  Walk-forward ({wf['n_steps']} steps, train={WF_TRAIN_LEN}, test={WF_TEST_LEN}):")
    print(f"    mean={wf['mean_accuracy']:.3f}  std={wf['std_accuracy']:.3f}  "
          f"min={wf['min_accuracy']:.3f}  max={wf['max_accuracy']:.3f}  "
          f"pct≥60%={wf['pct_steps_above_60']:.1%}")
    print()
    status = "✓ PASS" if bt.meets_60pct_target else "✗ FAIL"
    print(f"  Overall accuracy : {bt.overall_accuracy:.3f}  [{status} ≥60% target]")
    print("══════════════════════════════════════════════\n")


if __name__ == "__main__":
    bt = run_backtest(verbose=True)
    _print_result(bt)
