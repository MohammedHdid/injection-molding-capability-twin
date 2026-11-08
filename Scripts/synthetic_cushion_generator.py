#!/usr/bin/env python3
"""Generate illustrative synthetic input for the cushion-forecasting notebook.

Companion to:
    On-demand capability forecasting in injection moulding: a state-conditioned
    digital twin with audited cycle-level predictive bands — a case study on
    cushion monitoring
    Mohammed Hdid, Hakim Jebari, Hassan Samadi

This is a software demonstration, not a reconstruction of confidential records.
Reference marginal statistics come from the manuscript. The stochastic model,
feature coupling, run lengths and injected excursions are illustrative choices.
The output does not reproduce the paper's forecasts, calibration or conclusions.

Usage (Python 3.10+, numpy, pandas):
    python scripts/synthetic_cushion_generator.py --out synthetic_cushion.csv
    python scripts/synthetic_cushion_generator.py --verify
    python scripts/synthetic_cushion_generator.py --n-cycles 5000 --verify

CSV output uses a semicolon delimiter and day-first timestamps, matching the
notebook's input settings. Set DATA_PATH in a separate working copy of the
notebook to use the generated file. Preprocessing drops segment/split-start rows,
so the generated row count is not the final modelling-frame count. Synthetic
training and analysis results will differ from the saved industrial outputs.

Licence: CC0, retained from the supplied generator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "1.1.0"


@dataclass
class SurrogateConfig:
    n_cycles: int = 25_767
    n_segments: int = 17
    seed: int = 42
    cycle_time_s: float = 15.0
    segment_gap_s: float = 180.0
    timestamp_resolution_s: float = 60.0
    start_timestamp: str = "2026-01-01 06:00:00"
    target: float = 10.33
    lsl: float = 10.03
    usl: float = 10.63
    q: float = 0.01

    # Reference scales from the corrected manuscript, not guaranteed outputs.
    sigma_bulk: float = 0.02514
    sigma_seg: float = 0.01606
    sigma_w: float = 0.01470
    sigma_s: float = 0.00169
    rho_dC: float = -0.574

    # Table 1 marginal means and SDs. These do not specify a joint process law.
    feature_stats: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "Maximum_Injection_Pressure": (1531.3, 67.0),
        "Switch_Over_Pressure": (1530.3, 66.9),
        "Injection_Time": (1.018, 0.006),
        "Dosing_Time": (2.408, 0.046),
        "Back_pressure_ActValue": (67.7, 1.8),
        "SwitchOver_volume_ActValue": (12.996, 0.005),
        "Dosing_Volume_ActValue": (90.439, 0.496),
    })
    pressure_correlation: float = 0.997

    # Demonstration assumptions, not estimates from permutation importance.
    feature_segment_share: float = 0.45
    feature_persistence: float = 0.60
    exog_var_share: float = 0.0127
    n_increment_anomalies: int = 48
    n_level_anomalies: int = 44
    mad_k: float = 5.0


def _validate(cfg: SurrogateConfig) -> None:
    if cfg.n_segments < 1 or cfg.n_cycles < 60 * cfg.n_segments:
        raise ValueError("Allow at least 60 generated cycles per segment.")
    if not (0 < cfg.q and 0 < cfg.sigma_s < cfg.sigma_seg):
        raise ValueError("Resolution and slow variation must be positive.")
    if not cfg.sigma_bulk > cfg.sigma_seg > cfg.sigma_w > cfg.q / np.sqrt(12):
        raise ValueError("Require sigma_bulk > sigma_seg > sigma_w > q/sqrt(12).")
    if not cfg.lsl < cfg.target < cfg.usl:
        raise ValueError("Require lsl < target < usl.")
    if not (0 <= cfg.feature_segment_share <= 1 and
            abs(cfg.feature_persistence) < 1 and
            abs(cfg.pressure_correlation) <= 1 and
            0 <= cfg.exog_var_share < 1):
        raise ValueError("Invalid feature correlation or variance-share setting.")
    if cfg.n_increment_anomalies < 0 or cfg.n_increment_anomalies > cfg.n_cycles - 2:
        raise ValueError("Invalid number of impulse excursions.")
    if not 0 <= cfg.n_level_anomalies <= cfg.n_cycles:
        raise ValueError("Invalid sustained-excursion count.")
    if cfg.cycle_time_s <= 0 or cfg.timestamp_resolution_s <= 0:
        raise ValueError("Cycle time and timestamp resolution must be positive.")
    if cfg.segment_gap_s < cfg.timestamp_resolution_s:
        raise ValueError("The gap threshold must be at least the timestamp resolution.")


def _theta(rho: float) -> float:
    """Approximate the requested increment correlation using an MA(1) term.

    This relation applies to the fast component alone. Slow variation,
    quantisation, feature coupling and screening change the realised correlation.
    """
    correlation = lambda t: -((1 - t) ** 2) / (2 * (1 + t * t - t))
    if not -0.75 < rho <= -0.5:
        raise ValueError("This MA(1) construction requires -0.75 < rho_dC <= -0.5.")
    lo, hi = -1.0, 0.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if correlation(mid) < rho:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _segment_lengths(cfg: SurrogateConfig, rng) -> np.ndarray:
    # Allocate the minimum first; the remaining counts sum exactly to n_cycles.
    weights = rng.lognormal(0, 0.55, cfg.n_segments)
    return 60 + rng.multinomial(cfg.n_cycles - 60 * cfg.n_segments,
                               weights / weights.sum())


def _timestamps(lengths, cfg, rng):
    t = pd.Timestamp(cfg.start_timestamp).value / 1e9
    values = []
    for k, size in enumerate(lengths):
        steps = np.maximum(rng.normal(cfg.cycle_time_s, 0.35, size), 1.0)
        values.extend(t + np.cumsum(steps))
        t = values[-1]
        if k < len(lengths) - 1:
            # Leave enough gap to remain distinct after timestamp rounding.
            t += cfg.segment_gap_s * rng.uniform(2, 30) + cfg.timestamp_resolution_s
            if rng.random() < 0.25:
                t += 8 * 3600
    sec = np.floor(np.asarray(values) / cfg.timestamp_resolution_s) * cfg.timestamp_resolution_s
    return pd.to_datetime(sec, unit="s")


def _features(lengths, cfg, rng) -> pd.DataFrame:
    n = int(sum(lengths))
    share, phi = cfg.feature_segment_share, cfg.feature_persistence
    standards = {}
    for name in cfg.feature_stats:
        offsets = np.repeat(rng.normal(size=len(lengths)), lengths)
        innovations = rng.normal(size=n) * np.sqrt(1 - phi * phi)
        smooth = np.empty(n)
        smooth[0] = rng.normal()
        for i in range(1, n):
            smooth[i] = phi * smooth[i - 1] + innovations[i]
        standards[name] = np.sqrt(share) * offsets + np.sqrt(1 - share) * smooth
    pmax, pso = "Maximum_Injection_Pressure", "Switch_Over_Pressure"
    r = cfg.pressure_correlation
    standards[pso] = r * standards[pmax] + np.sqrt(1 - r * r) * standards[pso]
    return pd.DataFrame({name: mean + sd * standards[name]
                         for name, (mean, sd) in cfg.feature_stats.items()})


def _cushion(lengths, features, cfg, rng):
    n = int(sum(lengths))
    theta = _theta(cfg.rho_dC)
    sigma_q = cfg.q / np.sqrt(12)
    sigma_level = np.sqrt(cfg.sigma_seg ** 2 - cfg.sigma_w ** 2)
    if cfg.sigma_s >= sigma_level:
        raise ValueError("Slow innovation must be smaller than the level scale.")
    phi = np.sqrt(1 - (cfg.sigma_s / sigma_level) ** 2)
    sigma_offset = np.sqrt(cfg.sigma_bulk ** 2 - cfg.sigma_seg ** 2)
    sigma_eps = np.sqrt((cfg.sigma_w ** 2 - sigma_q ** 2) / (1 + theta * theta))
    level = np.empty(n)
    pos = 0
    for size in lengths:
        x = rng.normal(0, sigma_level)
        for j in range(size):
            x = phi * x + rng.normal(0, cfg.sigma_s)
            level[pos + j] = x
        pos += size
    eps = rng.normal(0, sigma_eps, n + 1)
    fast = eps[1:] + theta * eps[:-1]
    offsets = np.repeat(rng.normal(0, sigma_offset, len(lengths)), lengths)

    # Illustrative coupling. Signs follow the model's reported sensitivity;
    # magnitudes are rescaled, so these are not reproduced intervention effects.
    gains = {"Maximum_Injection_Pressure": -0.0030042,
             "Switch_Over_Pressure": -0.0031875, "Dosing_Time": -0.0018568}
    exog = np.zeros(n)
    for name, gain in gains.items():
        v = features[name].to_numpy()
        exog += gain * 100 * (v - v.mean()) / v.mean()
    if exog.std(ddof=1) > 0:
        exog *= np.sqrt(cfg.exog_var_share) * cfg.sigma_seg / exog.std(ddof=1)
    cushion = cfg.target + offsets + level + fast + exog

    # These are injected events, not intended matches to the paper's flag counts.
    scale = np.std(np.diff(cushion), ddof=1)
    ids = rng.choice(n - 2, cfg.n_increment_anomalies, replace=False) + 1
    cushion[ids] += rng.choice([-1, 1], len(ids)) * rng.uniform(6, 16, len(ids)) * scale
    remaining = cfg.n_level_anomalies
    while remaining:
        run = min(int(rng.integers(3, 12)), remaining)
        start = int(rng.integers(0, n - run + 1))
        cushion[start:start + run] += rng.choice([-1, 1]) * rng.uniform(0.10, 0.22)
        remaining -= run
    return np.round(cushion / cfg.q) * cfg.q


def generate(cfg: SurrogateConfig | None = None) -> pd.DataFrame:
    """Return synthetic rows with the notebook's required input column names."""
    cfg = cfg or SurrogateConfig()
    _validate(cfg)
    rng = np.random.default_rng(cfg.seed)
    lengths = _segment_lengths(cfg, rng)
    timestamps = _timestamps(lengths, cfg, rng)
    features = _features(lengths, cfg, rng)
    cushion = _cushion(lengths, features, cfg, rng)
    df = features.copy()
    df.insert(0, "timestamp", timestamps)
    df.insert(1, "Cycle_Counter", np.arange(1, len(df) + 1))
    df["Material_Cushion"] = cushion
    # Explicit synthetic identifiers, not plant identifiers.
    df["Machine"] = "SYNTHETIC_MACHINE"
    df["Part Number"] = "SYNTHETIC_PART"
    df["resource_id"] = "SYNTHETIC_RESOURCE"
    return df


def verify(df: pd.DataFrame, cfg: SurrogateConfig | None = None) -> pd.DataFrame:
    """Report synthetic diagnostics beside reference values; this is not validation.

    Keep removed rows as missing positions for lag calculations. Dropping them
    first would falsely join cycles separated by an excursion or a stoppage.
    The screens below are demonstration-wide MAD screens, not the notebook's
    exact training-only threshold-estimation protocol.
    """
    cfg = cfg or SurrogateConfig()
    c = df["Material_Cushion"].to_numpy(dtype=float)
    dt = df["timestamp"].diff().dt.total_seconds().to_numpy()
    segment = np.cumsum(np.nan_to_num(dt, nan=0) > cfg.segment_gap_s)
    dc = np.r_[np.nan, np.diff(c)]
    dc[np.r_[True, segment[1:] != segment[:-1]]] = np.nan
    good_dc = dc[np.isfinite(dc)]
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    inc_flag = np.isfinite(dc) & (np.abs(dc) > cfg.mad_k * max(mad(good_dc), 1e-12))
    lvl_flag = np.abs(c - np.median(c)) > cfg.mad_k * max(mad(c), cfg.q)
    clean = c.copy()
    clean[inc_flag | lvl_flag] = np.nan
    within, lag_variances = [], []
    pairs, next_pairs = [], []
    for sid in np.unique(segment):
        values = clean[segment == sid]
        if np.isfinite(values).sum() >= 500:
            within.append(np.nanstd(values, ddof=1))
        if np.isfinite(values).sum() >= 800:
            lag_variances.append(values)
        d = np.diff(values)
        valid = np.isfinite(d[:-1]) & np.isfinite(d[1:])
        pairs.extend(d[:-1][valid]); next_pairs.extend(d[1:][valid])
    rho = np.corrcoef(pairs, next_pairs)[0, 1] if len(pairs) > 2 else np.nan
    hs, variances = [], []
    for h in range(1, 41):
        diffs = [v[h:] - v[:-h] for v in lag_variances if len(v) > h]
        diffs = np.concatenate(diffs) if diffs else np.array([])
        diffs = diffs[np.isfinite(diffs)]
        if len(diffs) > 2:
            hs.append(h); variances.append(np.var(diffs, ddof=1))
    sigma_w = np.nan
    if len(hs) >= 2:
        slope, intercept = np.linalg.lstsq(np.column_stack([hs, np.ones(len(hs))]),
                                         variances, rcond=None)[0]
        sigma_w = np.sqrt(max(intercept / 2, 0))
    rows = [
        ("Rows before preprocessing", len(df), cfg.n_cycles),
        ("Segments", len(np.unique(segment)), cfg.n_segments),
        ("Increment flags", int(inc_flag.sum()), 104),
        ("Level flags", int(lvl_flag.sum()), 70),
        ("Combined flags", int((inc_flag | lvl_flag).sum()), 119),
        ("Screened bulk SD (mm)", np.nanstd(clean, ddof=1), cfg.sigma_bulk),
        ("Median within-segment SD (mm)", np.median(within) if within else np.nan, cfg.sigma_seg),
        ("Variogram fast scale (mm)", sigma_w, cfg.sigma_w),
        ("Screened increment lag-one correlation", rho, cfg.rho_dC),
        ("Resolution ratio", sigma_w / (cfg.q / np.sqrt(12)), cfg.sigma_w / (cfg.q / np.sqrt(12))),
    ]
    return pd.DataFrame(rows, columns=["Diagnostic", "Synthetic", "Reference"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, help="Output CSV; delimiter is ';'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cycles", type=int, default=25_767)
    parser.add_argument("--n-segments", type=int, default=17)
    parser.add_argument("--verify", action="store_true", help="Print descriptive diagnostics.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    cfg = SurrogateConfig(n_cycles=args.n_cycles, n_segments=args.n_segments, seed=args.seed)
    try:
        df = generate(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.quiet:
        print(f"Synthetic demonstration data: {len(df):,} rows, seed {cfg.seed}.")
        print("These data do not reproduce the industrial results. Diagnostics are descriptive.")
    if args.verify or args.out is None:
        print(verify(df, cfg).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, sep=";", index=False, date_format="%d/%m/%Y %H:%M:%S")
        if not args.quiet:
            print(f"Written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
