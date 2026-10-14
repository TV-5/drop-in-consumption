from datetime import date

import pandas as pd

from .config import Config


def windows(eval_date: date, cfg: Config) -> dict[str, pd.Timestamp]:
    """Inclusive bounds of the baseline and current windows."""
    ev = pd.Timestamp(eval_date)
    baseline_end = ev - pd.Timedelta(days=cfg.baseline_gap_days + 1)
    return {
        "baseline_start": baseline_end - pd.Timedelta(days=cfg.baseline_days - 1),
        "baseline_end": baseline_end,
        "current_start": ev - pd.Timedelta(days=cfg.current_window_days - 1),
        "current_end": ev,
    }


def build_baselines(daily: pd.DataFrame, eval_date: date, cfg: Config) -> pd.DataFrame:
    """One row per meter: baseline_daily_kwh, days_covered, window bounds.

    Window ends cfg.baseline_gap_days before eval_date and runs back
    cfg.baseline_days. The gap is load-bearing: without it a slow-onset
    bypass contaminates its own reference.

    Uses median, not mean. Returns days_covered so the caller can reject
    thin baselines rather than trusting a number built on 12 days.
    """
    w = windows(eval_date, cfg)
    meters = pd.Index(daily["meter_id"].unique(), name="meter_id")
    in_win = daily[daily["reading_date"].between(w["baseline_start"], w["baseline_end"])]
    g = in_win.groupby("meter_id")["kwh_import"]
    out = pd.DataFrame({
        "baseline_daily_kwh": g.median(),
        "days_covered": g.count(),
    }).reindex(meters)
    out["days_covered"] = out["days_covered"].fillna(0).astype(int)
    out.loc[out["days_covered"] < cfg.min_baseline_days_covered, "baseline_daily_kwh"] = float("nan")
    out["baseline_start"] = w["baseline_start"]
    out["baseline_end"] = w["baseline_end"]
    return out.reset_index()
