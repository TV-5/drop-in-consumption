import pandas as pd

from src.baseline import build_baselines, windows
from src.config import Config
from tests.helpers import CFG, EVAL, make


def test_windows_for_the_demo_date():
    w = windows(EVAL.date(), CFG)
    assert w == {
        "baseline_start": pd.Timestamp("2026-05-08"), "baseline_end": pd.Timestamp("2026-08-05"),
        "current_start": pd.Timestamp("2026-09-05"), "current_end": pd.Timestamp("2026-10-04"),
    }


def test_windows_leave_a_real_gap_between_before_and_now():
    w = windows(EVAL.date(), CFG)
    gap_days = (w["current_start"] - w["baseline_end"]).days - 1
    assert gap_days == CFG.baseline_gap_days == 30


def test_window_lengths_follow_config():
    cfg = Config(baseline_days=60, baseline_gap_days=10, current_window_days=14)
    w = windows(EVAL.date(), cfg)
    assert (w["baseline_end"] - w["baseline_start"]).days + 1 == 60
    assert (w["current_end"] - w["current_start"]).days + 1 == 14
    assert (w["current_start"] - w["baseline_end"]).days - 1 == 10


def test_median_robustness():
    daily, _ = make(n=1)
    base = build_baselines(daily, EVAL, CFG).at[0, "baseline_daily_kwh"]
    spike = daily["reading_date"] == EVAL - pd.Timedelta(days=100)  # inside the "before" period
    daily.loc[spike, "kwh_import"] *= 10
    after = build_baselines(daily, EVAL, CFG).at[0, "baseline_daily_kwh"]
    assert abs(after - base) / base < 0.02


def test_thin_baseline_is_null():
    daily, _ = make(n=1)
    daily = daily[daily["reading_date"] > EVAL - pd.Timedelta(days=80)]  # under 60 days in the "before" period
    out = build_baselines(daily, EVAL, CFG)
    assert pd.isna(out.at[0, "baseline_daily_kwh"])
    assert out.at[0, "days_covered"] < CFG.min_baseline_days_covered


def test_baseline_ignores_the_gap_and_the_current_period():
    daily, _ = make(n=1)
    before = build_baselines(daily, EVAL, CFG).at[0, "baseline_daily_kwh"]
    w = windows(EVAL.date(), CFG)
    daily.loc[daily["reading_date"] > w["baseline_end"], "kwh_import"] = 0.0  # a drop in the gap and "now"
    assert build_baselines(daily, EVAL, CFG).at[0, "baseline_daily_kwh"] == before
