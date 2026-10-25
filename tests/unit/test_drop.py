import pandas as pd
import pytest

from src.baseline import build_baselines, windows
from src.drop import compute_drop
from tests.helpers import CFG, EVAL, make


def flat(kwh_before=10.0, kwh_now=4.0):
    daily, _ = make(n=1)
    w = windows(EVAL.date(), CFG)
    daily["kwh_import"] = kwh_before
    daily.loc[daily["reading_date"] >= w["current_start"], "kwh_import"] = kwh_now
    return daily, w


def drop_row(daily):
    return compute_drop(daily, build_baselines(daily, EVAL, CFG), EVAL, CFG).iloc[0]


def test_drop_in_percent_and_kwh():
    r = drop_row(flat()[0])
    assert r["baseline_daily_kwh"] == 10.0 and r["current_daily_kwh"] == 4.0
    assert r["drop_kwh_per_day"] == pytest.approx(6.0)
    assert r["drop_pct"] == pytest.approx(60.0)


def test_readings_arrived_percent():
    daily, w = flat()
    daily.loc[daily["reading_date"] >= w["current_start"], "intervals_received"] = 24
    assert drop_row(daily)["comm_availability_pct"] == pytest.approx(50.0)


def test_missing_days_count_as_no_readings():
    daily, w = flat()
    now = daily["reading_date"] >= w["current_start"]
    daily = daily[~now | (daily["reading_date"] >= w["current_start"] + pd.Timedelta(days=15))]  # first 15 days missing
    assert drop_row(daily)["comm_availability_pct"] == pytest.approx(50.0)


def test_zero_baseline_has_no_drop_percent():
    assert pd.isna(drop_row(flat(kwh_before=0.0, kwh_now=0.0)[0])["drop_pct"])
