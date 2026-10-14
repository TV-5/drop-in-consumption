import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sample_data.generate import generate_population  # noqa: E402
from src.baseline import build_baselines  # noqa: E402
from src.config import Config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.schemas import parse  # noqa: E402

CFG = Config()
EVAL = pd.Timestamp("2026-10-04")


def make(n=40, days=180, kwh=8.0, seed=0, **master_overrides):
    """A clean population on one transformer: n identical-ish meters."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=EVAL, periods=days)
    daily = pd.DataFrame([
        {"meter_id": str(i), "reading_date": d, "kwh_import": kwh * rng.lognormal(0, 0.1),
         "intervals_expected": 48, "intervals_received": 48, "voltage_present_pct": 100.0}
        for i in range(n) for d in dates
    ])
    master = pd.DataFrame({
        "meter_id": [str(i) for i in range(n)], "consumer_id": [f"C{i}" for i in range(n)],
        "dt_id": "DT-1", "subdivision_id": "SD-1", "tariff_category": "LT-1 Domestic",
        "sanctioned_load_kw": 3.0, "phase_type": "1P", "status": "ACTIVE",
        "status_changed_on": pd.Timestamp("2023-01-01"), "meter_installed_on": pd.Timestamp("2023-01-01"),
        "has_registered_der": False,
    })
    for k, v in master_overrides.items():
        master[k] = v
    return daily, master


def scale(daily, meter_ids, since, factor):
    m = daily["meter_id"].isin(meter_ids) & (daily["reading_date"] >= since)
    daily.loc[m, "kwh_import"] *= factor


def test_seasonal_neutrality():
    daily, master = make()
    scale(daily, master["meter_id"], EVAL - pd.Timedelta(days=29), 0.6)  # uniform 40% drop
    s = run_pipeline(daily, master, EVAL, CFG)
    assert (s["status"] == "FLAGGED").sum() == 0


def test_bypass_flagged_against_seasonal_drop():
    daily, master = make()
    scale(daily, master["meter_id"], EVAL - pd.Timedelta(days=29), 0.6)
    scale(daily, ["0"], EVAL - pd.Timedelta(days=25), 0.3)
    s = run_pipeline(daily, master, EVAL, CFG).set_index("meter_id")
    assert s.at["0", "status"] == "FLAGGED"
    assert (s["status"] == "FLAGGED").sum() == 1
    assert "suspected" in s.at["0", "reason_text"].lower()


def test_median_robustness():
    daily, _ = make(n=1)
    base = build_baselines(daily, EVAL, CFG).at[0, "baseline_daily_kwh"]
    spike = daily["reading_date"] == EVAL - pd.Timedelta(days=60)
    daily.loc[spike, "kwh_import"] *= 10
    after = build_baselines(daily, EVAL, CFG).at[0, "baseline_daily_kwh"]
    assert abs(after - base) / base < 0.02


def test_thin_baseline_is_null():
    daily, _ = make(n=1)
    daily = daily[daily["reading_date"] > EVAL - pd.Timedelta(days=80)]  # ~50 baseline days
    assert pd.isna(build_baselines(daily, EVAL, CFG).at[0, "baseline_daily_kwh"])


def test_suppression_order_comm_poor_first():
    daily, master = make()
    master.loc[0, "status"] = "DISCONNECTED"
    m = (daily["meter_id"] == "0") & (daily["reading_date"] >= EVAL - pd.Timedelta(days=29))
    daily.loc[m, "intervals_received"] = 10
    s = run_pipeline(daily, master, EVAL, CFG).set_index("meter_id")
    assert s.at["0", "suppression_code"] == "COMM_POOR"


def test_absolute_floor():
    daily, master = make(kwh=1.3)
    scale(daily, ["0"], EVAL - pd.Timedelta(days=29), 0.3)  # 70% drop on 1.3 -> 0.9 kWh/day
    s = run_pipeline(daily, master, EVAL, CFG).set_index("meter_id")
    assert s.at["0", "status"] != "FLAGGED"
    assert s.at["0", "suppression_code"] == "BELOW_FLOOR"


def test_peer_fallback_widens_geography_not_tariff():
    daily, master = make(n=40)
    master["dt_id"] = [f"DT-{i % 4}" for i in range(40)]  # 10 per DT, below min group size
    s = run_pipeline(daily, master, EVAL, CFG)
    assert (s["peer_level"] == "sub-division").all()
    assert (s["peer_group_size"] == 40).all()


def test_idempotent_score_ids():
    daily, master = make()
    a = run_pipeline(daily, master, EVAL, CFG).sort_values("meter_id")
    b = run_pipeline(daily, master, EVAL, CFG).sort_values("meter_id")
    assert a["score_id"].tolist() == b["score_id"].tolist()
    assert a["score_id"].is_unique


def test_synthetic_population_recovery():
    data = generate_population(seed=7)
    daily, _ = parse(data["daily_consumption"].astype(str), "daily")
    master, _ = parse(data["meter_master"].astype(str), "master")
    s = run_pipeline(daily, master, EVAL, CFG).merge(data["ground_truth"], on="meter_id", how="left")
    s["injected_type"] = s["injected_type"].fillna("normal")
    bypass = s[s["injected_type"].isin(["partial_bypass", "full_bypass"])]
    # Bypasses on very small loads fall under the absolute kWh floor by design; every other one flags.
    above_floor = bypass[bypass["suppression_code"] != "BELOW_FLOOR"]
    assert (above_floor["status"] == "FLAGGED").all()
    assert (bypass.loc[bypass["suppression_code"] == "BELOW_FLOOR", "drop_kwh_per_day"] < CFG.min_drop_kwh_per_day).all()
    assert len(above_floor) >= 15
    for kind in ["vacancy", "comms_failure", "der_added", "disconnected", "new_connection", "meter_replaced"]:
        assert not (s.loc[s["injected_type"] == kind, "status"] == "FLAGGED").any(), kind
    assert (s.loc[s["injected_type"] == "normal", "status"] == "FLAGGED").mean() < 0.02


@pytest.mark.parametrize("kind", ["daily", "master", "officers"])
def test_schema_rejects_missing_columns(kind):
    _, errors = parse(pd.DataFrame({"foo": ["1"]}), kind)
    assert errors and errors[0].startswith("Missing columns")
