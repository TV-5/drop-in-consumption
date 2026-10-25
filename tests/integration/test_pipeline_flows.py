"""The whole detection pipeline on small hand-built populations and on the app's generated data."""
import pandas as pd
import pytest

from src.pipeline import run_pipeline
from tests.helpers import CFG, EVAL, make, scale

pytestmark = pytest.mark.integration


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


def test_every_meter_gets_exactly_one_row_with_a_stable_id():
    daily, master = make()
    a = run_pipeline(daily, master, EVAL, CFG).sort_values("meter_id")
    b = run_pipeline(daily, master, EVAL, CFG).sort_values("meter_id")
    assert len(a) == len(master) and a["meter_id"].is_unique
    assert a["score_id"].tolist() == b["score_id"].tolist() and a["score_id"].is_unique


# ------------------------------------------------------------------ the app's generated data
EXPECTED = {  # planted scenario -> outcome every such meter must get
    "vacancy": "SUPPRESSED:VACANT", "comms_failure": "SUPPRESSED:COMM_POOR", "meter_replaced": "SUPPRESSED:METER_REPLACED",
    "new_connection": "SUPPRESSED:NEW_CONNECTION", "der_added": "SUPPRESSED:DER_PRESENT",
    "disconnected": "SUPPRESSED:DISCONNECTED", "small_consumer_drop": "SUPPRESSED:BELOW_FLOOR",
    "full_bypass": "FLAGGED:",
}


def outcomes(generated):
    s = generated["scores"].merge(generated["truth"], on="meter_id", how="left")
    s["injected_type"] = s["injected_type"].fillna("normal")
    s["outcome"] = s["status"] + ":" + s["suppression_code"].fillna("")
    return s


@pytest.mark.parametrize("kind", list(EXPECTED))
def test_planted_scenario_gets_the_expected_outcome(generated, kind):
    s = outcomes(generated)
    assert set(s.loc[s["injected_type"] == kind, "outcome"]) == {EXPECTED[kind]}


def test_partial_bypasses_flag_unless_too_small(generated, real_cfg):
    s = outcomes(generated)
    pb = s[s["injected_type"] == "partial_bypass"]
    assert set(pb["outcome"]) <= {"FLAGGED:", "SUPPRESSED:BELOW_FLOOR"}
    assert (pb.loc[pb["outcome"] == "SUPPRESSED:BELOW_FLOOR", "drop_kwh_per_day"] < real_cfg.min_drop_kwh_per_day).all()
    assert (pb["outcome"] == "FLAGGED:").sum() >= 10


def test_normal_meters_are_not_flagged(generated):
    s = outcomes(generated)
    assert (s.loc[s["injected_type"] == "normal", "status"] == "FLAGGED").sum() == 0


def test_before_any_problem_is_planted_nothing_is_flagged(generated, real_cfg):
    """Checked as of 4 Sep, before any planted problem starts: no bypass can be caught yet."""
    s = run_pipeline(generated["daily"], generated["master"], pd.Timestamp("2026-09-04").date(), real_cfg)
    bypass = set(generated["truth"].query("injected_type in ['partial_bypass', 'full_bypass']")["meter_id"])
    assert not s.loc[s["meter_id"].isin(bypass), "status"].eq("FLAGGED").any()
    assert (s["status"] == "FLAGGED").mean() < 0.02
