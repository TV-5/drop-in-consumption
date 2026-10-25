import numpy as np
import pandas as pd
import pytest

from src.suppress import SUPPRESSION_CODES, apply_suppressions
from tests.helpers import CFG, make

BASELINE_START = pd.Timestamp("2026-05-08")


def row(**over):
    """One scored meter that would be CLEAR: a modest drop, readings fine, active, no solar."""
    r = {"meter_id": "0", "excess_drop_pct": 5.0, "drop_kwh_per_day": 5.0, "comm_availability_pct": 100.0,
         "baseline_daily_kwh": 8.0, "current_daily_kwh": 6.0, "baseline_start": BASELINE_START}
    return pd.DataFrame([{**r, **over}])


def code_for(scored, **master_over):
    daily, master = make(n=1)
    for k, v in master_over.items():
        master[k] = v
    return apply_suppressions(scored, master, daily, CFG).iloc[0]["suppression_code"]


@pytest.mark.parametrize("scored, master_over, expected", [
    (row(meter_id="missing"), {}, "NO_MASTER_RECORD"),
    (row(comm_availability_pct=50.0), {}, "COMM_POOR"),
    (row(comm_availability_pct=np.nan), {}, "COMM_POOR"),
    (row(), {"status": "DISCONNECTED"}, "DISCONNECTED"),
    (row(), {"meter_installed_on": pd.Timestamp("2026-09-10")}, "METER_REPLACED"),
    (row(baseline_daily_kwh=np.nan), {}, "NEW_CONNECTION"),
    (row(), {"has_registered_der": True}, "DER_PRESENT"),
    (row(current_daily_kwh=0.05), {}, "VACANT"),
    (row(excess_drop_pct=50.0, drop_kwh_per_day=1.0), {}, "BELOW_FLOOR"),
], ids=lambda v: v if isinstance(v, str) else "")
def test_each_innocent_explanation_has_its_code(scored, master_over, expected):
    assert code_for(scored, **master_over) == expected


def test_every_code_has_a_plain_explanation():
    assert set(SUPPRESSION_CODES) == {"NO_MASTER_RECORD", "COMM_POOR", "DISCONNECTED", "METER_REPLACED",
                                      "NEW_CONNECTION", "DER_PRESENT", "VACANT", "BELOW_FLOOR"}
    assert all(SUPPRESSION_CODES.values())


def test_small_drop_that_would_not_flag_is_not_below_floor():
    assert code_for(row(excess_drop_pct=5.0, drop_kwh_per_day=1.0)) == ""


def test_clear_meter_gets_no_code():
    daily, master = make(n=1)
    out = apply_suppressions(row(), master, daily, CFG).iloc[0]
    assert out["suppression_code"] == "" and out["status"] == ""


def test_first_matching_reason_wins():
    assert code_for(row(comm_availability_pct=10.0), status="DISCONNECTED") == "COMM_POOR"
    assert code_for(row(), status="DISCONNECTED", has_registered_der=True) == "DISCONNECTED"


def test_no_rows_are_dropped():
    daily, master = make(n=3)
    scored = pd.concat([row(meter_id="0"), row(meter_id="1", current_daily_kwh=0.0), row(meter_id="nope")])
    out = apply_suppressions(scored, master, daily, CFG)
    assert len(out) == 3
    assert list(out["status"]) == ["", "SUPPRESSED", "SUPPRESSED"]
