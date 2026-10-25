import numpy as np
import pandas as pd
import pytest

from src.baseline import windows
from src.score import score_and_explain, score_id
from tests.helpers import CFG, EVAL, make

W = windows(EVAL.date(), CFG)


def row(meter_id, **over):
    r = {"meter_id": meter_id, "status": "", "suppression_code": "", "baseline_daily_kwh": 10.0,
         "current_daily_kwh": 4.0, "drop_pct": 60.0, "peer_drop_pct": 0.0, "excess_drop_pct": 60.0,
         "drop_kwh_per_day": 6.0, "peer_group_size": 25, "peer_level": "transformer", "comm_availability_pct": 100.0}
    return {**r, **over}


def flat_daily(meter_ids, drop_from=None, before=10.0, after=2.0):
    """Constant use; from `drop_from` on it falls to `after`."""
    daily, _ = make(n=len(meter_ids))
    daily["meter_id"] = daily["meter_id"].map(dict(zip([str(i) for i in range(len(meter_ids))], meter_ids)))
    daily["kwh_import"] = before
    if drop_from is not None:
        daily.loc[daily["reading_date"] >= drop_from, "kwh_import"] = after
    return daily


def score(rows, daily):
    return score_and_explain(pd.DataFrame(rows), daily, EVAL.date(), CFG).set_index("meter_id")


def test_statuses_and_reasons():
    out = score([row("F"), row("C", excess_drop_pct=10.0), row("S", status="SUPPRESSED", suppression_code="VACANT")],
                flat_daily(["F", "C", "S"], drop_from=pd.Timestamp("2026-09-20")))
    assert out.at["F", "status"] == "FLAGGED" and out.at["C", "status"] == "CLEAR" and out.at["S", "status"] == "SUPPRESSED"
    assert "suspected" in out.at["F", "reason_text"].lower() and "recommended for inspection" in out.at["F", "reason_text"]
    assert out.at["C", "reason_text"] == ""
    assert out.at["S", "reason_text"].startswith("Not scored:")


def test_onset_is_at_or_just_before_the_drop():
    # The onset uses the median of the NEXT five days, so it fires once three of them are low: up to two days
    # before the real drop. Known behaviour (reported to the project owner); tighten this if the rule changes.
    drop = pd.Timestamp("2026-09-20")
    onset = pd.Timestamp(score([row("F")], flat_daily(["F"], drop_from=drop)).at["F", "onset_date"])
    assert drop - pd.Timedelta(days=2) <= onset <= drop


def test_onset_falls_back_to_the_start_of_now_when_no_clear_drop():
    out = score([row("F")], flat_daily(["F"]))  # flagged on the numbers, but daily use never dips
    assert pd.Timestamp(out.at["F", "onset_date"]) == W["current_start"]


def test_unbilled_estimate_counts_only_the_drop_beyond_neighbours():
    onset = pd.Timestamp("2026-09-20")
    out = score([row("F", excess_drop_pct=40.0, peer_drop_pct=20.0)], flat_daily(["F"], drop_from=onset))
    days = (EVAL - pd.Timestamp(out.at["F", "onset_date"])).days + 1
    assert out.at["F", "est_unbilled_kwh"] == pytest.approx(10.0 * 0.40 * days)


def test_ranked_by_kwh_at_risk_not_by_percentage():
    onset = pd.Timestamp("2026-09-20")
    rows = [row("small", baseline_daily_kwh=5.0, excess_drop_pct=60.0),
            row("big", baseline_daily_kwh=20.0, excess_drop_pct=25.0)]
    out = score_and_explain(pd.DataFrame(rows), flat_daily(["small", "big"], drop_from=onset), EVAL.date(), CFG)
    assert list(out["meter_id"]) == ["big", "small"]


def test_only_flagged_meters_get_an_estimate():
    out = score([row("C", excess_drop_pct=10.0)], flat_daily(["C"]))
    assert np.isnan(out.at["C", "est_unbilled_kwh"])


def test_score_id_is_stable_and_specific():
    a = score_id("1", EVAL, "v1")
    assert a == score_id("1", EVAL, "v1")
    assert len({a, score_id("2", EVAL, "v1"), score_id("1", EVAL + pd.Timedelta(days=1), "v1"),
                score_id("1", EVAL, "v2")}) == 4
