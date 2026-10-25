import datetime

import pandas as pd

from src.deliver import EMAIL_COLUMNS, build_emails
from src.legal import COPYRIGHT

EVAL = datetime.date(2026, 10, 4)


def flagged(rows):
    base = {"consumer_id": "C", "dt_id": "DT-1", "drop_pct": 60.4, "excess_drop_pct": 40.6,
            "onset_date": datetime.date(2026, 9, 20), "reason_text": "Suspected consumption drop", "status": "FLAGGED"}
    return pd.DataFrame([{**base, **r} for r in rows])


def officers(rows):
    return pd.DataFrame(rows, columns=["subdivision_id", "se_name", "se_email"])


def test_one_email_per_se_covering_all_their_areas():
    scores = flagged([{"meter_id": "1", "subdivision_id": "SD-1", "est_unbilled_kwh": 100.0},
                      {"meter_id": "2", "subdivision_id": "SD-2", "est_unbilled_kwh": 300.0},
                      {"meter_id": "3", "subdivision_id": "SD-3", "est_unbilled_kwh": 50.0}])
    offs = officers([("SD-1", "SE A", "a@example.com"), ("SD-2", "SE A", "a@example.com"),
                     ("SD-3", "SE B", "b@example.com")])
    emails, unmapped = build_emails(scores, offs, EVAL)
    assert unmapped == []
    by = {e.se_email: e for e in emails}
    assert set(by) == {"a@example.com", "b@example.com"}
    assert by["a@example.com"].subdivisions == ["SD-1", "SD-2"] and by["a@example.com"].n_flagged == 2
    assert by["a@example.com"].kwh_at_risk == 400.0


def test_areas_without_an_se_are_reported_not_sent():
    scores = flagged([{"meter_id": "1", "subdivision_id": "SD-9", "est_unbilled_kwh": 10.0}])
    emails, unmapped = build_emails(scores, officers([("SD-1", "SE A", "a@example.com")]), EVAL)
    assert emails == [] and unmapped == ["SD-9"]


def test_only_flagged_meters_are_listed_biggest_risk_first():
    scores = pd.concat([
        flagged([{"meter_id": "small", "subdivision_id": "SD-1", "est_unbilled_kwh": 10.0},
                 {"meter_id": "big", "subdivision_id": "SD-1", "est_unbilled_kwh": 90.0}]),
        flagged([{"meter_id": "clear", "subdivision_id": "SD-1", "est_unbilled_kwh": 0.0, "status": "CLEAR"}])])
    (email,), _ = build_emails(scores, officers([("SD-1", "SE A", "a@example.com")]), EVAL)
    assert list(email.table["Meter"]) == ["big", "small"]
    assert list(email.table.columns) == list(EMAIL_COLUMNS.values())
    assert email.table["Drop %"].tolist() == [60, 60]


def test_subject_and_body_wording():
    scores = flagged([{"meter_id": "1", "subdivision_id": "SD-1", "est_unbilled_kwh": 1234.0}])
    (email,), _ = build_emails(scores, officers([("SD-1", "SE A", "a@example.com")]), EVAL)
    assert email.subject == ("Suspected consumption-drop cases, week to 04 Oct 2026: "
                             "1 flagged for inspection, 1,234 kWh at risk")
    assert "<b>suspected</b>" in email.html and "recommended for inspection" in email.html
    assert COPYRIGHT.replace("©", "&copy;") in email.html or COPYRIGHT in email.html


def test_names_are_escaped_in_the_html():
    scores = flagged([{"meter_id": "<b>1</b>", "subdivision_id": "SD-1", "est_unbilled_kwh": 1.0}])
    (email,), _ = build_emails(scores, officers([("SD-1", "R&D <Ops>", "a@example.com")]), EVAL)
    assert "R&amp;D &lt;Ops&gt;" in email.html
    assert "&lt;b&gt;1&lt;/b&gt;" in email.html
