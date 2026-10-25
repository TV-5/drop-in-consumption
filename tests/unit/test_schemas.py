import pandas as pd
import pytest

from src.schemas import SCHEMAS, parse, template


def daily_raw(**over):
    r = {"meter_id": ["1", "2"], "reading_date": ["2026-09-01", "2026-09-01"], "kwh_import": ["3.1", "2.0"],
         "intervals_expected": ["48", "48"], "intervals_received": ["47", "48"], "voltage_present_pct": ["100", "99.5"]}
    return pd.DataFrame({**r, **over})


def master_raw(**over):
    r = {"meter_id": ["1"], "consumer_id": ["C1"], "dt_id": ["DT-1"], "subdivision_id": ["SD-1"],
         "tariff_category": ["LT-1 Domestic"], "sanctioned_load_kw": ["3"], "phase_type": ["1P"], "status": ["ACTIVE"],
         "status_changed_on": ["2023-04-11"], "meter_installed_on": ["2023-04-11"], "has_registered_der": ["false"]}
    return pd.DataFrame({**r, **over})


def officers_raw(**over):
    r = {"subdivision_id": ["SD-1"], "se_name": ["SE One"], "se_email": ["se.one@example.com"]}
    return pd.DataFrame({**r, **over})


def errs(raw, kind):
    return " | ".join(parse(raw, kind)[1])


def test_valid_files_parse_with_types():
    df, errors = parse(daily_raw(), "daily")
    assert errors == []
    assert pd.api.types.is_datetime64_any_dtype(df["reading_date"])
    assert df["kwh_import"].tolist() == [3.1, 2.0]
    m, errors = parse(master_raw(has_registered_der=["Yes"]), "master")
    assert errors == [] and bool(m.at[0, "has_registered_der"]) is True
    assert parse(officers_raw(), "officers")[1] == []


@pytest.mark.parametrize("kind", ["daily", "master", "officers"])
def test_missing_columns_rejected(kind):
    _, errors = parse(pd.DataFrame({"foo": ["1"]}), kind)
    assert errors and errors[0].startswith("Missing columns")


def test_empty_file_rejected():
    assert "File has no rows" in errs(daily_raw().iloc[0:0], "daily")


@pytest.mark.parametrize("raw, kind, message", [
    (daily_raw(meter_id=["1", ""]), "daily", "meter_id: 1 blank values"),
    (daily_raw(reading_date=["01/09/2026", "2026-09-01"]), "daily", "not in YYYY-MM-DD format"),
    (daily_raw(kwh_import=["3.1", "abc"]), "daily", "kwh_import: 1 non-numeric values"),
    (daily_raw(meter_id=["1", "1"]), "daily", "duplicate rows on meter_id + reading_date"),
    (daily_raw(kwh_import=["-1", "2"]), "daily", "kwh_import: negative values present"),
    (daily_raw(intervals_received=["49", "48"]), "daily", "intervals_received exceeds intervals_expected"),
    (master_raw(has_registered_der=["maybe"]), "master", "values not true/false"),
    (master_raw(meter_installed_on=[""]), "master", "meter_installed_on: blank values"),
    (officers_raw(se_email=["not-an-email"]), "officers", "se_email: 1 invalid addresses"),
    (officers_raw(subdivision_id=["SD-1", "SD-1"], se_name=["A", "B"], se_email=["a@x.com", "b@x.com"]),
     "officers", "duplicate rows on subdivision_id"),
], ids=["blank", "bad-date", "non-numeric", "duplicate", "negative-kwh", "too-many-readings", "bad-bool",
        "no-install-date", "bad-email", "duplicate-officer"])
def test_bad_values_are_reported(raw, kind, message):
    assert message in errs(raw, kind)


@pytest.mark.parametrize("kind", list(SCHEMAS))
def test_template_has_the_required_header(kind):
    assert list(template(kind).columns) == list(SCHEMAS[kind]["columns"])
