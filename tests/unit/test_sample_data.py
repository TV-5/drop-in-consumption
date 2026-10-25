import pandas as pd
import pytest

from sample_data import hierarchy
from sample_data.generate import INJECTIONS, SCENARIOS, generate_population


@pytest.fixture(scope="module")
def small():
    return generate_population(160, 7, "2026-10-04", "SD-JPR-07")


def test_same_seed_same_data(small):
    again = generate_population(160, 7, "2026-10-04", "SD-JPR-07")
    for name in ("daily_consumption", "meter_master", "ground_truth"):
        pd.testing.assert_frame_equal(small[name], again[name])


def test_different_seed_different_data(small):
    other = generate_population(160, 8, "2026-10-04", "SD-JPR-07")
    assert not small["daily_consumption"]["kwh_import"].equals(other["daily_consumption"]["kwh_import"])


def test_every_meter_has_every_day(small):
    master, daily = small["meter_master"], small["daily_consumption"]
    assert len(master) == hierarchy.SUBDIVISIONS["SD-JPR-07"]["meters"]
    counts = daily.groupby("meter_id").size()
    assert set(counts.index) == set(master["meter_id"])
    new = set(small["ground_truth"].query("injected_type == 'new_connection'")["meter_id"])
    assert (counts.drop(list(new)) == 160).all()  # every meter has every day...
    assert (counts[list(new)] == 50).all()        # ...except new connections, connected 50 days ago
    assert pd.to_datetime(daily["reading_date"]).max() == pd.Timestamp("2026-10-04")


def test_planted_scenarios_match_the_plan(small):
    counts = small["ground_truth"]["injected_type"].value_counts().to_dict()
    assert counts == dict(INJECTIONS)
    assert set(counts) | {"normal"} == set(SCENARIOS)


@pytest.mark.parametrize("sd_id", list(hierarchy.SUBDIVISIONS))
def test_network_meters_add_up(sd_id):
    net = hierarchy.network(sd_id)
    assert net["meters"].sum() == hierarchy.SUBDIVISIONS[sd_id]["meters"]
    assert net["dt_id"].is_unique


def test_officer_record_shape():
    off = hierarchy.officer("SD-JPR-07")
    assert list(off.columns) == ["subdivision_id", "se_name", "se_email"] and len(off) == 1
    assert off.at[0, "se_email"].endswith("@example.com")
