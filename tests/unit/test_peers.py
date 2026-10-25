import numpy as np
import pandas as pd
import pytest

from src.peers import compute_peer_excess
from tests.helpers import CFG, make


def drops(values):
    return pd.DataFrame({"meter_id": [str(i) for i in range(len(values))], "drop_pct": values})


def test_excess_is_drop_beyond_the_neighbours_median():
    _, master = make(n=25)
    out = compute_peer_excess(drops([20.0] * 24 + [70.0]), master, CFG).set_index("meter_id")
    assert out.at["24", "peer_drop_pct"] == 20.0
    assert out.at["24", "excess_drop_pct"] == pytest.approx(50.0)
    assert out.at["24", "peer_level"] == "transformer"
    assert out.at["24", "peer_group_size"] == 25


def test_load_bands_form_separate_peer_groups():
    _, master = make(n=40)
    master["sanctioned_load_kw"] = [3.0] * 20 + [7.5] * 20
    out = compute_peer_excess(drops([10.0] * 20 + [40.0] * 20), master, CFG)
    assert (out["peer_level"] == "transformer").all()
    assert (out["excess_drop_pct"] == 0).all()


def test_small_transformer_groups_fall_back_to_the_sub_division():
    _, master = make(n=40)
    master["dt_id"] = [f"DT-{i % 4}" for i in range(40)]  # 10 per transformer, below the minimum of 20
    out = compute_peer_excess(drops([25.0] * 40), master, CFG)
    assert (out["peer_level"] == "sub-division").all()
    assert (out["peer_group_size"] == 40).all()


def test_meters_without_a_drop_do_not_count_as_peers():
    _, master = make(n=26)
    out = compute_peer_excess(drops([30.0] * 25 + [np.nan]), master, CFG)
    assert (out["peer_group_size"] == 25).all()
    assert pd.isna(out.set_index("meter_id").at["25", "excess_drop_pct"])
