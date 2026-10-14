from datetime import date

import pandas as pd

from .baseline import build_baselines
from .config import Config
from .drop import compute_drop
from .peers import compute_peer_excess
from .score import OUTPUT_COLUMNS, score_and_explain
from .suppress import apply_suppressions

# Carried alongside the output contract so the UI and emails can group and name meters.
CONTEXT_COLUMNS = ["consumer_id", "dt_id", "subdivision_id", "tariff_category", "sanctioned_load_kw", "peer_level"]


def run_pipeline(daily: pd.DataFrame, master: pd.DataFrame, eval_date: date, cfg: Config) -> pd.DataFrame:
    """One row per meter for eval_date, every meter present: FLAGGED, CLEAR or SUPPRESSED."""
    daily = daily[daily["reading_date"] <= pd.Timestamp(eval_date)]
    baselines = build_baselines(daily, eval_date, cfg)
    drops = compute_drop(daily, baselines, eval_date, cfg)
    peers = compute_peer_excess(drops, master, cfg)
    suppressed = apply_suppressions(peers, master, daily, cfg)
    scored = score_and_explain(suppressed, daily, eval_date, cfg)
    scored = scored.merge(master[["meter_id", "consumer_id"]], on="meter_id", how="left")
    return scored[OUTPUT_COLUMNS + CONTEXT_COLUMNS]
