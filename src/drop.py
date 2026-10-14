from datetime import date

import pandas as pd

from .baseline import windows
from .config import Config


def compute_drop(daily: pd.DataFrame, baselines: pd.DataFrame,
                 eval_date: date, cfg: Config) -> pd.DataFrame:
    """Adds current_daily_kwh, drop_pct, drop_kwh_per_day, comm_availability_pct.

    Both the percentage and the absolute kWh are carried forward. A 70% drop
    on a consumer using one unit a day is noise.
    """
    w = windows(eval_date, cfg)
    cur = daily[daily["reading_date"].between(w["current_start"], w["current_end"])]
    g = cur.groupby("meter_id")
    # Days with no row at all count as zero intervals received.
    expected_per_day = daily.groupby("meter_id")["intervals_expected"].median()
    received = g["intervals_received"].sum()

    out = baselines.set_index("meter_id").copy()
    out["current_daily_kwh"] = g["kwh_import"].median()
    expected_total = expected_per_day.reindex(out.index) * cfg.current_window_days
    out["comm_availability_pct"] = (received.reindex(out.index).fillna(0) / expected_total * 100).clip(upper=100)
    out["drop_kwh_per_day"] = out["baseline_daily_kwh"] - out["current_daily_kwh"]
    out["drop_pct"] = out["drop_kwh_per_day"] / out["baseline_daily_kwh"] * 100
    out.loc[out["baseline_daily_kwh"] <= 0, "drop_pct"] = float("nan")
    return out.reset_index()
