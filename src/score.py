import hashlib
from datetime import date

import numpy as np
import pandas as pd

from .baseline import windows
from .config import Config
from .suppress import SUPPRESSION_CODES

REASON_TEMPLATE = (
    "Suspected consumption drop, recommended for inspection. "
    "Consumption down {drop_pct:.0f}% since {onset_date}, {excess_drop_pct:.0f} points steeper than "
    "{peer_group_size} comparable connections ({peer_level} peers). "
    "About {est_unbilled_kwh:,.0f} units possibly unbilled. "
    "Meter reporting normally ({comm_availability_pct:.0f}%)."
)

OUTPUT_COLUMNS = [
    "score_id", "meter_id", "eval_date", "baseline_daily_kwh", "current_daily_kwh", "drop_pct",
    "peer_drop_pct", "excess_drop_pct", "drop_kwh_per_day", "onset_date", "est_unbilled_kwh",
    "peer_group_size", "comm_availability_pct", "status", "suppression_code", "reason_text",
    "model_version",
]


def score_id(meter_id: str, eval_date: date, model_version: str) -> str:
    return hashlib.sha1(f"{meter_id}|{pd.Timestamp(eval_date).date()}|{model_version}".encode()).hexdigest()[:16]


def _onset_dates(daily: pd.DataFrame, flagged: pd.DataFrame, eval_date: date, cfg: Config) -> pd.Series:
    """First day in the current window where the forward 5-day median falls below
    onset_fraction of the peer-adjusted baseline. Falls back to the window start."""
    w = windows(eval_date, cfg)
    cur = daily[daily["reading_date"].between(w["current_start"], w["current_end"])
                & daily["meter_id"].isin(flagged["meter_id"])]
    expected = (flagged["baseline_daily_kwh"] * (1 - flagged["peer_drop_pct"] / 100)).set_axis(flagged["meter_id"])
    onsets = {}
    for meter_id, s in cur.sort_values("reading_date").groupby("meter_id"):
        kwh = s["kwh_import"].reset_index(drop=True)
        fwd = kwh[::-1].rolling(5, min_periods=3).median()[::-1]
        below = (fwd < cfg.onset_fraction * expected[meter_id]).to_numpy()
        onsets[meter_id] = s["reading_date"].iloc[int(np.argmax(below))] if below.any() else w["current_start"]
    return flagged["meter_id"].map(onsets).fillna(w["current_start"])


def score_and_explain(df: pd.DataFrame, daily: pd.DataFrame, eval_date: date, cfg: Config) -> pd.DataFrame:
    """Sets status, onset_date, est_unbilled_kwh, reason_text, score_id.

    FLAGGED when excess_drop_pct exceeds threshold and status is not already SUPPRESSED.

    est_unbilled_kwh = (baseline * excess_drop_pct) * days since onset_date. Only the drop
    beyond peers counts, so the seasonal part of the drop is not billed as theft.

    Ranking is by est_unbilled_kwh, not by excess_drop_pct.
    """
    df = df.copy()
    ev = pd.Timestamp(eval_date)
    df["eval_date"] = ev.date()
    df["model_version"] = cfg.model_version
    df["score_id"] = [score_id(m, ev, cfg.model_version) for m in df["meter_id"]]

    flag = (df["status"] != "SUPPRESSED") & (df["excess_drop_pct"] > cfg.excess_drop_threshold_pct)
    df["status"] = np.where(df["status"] == "SUPPRESSED", "SUPPRESSED", np.where(flag, "FLAGGED", "CLEAR"))

    df["onset_date"] = pd.NaT
    df["est_unbilled_kwh"] = np.nan
    if flag.any():
        f = df[flag]
        onset = _onset_dates(daily, f, eval_date, cfg)
        days = (ev - onset).dt.days + 1
        df.loc[flag, "onset_date"] = onset.values
        df.loc[flag, "est_unbilled_kwh"] = (f["baseline_daily_kwh"] * f["excess_drop_pct"] / 100 * days).values

    def reason(r):
        if r["status"] == "FLAGGED":
            return REASON_TEMPLATE.format(**{**r, "onset_date": pd.Timestamp(r["onset_date"]).strftime("%d %b %Y")})
        if r["status"] == "SUPPRESSED":
            return f"Not scored: {SUPPRESSION_CODES[r['suppression_code']]}."
        return ""

    df["reason_text"] = df.apply(reason, axis=1)
    df["onset_date"] = pd.to_datetime(df["onset_date"]).dt.date
    order = {"FLAGGED": 0, "CLEAR": 1, "SUPPRESSED": 2}
    df = df.sort_values(["status", "est_unbilled_kwh"], key=lambda s: s.map(order) if s.name == "status" else -s.fillna(0))
    return df.reset_index(drop=True)
