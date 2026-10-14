import numpy as np
import pandas as pd

from .config import Config

# Evaluated in this order; the first match wins so the reason is the most specific one.
SUPPRESSION_CODES = {
    "NO_MASTER_RECORD": "Meter not found in meter master",
    "COMM_POOR": "Meter communication below threshold; drop may be missing data",
    "DISCONNECTED": "Connection status is not ACTIVE",
    "METER_REPLACED": "Meter installed inside the analysis window",
    "NEW_CONNECTION": "Too little baseline history to compare against",
    "DER_PRESENT": "Registered rooftop solar / DER lowers import legitimately",
    "VACANT": "Near-zero consumption with no residual load; likely vacant premises",
    "BELOW_FLOOR": "Drop too small in absolute kWh to act on",
}


def apply_suppressions(scored: pd.DataFrame, master: pd.DataFrame, daily: pd.DataFrame,
                       cfg: Config) -> pd.DataFrame:
    """Sets status=SUPPRESSED and a suppression_code. Drops no rows."""
    m = master[["meter_id", "status", "meter_installed_on", "has_registered_der"]]
    df = scored.merge(m, on="meter_id", how="left", suffixes=("", "_master"))
    first_reading = daily.groupby("meter_id")["reading_date"].min()
    df["first_reading"] = df["meter_id"].map(first_reading)
    window_start = df["baseline_start"]

    # A replaced meter has readings from before its install date; a new connection does not.
    installed_in_window = df["meter_installed_on"] >= window_start
    has_prior_history = df["first_reading"] < df["meter_installed_on"]
    would_flag = df["excess_drop_pct"] > cfg.excess_drop_threshold_pct

    conditions = {
        "NO_MASTER_RECORD": df["status"].isna(),
        "COMM_POOR": ~(df["comm_availability_pct"] >= cfg.min_comm_availability_pct),
        "DISCONNECTED": df["status"].str.upper() != "ACTIVE",
        "METER_REPLACED": installed_in_window & has_prior_history,
        "NEW_CONNECTION": df["baseline_daily_kwh"].isna(),
        "DER_PRESENT": df["has_registered_der"].fillna(False).astype(bool),
        "VACANT": df["current_daily_kwh"].fillna(0) < cfg.vacancy_kwh_per_day,
        # Only matters for meters that would otherwise flag; the rest are simply CLEAR.
        "BELOW_FLOOR": would_flag & (df["drop_kwh_per_day"] < cfg.min_drop_kwh_per_day),
    }
    codes = list(conditions)
    df["suppression_code"] = np.select([conditions[c].fillna(False).astype(bool) for c in codes],
                                       codes, default="")
    df["status"] = np.where(df["suppression_code"] != "", "SUPPRESSED", "")
    return df.drop(columns=["first_reading", "meter_installed_on", "has_registered_der"])
