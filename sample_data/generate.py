"""Synthetic population for testing the pipeline when no labelled real data exists.

Usage: python sample_data/generate.py [--subdivision SD-JPR-07] [--days 180] [--seed 7] [--end 2026-10-04]
Writes daily_consumption.csv, meter_master.csv, officers.csv and ground_truth.csv next to this file.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sample_data.hierarchy import SUBDIVISIONS, network, officer
except ImportError:  # run as a script from inside sample_data/
    from hierarchy import SUBDIVISIONS, network, officer

# (injected_type, count) - everything else is a normal meter.
INJECTIONS = [
    ("partial_bypass", 15),
    ("full_bypass", 8),
    ("vacancy", 6),
    ("comms_failure", 6),
    ("meter_replaced", 4),
    ("new_connection", 5),
    ("der_added", 5),
    ("disconnected", 3),
    ("small_consumer_drop", 4),
]

# injected_type -> (what is injected, what the pipeline should do with it). Shown to users in the app.
SCENARIOS = {
    "normal": ("Ordinary consumption with day-to-day noise and the shared seasonal swing.",
               "Not flagged: the drop matches its peers"),
    "partial_bypass": ("Consumption steps down to 30-50% of normal and stays there.",
                       "FLAGGED (unless the load is too small to act on)"),
    "full_bypass": ("Consumption steps to 4-8% of normal while supply voltage stays present.",
                    "FLAGGED"),
    "vacancy": ("Consumption falls to almost exactly zero: nobody is living there.",
                "Suppressed as VACANT"),
    "comms_failure": ("Only 10-30 of 48 half-hourly readings arrive, so recorded kWh falls.",
                      "Suppressed as COMM_POOR"),
    "meter_replaced": ("A new meter is installed mid-window and reads 50% lower.",
                       "Suppressed as METER_REPLACED"),
    "new_connection": ("Connected only 50 days ago, so there is no baseline to compare against.",
                       "Suppressed as NEW_CONNECTION"),
    "der_added": ("Rooftop solar is registered and grid import falls by 55%.",
                  "Suppressed as DER_PRESENT"),
    "disconnected": ("The utility disconnects the supply: zero kWh, zero voltage.",
                     "Suppressed as DISCONNECTED"),
    "small_consumer_drop": ("A tiny consumer (about 1.2 kWh/day) drops 70%.",
                            "Suppressed as BELOW_FLOOR: too few kWh to justify a visit"),
}


def generate_population(days: int = 180, seed: int = 7,
                        end: str = "2026-10-04", subdivision: str = "SD-JPR-07") -> dict[str, pd.DataFrame]:
    """Returns daily, master, officers, ground_truth and network dataframes for one sub-division.

    A seasonal swing (about -35% from the baseline window to the current window) is
    applied to every meter, so the peer logic is exercised.
    """
    # Each sub-division gets its own population for the same seed.
    rng = np.random.default_rng([seed, int(subdivision.rsplit("-", 1)[1])])
    end_ts = pd.Timestamp(end)
    dates = pd.date_range(end=end_ts, periods=days, freq="D")
    t = np.arange(days)

    net = network(subdivision)
    n_meters = int(net["meters"].sum())  # every meter installed in the sub-division
    meter_ids = [str(4400000 + int(subdivision.rsplit("-", 1)[1]) * 10000 + i) for i in range(n_meters)]
    dt_of = rng.permutation(np.repeat(net["dt_id"].to_numpy(), net["meters"]))
    tariff = rng.choice(["LT-1 Domestic", "LT-2 Commercial"], n_meters, p=[0.8, 0.2])
    load = np.where(tariff == "LT-1 Domestic", rng.choice([1.0, 2.0, 3.0, 4.0], n_meters),
                    rng.choice([5.0, 7.5, 10.0], n_meters))
    base_kwh = load * rng.uniform(1.8, 3.0, n_meters)
    seasonal = 1.25 - 0.55 * (t / (days - 1))  # summer peak sliding into autumn

    types = np.array(["normal"] * n_meters, dtype=object)
    pool = rng.permutation(n_meters)
    i = 0
    for kind, count in INJECTIONS:
        types[pool[i:i + count]] = kind
        i += count

    master = pd.DataFrame({
        "meter_id": meter_ids,
        "consumer_id": [f"C88{int(m) % 100000:05d}" for m in meter_ids],
        "dt_id": dt_of,
        "subdivision_id": subdivision,
        "tariff_category": tariff,
        "sanctioned_load_kw": load,
        "phase_type": np.where(load > 4, "3P", "1P"),
        "status": "ACTIVE",
        "status_changed_on": "2023-04-11",
        "meter_installed_on": "2023-04-11",
        "has_registered_der": "false",
    })

    rows, truth = [], []
    for k, mid in enumerate(meter_ids):
        kind = types[k]
        kwh = base_kwh[k] * seasonal * rng.lognormal(0, 0.18, days)
        received = np.full(days, 48)
        voltage = np.full(days, 100.0)
        keep = np.ones(days, dtype=bool)
        onset = int(rng.integers(days - 28, days - 18))
        true_unbilled = 0.0

        if kind == "small_consumer_drop":
            kwh = 1.2 * seasonal * rng.lognormal(0, 0.18, days)
            kwh[onset:] *= 0.3
        elif kind == "partial_bypass":
            factor = rng.uniform(0.3, 0.5)
            true_unbilled = kwh[onset:].sum() * (1 - factor)
            kwh[onset:] *= factor
        elif kind == "full_bypass":
            factor = rng.uniform(0.04, 0.08)
            true_unbilled = kwh[onset:].sum() * (1 - factor)
            kwh[onset:] *= factor
        elif kind == "vacancy":
            kwh[onset:] = rng.uniform(0, 0.03, days - onset)
        elif kind == "comms_failure":
            received[onset:] = rng.integers(10, 30, days - onset)
            kwh[onset:] *= received[onset:] / 48
        elif kind == "meter_replaced":
            kwh[onset:] *= 0.5
            master.loc[k, "meter_installed_on"] = dates[onset].strftime("%Y-%m-%d")
        elif kind == "new_connection":
            start = days - 50
            keep[:start] = False
            master.loc[k, "meter_installed_on"] = dates[start].strftime("%Y-%m-%d")
        elif kind == "der_added":
            kwh[onset:] *= 0.45
            master.loc[k, "has_registered_der"] = "true"
        elif kind == "disconnected":
            kwh[onset:] = 0
            voltage[onset:] = 0
            master.loc[k, "status"] = "DISCONNECTED"
            master.loc[k, "status_changed_on"] = dates[onset].strftime("%Y-%m-%d")

        rows.append(pd.DataFrame({
            "meter_id": mid,
            "reading_date": dates[keep].strftime("%Y-%m-%d"),
            "kwh_import": np.round(kwh[keep], 2),
            "intervals_expected": 48,
            "intervals_received": received[keep],
            "voltage_present_pct": voltage[keep],
        }))
        if kind != "normal":
            truth.append({"meter_id": mid, "injected_type": kind,
                          "onset_date": dates[onset].strftime("%Y-%m-%d"),
                          "true_unbilled_kwh": round(true_unbilled, 1)})

    return {
        "daily_consumption": pd.concat(rows, ignore_index=True),
        "meter_master": master,
        "officers": officer(subdivision),
        "ground_truth": pd.DataFrame(truth),
        "network": net,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdivision", default="SD-JPR-07", choices=list(SUBDIVISIONS))
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--end", default="2026-10-04")
    a = ap.parse_args()
    out_dir = Path(__file__).parent
    for name, df in generate_population(a.days, a.seed, a.end, a.subdivision).items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"wrote {name}.csv ({len(df):,} rows)")
