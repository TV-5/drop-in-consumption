"""Synthetic population for testing the pipeline when no labelled real data exists.

Usage: python sample_data/generate.py [--meters 500] [--days 180] [--seed 7] [--end 2026-10-04]
Writes daily_consumption.csv, meter_master.csv, officers.csv and ground_truth.csv next to this file.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def generate_population(n_meters: int = 500, days: int = 180, seed: int = 7,
                        end: str = "2026-10-04") -> dict[str, pd.DataFrame]:
    """Returns daily, master, officers and ground_truth dataframes.

    A seasonal swing (about -35% from the baseline window to the current window) is
    applied to every meter, so the peer logic is exercised.
    """
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end)
    dates = pd.date_range(end=end_ts, periods=days, freq="D")
    t = np.arange(days)

    subdivisions = ["SD-JPR-07", "SD-JPR-08"]
    dts = [f"DT-JPR-{140 + i:04d}" for i in range(10)]
    meter_ids = [str(4471000 + i) for i in range(n_meters)]
    dt_of = rng.choice(dts, n_meters)
    sd_of = np.where(np.isin(dt_of, dts[:5]), subdivisions[0], subdivisions[1])
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
        "subdivision_id": sd_of,
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

    officers = pd.DataFrame({
        "subdivision_id": subdivisions,
        "se_name": ["SE Jaipur Circle A", "SE Jaipur Circle B"],
        "se_email": ["se.circle.a@example.com", "se.circle.b@example.com"],
    })
    return {
        "daily_consumption": pd.concat(rows, ignore_index=True),
        "meter_master": master,
        "officers": officers,
        "ground_truth": pd.DataFrame(truth),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meters", type=int, default=500)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--end", default="2026-10-04")
    a = ap.parse_args()
    out_dir = Path(__file__).parent
    for name, df in generate_population(a.meters, a.days, a.seed, a.end).items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"wrote {name}.csv ({len(df):,} rows)")
