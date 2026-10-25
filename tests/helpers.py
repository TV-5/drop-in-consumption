"""Builders shared by the tests: small synthetic populations and app navigation."""
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import Config

ROOT = Path(__file__).resolve().parents[1]
CFG = Config()
EVAL = pd.Timestamp("2026-10-04")
OWNER = "owner.demo@example.com"  # stands in for the real demo inbox in every test


def make(n=40, days=180, kwh=8.0, seed=0, **master_overrides):
    """A clean population on one transformer: n identical-ish meters with full readings."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=EVAL, periods=days)
    daily = pd.DataFrame([
        {"meter_id": str(i), "reading_date": d, "kwh_import": kwh * rng.lognormal(0, 0.1),
         "intervals_expected": 48, "intervals_received": 48, "voltage_present_pct": 100.0}
        for i in range(n) for d in dates
    ])
    master = pd.DataFrame({
        "meter_id": [str(i) for i in range(n)], "consumer_id": [f"C{i}" for i in range(n)],
        "dt_id": "DT-1", "subdivision_id": "SD-1", "tariff_category": "LT-1 Domestic",
        "sanctioned_load_kw": 3.0, "phase_type": "1P", "status": "ACTIVE",
        "status_changed_on": pd.Timestamp("2023-01-01"), "meter_installed_on": pd.Timestamp("2023-01-01"),
        "has_registered_der": False,
    })
    for k, v in master_overrides.items():
        master[k] = v
    return daily, master


def scale(daily, meter_ids, since, factor):
    """Multiply the given meters' use from `since` onwards (a drop when factor < 1)."""
    m = daily["meter_id"].isin(meter_ids) & (daily["reading_date"] >= since)
    daily.loc[m, "kwh_import"] *= factor


# ------------------------------------------------------------------ app navigation (AppTest)
def app_test() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def button(at: AppTest, *, key=None, starts=None):
    for b in at.button:
        if (key and b.key == key) or (starts and b.label.startswith(starts)):
            return b
    raise AssertionError(f"button not found: key={key} starts={starts}; have {[(b.key, b.label) for b in at.button]}")


def errors(at: AppTest) -> list[str]:
    return [e.value for e in at.exception]


def to_step1(at: AppTest) -> AppTest:
    return button(at, key="intro_start").click().run()


def generate(at: AppTest) -> AppTest:
    return button(at, starts="Generate data for").click().run()


def run_detection(at: AppTest) -> AppTest:
    at = button(at, key="wiz_next").click().run()
    return button(at, starts="Run detection").click().run()


def to_step3(at: AppTest) -> AppTest:
    """Fresh app → generated → detected → Step 3."""
    at = generate(to_step1(at.run()))
    at = run_detection(at)
    return button(at, key="wiz_next").click().run()


def page_bytes(at: AppTest) -> bytes:
    """Everything the app sent to the browser for the current page, as one byte string."""
    out = []

    def walk(node):
        proto = getattr(node, "proto", None)
        if proto is not None:
            out.append(proto.SerializeToString())
        for child in getattr(node, "children", {}).values():
            walk(child)
    walk(at._tree)
    return b"".join(out)
