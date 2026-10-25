"""Pure display helpers used by the app. Kept out of app.py, which runs the whole UI on import, so they can be tested."""
import base64
import re

import pandas as pd

from .config import Config
from .suppress import SUPPRESSION_CODES


def mask(addr: str) -> str:
    name, _, domain = str(addr).partition("@")
    return f"{name[:2]}***@{domain}" if domain else "***"


def shown(addr: str, inbox: str) -> str:
    """The demo inbox is the project owner's own address: it is shown masked everywhere, and only used to send."""
    return mask(addr) if inbox and str(addr).strip().lower() == inbox.lower() else addr


def masked_officers(df: pd.DataFrame, inbox: str) -> pd.DataFrame:
    return df.assign(se_email=df["se_email"].map(lambda a: shown(a, inbox)))


def unmask_officers(df: pd.DataFrame, inbox: str) -> pd.DataFrame:
    """The directory editor shows the demo inbox masked; a row left as that mask still means the demo inbox."""
    return df.assign(se_email=df["se_email"].replace(mask(inbox), inbox)) if inbox else df


def svg_as_img(html: str) -> str:
    """st.html's sanitizer removes inline <svg>, so each drawing becomes an <img> of itself (data URI)."""
    def one(m: re.Match) -> str:
        attrs = m.group(1)
        get = lambda name: (re.search(r'(?<![\w-])' + name + r'="([^"]*)"', attrs) or [None, ""])[1]
        size = f"width:{get('width')}px;height:{get('height')}px;" if get("width") else ""
        svg = m.group(0).replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
        src = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return f'<img src="{src}" alt="{get("aria-label")}" style="{size}{get("style")}">'
    return re.sub(r"<svg([^>]*)>.*?</svg>", one, html, flags=re.S)


def decision_text(r: pd.Series, cfg: Config, html: bool = False) -> str:
    """The flag rule applied to one meter, in plain words."""
    b = (lambda t: f"<b>{t}</b>") if html else (lambda t: f"**{t}**")
    t, floor = cfg.excess_drop_threshold_pct, cfg.min_drop_kwh_per_day
    if pd.isna(r["drop_pct"]) or pd.isna(r["peer_drop_pct"]):
        return (f"There is not enough history to compare this meter against, so it is {b('not flagged')} "
                f"({SUPPRESSION_CODES.get(r['suppression_code'], '').lower() or 'no baseline'}).")
    me, peers, extra = r["drop_pct"], r["peer_drop_pct"], r["excess_drop_pct"]
    s = (f"It now uses {b(f'{me:.0f}% less')} than before ({r['baseline_daily_kwh']:.1f} → "
         f"{r['current_daily_kwh']:.1f} kWh a day). Its neighbours on the same transformer use "
         f"{b(f'{peers:.0f}% less')}, which is the normal seasonal dip. ")
    if extra > 0:
        s += f"So it fell {b(f'{extra:.0f} percentage points more')} than its neighbours. "
    else:
        s += f"So it fell {b('no more than its neighbours')}. "
    over = extra > t
    s += f"The limit is {t:g} points, so this is {b('over the limit' if over else 'within the limit')}. "
    if r["status"] == "FLAGGED":
        return s + f"The drop is also {r['drop_kwh_per_day']:.1f} kWh a day (more than the {floor:g} kWh minimum), " \
                   f"so it is {b('⚑ flagged for inspection')}."
    if r["status"] == "SUPPRESSED":
        why = SUPPRESSION_CODES[r["suppression_code"]].lower()
        return s + (f"But it is {b('not flagged')}: {why}." if over else
                    f"It is {b('not flagged')} ({why}).")
    return s + f"It is {b('not flagged')}."
