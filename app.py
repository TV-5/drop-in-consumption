"""Consumption-drop theft detection MVP: generate data -> detect -> send to SE.

Run with:  streamlit run app.py
"""
import base64
import os
import re
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.baseline import windows
from src.config import load_config
from src.deliver import build_emails, send_emails
from src.pipeline import run_pipeline
from sample_data import hierarchy
from sample_data.generate import SCENARIOS, generate_population
from src.schemas import SCHEMAS, parse
from src.suppress import SUPPRESSION_CODES

ROOT = Path(__file__).parent
LOG_PATH = ROOT / "delivery_log.csv"
DAILY_SEND_LIMIT = 20
# Demo inbox: every email goes here in demo mode, and every SE in the directory defaults to it. Read from the
# server's settings only (MAIL_TO, else the sender's own address) so the owner's address never sits in the code.
DEMO_INBOX = os.getenv("MAIL_TO", "").strip() or os.getenv("SMTP_USER", "").strip()
# Real per-SE delivery is off unless the server opts in. On a public page anyone could otherwise use the
# project's mail account to write to whatever address they type into the directory.
SE_DELIVERY_ALLOWED = os.getenv("ALLOW_SE_DELIVERY", "").strip().lower() == "true"
STEPS = ["1. Generate data", "2. Detect suspected thefts", "3. Send to SE"]
INTRO = "intro"  # landing page; not one of the steps, so it has no place on the progress track

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Consumption-drop detection", page_icon="⚡", layout="wide")
cfg = load_config(ROOT / "config.yaml")
ss = st.session_state
ss.setdefault("data", {})       # kind -> parsed, valid dataframe
ss.setdefault("step", INTRO)  # first visit lands on the intro
ss.setdefault("scores", None)
ss.setdefault("truth", None)  # ground_truth of the generated population
ss.setdefault("eval_date", None)
SEED = 7
MIN_DAYS = cfg.baseline_days + cfg.baseline_gap_days + cfg.current_window_days


def start_over():
    """Home: back to the intro page with a fresh analysis (drops the data, results, sends and all widget state)."""
    for k in list(ss.keys()):
        del ss[k]
    ss.step = INTRO
    ss.scroll_to_top = 1


def go(step: str):
    if step != ss.step:
        ss.scroll_to_top = ss.get("scroll_to_top", 0) + 1  # a new page opens at its top
    ss.step = step


# Streamlit keeps the scroll position when the page underneath changes, so a navigation would otherwise open the
# next page wherever the last one was scrolled to. Runs in the app page via a same-origin iframe; the counter makes
# each navigation's iframe new, so it runs again even when two navigations happen in a row.
SCROLL_TO_TOP_JS = """<script>
(function () {
  const d = window.parent.document;
  function top() {
    window.parent.scrollTo(0, 0);
    d.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main')
      .forEach(function (el) { el.scrollTop = 0; });
  }
  top(); setTimeout(top, 80); setTimeout(top, 300);
})();
</script><!-- navigation NAV -->"""


def scroll_to_top():
    n = ss.pop("scroll_to_top", 0)
    if n:
        with st.container(key="scroll_js"):
            st.iframe(SCROLL_TO_TOP_JS.replace("NAV", str(n)), height=1)


def all_loaded() -> bool:
    return all(k in ss.data for k in SCHEMAS)


# ---------------------------------------------------------------- page-wide styles
def app_styles():
    """App bar, title button, progress track, headings, wizard footer and intro page."""
    st.html("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@700;800&family=IBM+Plex+Mono:wght@500&display=swap');
  /* Dark app bar: Streamlit's own top bar, recoloured, with the project title drawn into it. */
  [data-testid="stHeader"] { background: #151821 !important; }
  /* The title is a real button laid over the bar: clicking it starts over. */
  .st-key-home { position: fixed; top: 0; left: 5rem; height: 3.75rem; z-index: 999992; display: flex;
                 align-items: center; max-width: calc(100% - 20rem); }
  .st-key-home button { background: transparent !important; border: none !important; padding: 0 !important;
                        min-height: 0 !important; box-shadow: none !important; }
  .st-key-home button p { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.3rem;
                          letter-spacing: -.01em; color: #fff !important; white-space: nowrap; overflow: hidden;
                          text-overflow: ellipsis; }
  .st-key-home button:hover p { opacity: .85; }
  [data-testid="stHeader"] button, [data-testid="stHeader"] a, [data-testid="stHeader"] span,
  [data-testid="stHeader"] svg { color: #fff !important; }
  [data-testid="stAppDeployButton"] button { border: 1px solid rgba(255,255,255,.3) !important; border-radius: 8px; }
  [data-testid="stMainBlockContainer"] { padding-top: 5rem; }

  /* Connected progress track: one segment per step, label and status underneath. */
  /* Page heading: small step label above a Manrope title, intro at reading width */
  .eyebrow { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .75rem;
             letter-spacing: .18em; color: #2459d6; margin: 8px 0 4px; }
  .page-title { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 2.3rem; letter-spacing: -.02em;
                margin: 0 0 6px; padding: 0; line-height: 1.15; }
  .page-intro { max-width: 46rem; line-height: 1.6; margin: 0; }

  /* Wizard footer: progress line, Back, step dots, Next */
  /* Intro page: hero band and closing call to action */
  .st-key-intro_hero { background: #f6f8fc; border: 1px solid #e3e7ef; border-radius: 16px;
                       padding: clamp(24px, 4vw, 48px); }
  .st-key-intro_cta { background: #151821; border-radius: 16px; padding: clamp(24px, 4vw, 44px); }
  .st-key-intro_hero button[kind="primary"], .st-key-intro_cta button[kind="primary"] {
    min-height: 48px; padding: 0 24px; font-weight: 700; }
  .intro-read { display: inline-flex; align-items: center; justify-content: center; min-height: 48px;
                padding: 0 24px; margin-top: 28px; border-radius: 8px; background: #2459d6; color: #ffffff !important;
                font-weight: 700; font-size: 16px; text-decoration: none !important; }
  .intro-read:hover { background: #1d4bb8; }
  .st-key-intro_start button { padding: 0 !important; min-height: 0 !important; }
  .st-key-intro_start button p { color: #2459d6; font-weight: 600; font-size: 15px; }
  .st-key-intro_start button:hover p { text-decoration: underline; }
  .intro-meta { font-size: 14px; color: #5b6172; }
  /* Intro sections: each heading pins under the app bar while its section is read and the next one pushes it out.
     One amber strip, pinned right under the app bar on a track of the bar's colour, shows progress through the whole explainer and is full at
     its end (scroll-driven, no script). */
  [data-testid="stMain"] { scroll-behavior: smooth; }
  .sec { scroll-margin-top: 3.75rem; }
  .sec-head { position: sticky; top: 3.75rem; z-index: 5; background: #ffffff; padding: 16px 0 12px; margin-bottom: 12px; }
  .sec-meta { display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
              font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
  .sec-label { letter-spacing: .16em; color: #2459d6; }
  .sec-count { color: #5b6172; }
  .sec-head h2 { font-family: 'Manrope', sans-serif !important; font-weight: 800; font-size: clamp(22px, 2.6vw, 32px);
                 line-height: 1.15; letter-spacing: -.02em; margin: 6px 0 0; padding: 0; color: #151821; }
  .read-track { position: fixed; top: 3.75rem; left: 0; right: 0; height: 4px; z-index: 6; background: #151821; }
  .read-bar { height: 4px; background: #ffb020; box-shadow: 0 0 10px rgba(255,176,32,.75); transform-origin: left;
              animation: sec-fill linear both;
              animation-timeline: --explainer; animation-range: contain 0% contain 100%; }
  @keyframes sec-fill { from { transform: scaleX(0); } to { transform: scaleX(1); } }
  @supports not (animation-timeline: view()) { .read-track { display: none; } }
  .st-key-scroll_js { height: 0; overflow: hidden; margin: 0; }
  .st-key-wizard_footer { margin-top: 2rem; border-top: 1px solid rgba(128,128,128,.2); padding-top: 0; }
  .wiz-line { height: 3px; background: #e6e7eb; margin-top: -1px; }
  .wiz-line div { height: 3px; background: #2459d6; }
  .st-key-wizard_footer [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; align-items: center; }
  .st-key-wizard_footer [data-testid="stColumn"] { min-width: 0 !important; }
  .wiz-mid { display: flex; justify-content: center; align-items: center; gap: 6px; font-size: .8rem; opacity: .85; }
  .wiz-dot { width: 7px; height: 7px; border-radius: 4px; background: #cfd3dc; }
  .wiz-dot.on { width: 22px; background: #2459d6; }
  .wiz-mid span { margin-left: 10px; white-space: nowrap; }
  @media (max-width: 640px) { .wiz-mid span { display: none; } }
  [class*="st-key-stepper"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 8px !important; }
  [class*="st-key-stepper"] [data-testid="stColumn"] { flex: 1 1 0 !important; width: auto !important; min-width: 0 !important; }
  [class*="st-key-stepper"] button { justify-content: flex-start; border: none !important; border-radius: 0 !important;
                           background: transparent !important; box-shadow: none !important; min-height: 0 !important;
                           padding: 12px 0 2px !important; position: relative; }
  [class*="st-key-stepper"] button > div { justify-content: flex-start; width: 100%; }
  [class*="st-key-stepper"] button p { text-align: left; }
  [class*="st-key-stepper"] button::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
                                   border-radius: 3px; background: #e6e7eb; }
  [class*="st-key-stepper"] [class*="_ok"] button::before, [class*="st-key-stepper"] [class*="_current"] button::before {
    background: #2459d6; }
  [class*="st-key-stepper"] button p { font-size: .85rem; font-weight: 700; color: #3c3f4a; white-space: nowrap;
                             overflow: hidden; text-overflow: ellipsis; }
  [class*="st-key-stepper"] [class*="_current"] button p { color: #2459d6; }
  [class*="st-key-stepper"] button p span { font-size: .75rem; font-weight: 400; margin-left: 6px; }
  [class*="st-key-stepper"] button:disabled { opacity: .7; }
  [class*="st-key-stepper"] button:not(:disabled):hover p { text-decoration: underline; }
  @media (max-width: 640px) {
    .st-key-home { left: 1rem; max-width: calc(100% - 8rem); }
    .st-key-home button p { font-size: 1.05rem; }
    [class*="st-key-stepper"] button p { font-size: .78rem; }
    [class*="st-key-stepper"] button p span { display: none; }
  }
</style>""")


# ---------------------------------------------------------------- progress ribbon
STEP_NAMES = ["Generate data", "Detect thefts", "Send to SE"]


def stepper():
    """Progress track under the app bar: where you are, what each step produced, and which steps can be opened yet."""
    scores = ss.scores
    n_flagged = 0 if scores is None else int((scores["status"] == "FLAGGED").sum())
    res = ss.get("last_result")
    n_sent = 0 if res is None else int(res["status"].isin(["SENT", "TEST_SENT"]).sum())
    done = [all_loaded(), scores is not None, n_sent > 0]
    reachable = [True, all_loaded(), n_flagged > 0]
    if all_loaded():
        daily = ss.data["daily"]
        sub1 = f"{daily['meter_id'].nunique():,} meters · {daily['reading_date'].nunique()} days"
    else:
        sub1 = "start here"
    sub2 = f"{n_flagged} suspected" if scores is not None else ("ready to run" if all_loaded() else "needs data")
    sub3 = f"{n_sent} sent" if n_sent else ("ready to send" if n_flagged else "needs flagged meters")
    subs = [sub1, sub2, sub3]
    cur = STEPS.index(ss.step)
    cols = st.columns(len(STEPS))
    for i, (col, name) in enumerate(zip(cols, STEP_NAMES)):
        # Segment colour comes from CSS keyed on this name: "_current" and "_ok" (done) are blue.
        state = "current" if i == cur else "done" if done[i] else "todo"
        mark = "✓" if done[i] and i != cur else str(i + 1)
        col.button(f"{mark} · {name} :gray[{subs[i]}]", key=f"step{i}_{state}{'_ok' if done[i] else ''}",
                   width="stretch", disabled=not reachable[i], help=subs[i], on_click=go, args=(STEPS[i],))


def page_heading(i: int, title: str, intro: str = ""):
    st.html(f'<div class="eyebrow">STEP {i + 1} OF {len(STEPS)}</div><h1 class="page-title">{title}</h1>'
            + (f'<p class="page-intro">{intro}</p>' if intro else ""))


NEXT_LABELS = ["Detect thefts →", "Send to SE →"]
NEXT_BLOCKED = ["Generate data first", "Run detection and flag at least one meter first"]


def wizard_footer():
    """Back / step dots / Next, at the end of every page, so moving on never needs a scroll back up."""
    cur = STEPS.index(ss.step)
    scores = ss.scores
    reachable = [True, all_loaded(), scores is not None and bool((scores["status"] == "FLAGGED").any())]
    dots = "".join(f'<div class="wiz-dot{" on" if i == cur else ""}"></div>' for i in range(len(STEPS)))
    st.html(f'<div class="wiz-line"><div style="width:{(cur + 1) / len(STEPS) * 100:.1f}%"></div></div>')
    c = st.columns([1, 2, 1], vertical_alignment="center")
    c[0].button("← Back", key="wiz_back", disabled=cur == 0, on_click=go, args=(STEPS[max(cur - 1, 0)],))
    c[1].html(f'<div class="wiz-mid">{dots}<span>Step {cur + 1} of {len(STEPS)}</span></div>')
    if cur < len(STEPS) - 1:
        ok = reachable[cur + 1]
        with c[2]:
            st.button(NEXT_LABELS[cur], key="wiz_next", type="primary", width="stretch", disabled=not ok,
                      help=None if ok else NEXT_BLOCKED[cur], on_click=go, args=(STEPS[cur + 1],))


# ---------------------------------------------------------------- intro
INTRO_HTML = (ROOT / "content" / "intro.html").read_text(encoding="utf-8")


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


def intro_page():
    """What the project is, for a reader from outside the power sector. Text lives in content/intro.html."""
    html = svg_as_img(INTRO_HTML.replace("{THRESHOLD}", f"{cfg.excess_drop_threshold_pct:g}")
                      .replace("{FLOOR}", f"{cfg.min_drop_kwh_per_day:g}"))
    hero_text, hero_card, body, cta = html.split("<!-- split -->")
    with st.container(key="intro_hero"):
        c = st.columns([1.15, 1], gap="large", vertical_alignment="center")
        with c[0]:
            # Reading comes first: the explainer is the main action, the demo a small skip link.
            st.html(hero_text + '<a class="intro-read" href="#explainer">Read the explainer ↓</a>')
            st.button("or skip straight to the demo →", type="tertiary", key="intro_start",
                      on_click=go, args=(STEPS[0],))
            st.html('<div class="intro-meta">The demo: 3 steps · about 3 minutes · generated data, '
                    'no real customers</div>')
        c[1].html(hero_card)
    st.html(body)
    with st.container(key="intro_cta"):
        st.html(cta)
        st.button("Start the demo →", type="primary", key="intro_start_bottom", on_click=go, args=(STEPS[0],))


# ---------------------------------------------------------------- step 1
@st.cache_data(show_spinner=False)
def generate(days: int, end: str, subdivision: str) -> dict[str, pd.DataFrame]:
    # Fixed seed: the same area and settings always give the same data. Real data would come from the MDM database.
    return generate_population(days, SEED, end, subdivision)


def scenario_label(kind: str) -> str:
    return {"der_added": "Rooftop solar (DER) added"}.get(kind, kind.replace("_", " ").capitalize())


def step_generate():
    page_heading(0, "Generate data",
                 "This is a mock project, so instead of asking you for real utility data, the app builds a synthetic "
                 "population of smart meters. A handful of meters have known problems planted in them, so after "
                 "detection you can see exactly which ones the pipeline caught and which it correctly ignored.")

    st.subheader("Choose the area to analyse")
    st.write("One sub-division is analysed at a time, the same unit an inspection list is sent for.")
    sd_id = st.selectbox("Sub-division", list(hierarchy.SUBDIVISIONS), format_func=hierarchy.label, key="sd_pick")
    show_hierarchy(sd_id)

    st.subheader("Set up the data")
    st.info("**Why the data is generated here.** This demo creates realistic meter readings so the analysis can "
            "be shown end to end. On top of normal usage, we deliberately inject different patterns into some "
            "meters, such as a meter bypass, a vacant house, new rooftop solar or a meter that stopped "
            "communicating, so every scenario the pipeline must handle is covered.\n\n"
            "**In real life** there is no generation step. Readings come straight from the utility's live meter "
            "database, and this service runs in batches (for example, once a week) to produce the inspection list.",
            icon="ℹ️")
    st.write(f"Data is generated for all **{hierarchy.SUBDIVISIONS[sd_id]['meters']:,} smart meters** installed in "
             f"{hierarchy.SUBDIVISIONS[sd_id]['name']}. Choose how much history to create.")
    # Not a form: the timeline below has to follow the date as it is picked.
    c = st.columns(2)
    days = c[0].number_input("Days of history", MIN_DAYS, 365, 180, step=10,
                             help=f"At least {MIN_DAYS} days, so every meter has a full 'before' and 'now' period.")
    end = c[1].date_input("Last reading date", value=pd.Timestamp("2026-10-04"),
                          help="The most recent day of readings. Step 2 runs the check as of this date by default.")
    with st.container(border=True):
        show_windows(end, pd.Timestamp(end) - pd.Timedelta(days=int(days) - 1), end,
                     lead=f"**How the {int(days)} days of history are used.** In Step 2 the check runs as of the "
                          "last reading date by default, and compares the 'before' and 'now' periods:")
    submitted = st.button(f"Generate data for {hierarchy.SUBDIVISIONS[sd_id]['name']}", type="primary")

    if submitted:
        with st.spinner("Generating meters…"):
            data = generate(int(days), str(end), sd_id)
        for kind, spec in SCHEMAS.items():
            df, errors = parse(data[spec["filename"].removesuffix(".csv")].astype(str), kind)
            for e in errors:
                st.error(e)
            ss.data[kind] = df
        ss.data["officers"] = default_officers(sd_id)
        ss.officer_base = ss.data["officers"]
        ss.officer_ver = ss.get("officer_ver", 0) + 1  # fresh directory editor for the new data
        ss.truth = data["ground_truth"]
        ss.sd_id = sd_id
        ss.scores = None  # new data invalidates any previous run
        ss.preview = None
        ss.pop("last_result", None)

    if not all_loaded():
        st.info("Choose a sub-division and generate the data to continue.")
        return
    if ss.sd_id != sd_id:
        st.warning(f"The data below is still for {hierarchy.label(ss.sd_id)}. "
                   "Generate again to analyse the sub-division selected above.")

    st.success(f"Data generated for {hierarchy.label(ss.sd_id)}.")
    explain_data()
    explain_scenarios()
    show_example()


def show_hierarchy(sd_id: str):
    sd = hierarchy.SUBDIVISIONS[sd_id]
    div = hierarchy.DIVISIONS[sd["division"]]
    net = hierarchy.network(sd_id)
    n_feeders, n_dts = net["feeder_id"].nunique(), len(net)
    st.html(f"""
<style>
  .net-flow {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 4px 0 8px; }}
  .net-box {{ padding: 18px 18px 16px; border-radius: 10px; border: 1px solid #e3e7ef; background: #f6f8fc; }}
  .net-num {{ font-family: 'Manrope', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1.1;
             color: #2459d6; margin-bottom: 6px; }}
  .net-lbl {{ font-size: .95rem; font-weight: 700; }}
  .net-sub {{ font-size: .8rem; opacity: .7; margin-top: 2px; }}
  @media (max-width: 640px) {{ .net-flow {{ grid-template-columns: 1fr; }} }}
</style>
<div class="net-flow">
  <div class="net-box"><div class="net-num">{sd['meters']:,}</div><div class="net-lbl">smart meters</div>
    <div class="net-sub">one in every home or shop</div></div>
  <div class="net-box"><div class="net-num">{n_feeders}</div><div class="net-lbl">feeders</div>
    <div class="net-sub">11 kV lines that carry power into the area</div></div>
  <div class="net-box"><div class="net-num">{n_dts}</div><div class="net-lbl">transformers (DTs)</div>
    <div class="net-sub">the pole transformers each street hangs off</div></div>
</div>""")
    st.write(f"In plain words: the **{sd['meters']:,} meters** in {sd['name']} get their power through "
             f"**{n_feeders} feeders**, and those feeders split into **{n_dts} transformers**. That works out to "
             f"about **{sd['meters'] // n_dts} meters per transformer**. Homes on the same transformer are "
             "neighbours, so the pipeline compares each meter with them.")

    with st.expander("See the full network: substations, feeders and transformers"):
        by_feeder = (net.groupby(["substation_id", "feeder_id"], sort=False)
                     .agg(Transformers=("dt_id", "size"), Meters=("meters", "sum")).reset_index()
                     .rename(columns={"substation_id": "Substation", "feeder_id": "Feeder"}))
        st.markdown(f"**{net['substation_id'].nunique()} substations → {n_feeders} feeders → {n_dts} transformers**")
        st.dataframe(by_feeder, width="stretch", hide_index=True)
        feeder = st.selectbox("Pick a feeder to see its transformers", by_feeder["Feeder"], key="feeder_pick")
        dts = net[net["feeder_id"] == feeder]
        st.caption(f"{feeder} serves {dts['meters'].sum():,} meters through {len(dts)} transformers.")
        st.dataframe(dts[["dt_id", "meters"]].rename(columns={"dt_id": "Transformer (DT)", "meters": "Meters"}),
                     width="stretch", hide_index=True)

    with st.expander("See the office hierarchy: who is responsible for this area"):
        st.markdown("\n".join([
            f"- **Circle** · {hierarchy.CIRCLE['name']} ({hierarchy.CIRCLE['id']}) · {hierarchy.CIRCLE['se_name']}",
            f"    - **Division** · {div['name']} ({sd['division']}) · {div['ee_name']}",
            f"        - **Sub-division** · {sd['name']} ({sd_id}) · {sd['ae_name']}",
        ]))
        st.caption("The inspection list for this sub-division is emailed to the circle SE in Step 3.")


def explain_data():
    st.subheader("What the data contains")
    sd = hierarchy.SUBDIVISIONS[ss.sd_id]
    st.write(f"Smart meters in {sd['name']} sub-division, spread across its {len(hierarchy.network(ss.sd_id))} "
             "distribution transformers (DTs). Every meter belongs to one DT, which is how the pipeline finds "
             "comparable neighbours (peers).")
    t1, t2, t3 = st.tabs(["Daily consumption", "Meter master", "Officers"])
    with t1:
        st.write("One row per meter per day: the kWh imported from the grid, how many of the 48 half-hourly "
                 "readings actually arrived, and the share of the day supply voltage was present.")
        show_summary("daily", ss.data["daily"])
        st.dataframe(ss.data["daily"].head(200), width="stretch", hide_index=True, height=220)
    with t2:
        st.write("One row per meter: who the consumer is, which DT and sub-division it sits on, its tariff and "
                 "sanctioned load, whether it is still connected, when it was installed, and whether rooftop "
                 "solar is registered.")
        show_summary("master", ss.data["master"])
        st.dataframe(ss.data["master"], width="stretch", hide_index=True, height=220)
    with t3:
        st.write("Which Superintending Engineer (SE) receives this sub-division's weekly list of suspected "
                 f"cases. Every SE defaults to the demo inbox ({shown(DEMO_INBOX) or 'not configured'}); Step 3 lets you edit the list or "
                 "upload the utility's own.")
        show_summary("officers", ss.data["officers"])
        st.dataframe(masked_officers(ss.data["officers"]), width="stretch", hide_index=True)


def explain_scenarios():
    st.subheader("Scenarios the data covers")
    t = cfg.excess_drop_threshold_pct
    st.html(f"""
<style>
  .rule {{ border: 2px solid #eb6834; background: rgba(235,104,52,.08); border-radius: 12px; padding: 16px 18px; }}
  .rule-head {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; }}
  .rule-num {{ font-size: 2.4rem; font-weight: 800; color: #eb6834; line-height: 1; }}
  .rule-title {{ font-size: 1.1rem; font-weight: 700; }}
  .rule p {{ margin: 10px 0 0; }}
  .rule ul {{ margin: 6px 0 0; padding-left: 20px; }}
</style>
<div class="rule">
  <div class="rule-head"><span class="rule-num">{t:g}%</span>
    <span class="rule-title">The flag rule: a drop of more than {t:g} percentage points beyond the neighbours</span></div>
  <p>A meter is flagged for <b>drop in consumption</b> when its usage has fallen <b>more than {t:g} percentage
    points further than the other meters on the same transformer</b>.</p>
  <p><b>Example:</b> if the neighbours used 35% less this month, a meter that used 60% less is 25 points beyond
    them and gets flagged. A meter that used 50% less is only 15 points beyond them and is left alone.</p>
  <p>To avoid false alarms, a meter is also left alone when:</p>
  <ul>
    <li>the drop is smaller than <b>{cfg.min_drop_kwh_per_day:g} kWh a day</b> (too small to be worth a visit)</li>
    <li>fewer than <b>{cfg.min_comm_availability_pct:g}%</b> of its readings arrived (the drop may just be missing data)</li>
    <li>there is an innocent reason: rooftop solar, a vacant house, a new or replaced meter, or a disconnection</li>
  </ul>
</div>""")
    st.write("**Every** meter's consumption slides by about 35% from the summer peak into autumn. That "
             "is the trap: a naive \"consumption dropped\" rule would flag the whole town. The pipeline has to "
             "compare each meter with its peers on the same transformer, and only flag drops well beyond theirs.")
    truth = ss.truth
    counts = truth["injected_type"].value_counts()
    n_normal = len(ss.data["master"]) - len(truth)
    table = pd.DataFrame([
        {"Scenario": scenario_label(k),
         "Meters": n_normal if k == "normal" else int(counts.get(k, 0)),
         "What was injected": what, "What detection should do": expect}
        for k, (what, expect) in SCENARIOS.items()
    ])
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={"What was injected": st.column_config.TextColumn(width="large"),
                                "What detection should do": st.column_config.TextColumn(width="medium")})
    st.caption("Only the two bypass scenarios are theft. Every other drop has an innocent explanation, and the "
               "pipeline must recognise it rather than send an engineer to someone's door.")


def example_kinds() -> list[str]:
    counts = ss.truth["injected_type"].value_counts()
    return [k for k in SCENARIOS if k != "normal" and k in counts]


def example_for(kind: str) -> tuple[pd.Series, pd.Series]:
    """(ground-truth row, detection result) for the first meter injected with this scenario."""
    row = ss.truth[ss.truth["injected_type"] == kind].iloc[0]
    res = preview_scores(ss.data["daily"]["reading_date"].max().date())
    return row, res[res["meter_id"] == row["meter_id"]].iloc[0]


def teaser(body_html: str):
    """Medium-style preview: the text fades out, and the buttons below decide whether to read on."""
    st.html(f"""
<style>
  .teaser {{ -webkit-mask-image: linear-gradient(to bottom, #000 35%, transparent 100%);
            mask-image: linear-gradient(to bottom, #000 35%, transparent 100%); max-height: 170px; overflow: hidden; }}
  .teaser p {{ margin: 0 0 10px; line-height: 1.55; }}
</style>
<div class="teaser">{body_html}</div>""")


def show_example():
    """Faded teaser by default, so moving on is one obvious click; the full walk-through is opt-in."""
    ss.setdefault("show_example", False)
    st.subheader("See an example meter")
    if not ss.show_example:
        kind = example_kinds()[0]
        row, res = example_for(kind)
        teaser("<p>Follow one meter from the data, step by step: who it belongs to, what it used to use, what it "
               "uses now, and <b>exactly where on its chart the pipeline flagged it</b>, or why it decided to leave "
               f"it alone.</p><p><b>{scenario_label(kind)}, meter {row['meter_id']}:</b> "
               f"{decision_text(res, html=True)}</p>")
        st.button("Read the full example ↓", on_click=lambda: ss.update(show_example=True))
        return

    kind = st.selectbox("Pick a scenario", example_kinds(), format_func=scenario_label)
    row, res = example_for(kind)
    eval_date = ss.data["daily"]["reading_date"].max().date()
    st.caption(f"Detection result as of the last reading date, {eval_date:%d %b %Y}. "
               "Step 2 lets you run it for another date.")
    meter_drilldown(res, eval_date, injected_onset=row["onset_date"],
                    note=f"{SCENARIOS[kind][0]} Expected outcome: {SCENARIOS[kind][1]}.")
    st.button("Hide the example ↑", on_click=lambda: ss.update(show_example=False))


def meter_drilldown(res: pd.Series, eval_date, injected_onset=None, note: str = ""):
    """Everything about one meter's result: who it is, the decision in plain words, its chart and raw readings."""
    meter_id = res["meter_id"]
    daily, master = ss.data["daily"], ss.data["master"]
    show_meter_card(master[master["meter_id"] == meter_id].iloc[0])
    show_verdict(res)
    meter_chart(res, eval_date, injected_onset)
    st.caption((note + " " if note else "") + "Thick blue lines: what the meter used to use, and what it would "
               "use now if it had dropped only as much as its neighbours.")
    with st.expander(f"Daily readings for meter {meter_id}"):
        st.dataframe(daily[daily["meter_id"] == meter_id].sort_values("reading_date", ascending=False)
                     .rename(columns={"reading_date": "Date", "kwh_import": "kWh used",
                                      "intervals_expected": "Readings expected", "intervals_received": "Readings arrived",
                                      "voltage_present_pct": "Supply on (%)"}).drop(columns="meter_id"),
                     width="stretch", hide_index=True, height=260)


def meter_chart(res: pd.Series, eval_date, injected_onset=None):
    daily, master = ss.data["daily"], ss.data["master"]
    meter_id = res["meter_id"]
    w = windows(eval_date, cfg)
    dt = master.loc[master["meter_id"] == meter_id, "dt_id"].iloc[0]
    neighbours = set(master.loc[master["dt_id"] == dt, "meter_id"]) - {meter_id}
    typical = (daily[daily["meter_id"].isin(neighbours)].groupby("reading_date")["kwh_import"].median()
               .reset_index().assign(series="Typical neighbour (same transformer)"))
    example = daily.loc[daily["meter_id"] == meter_id, ["reading_date", "kwh_import"]].assign(
        series=f"Meter {meter_id}")
    bands = alt.Chart(pd.DataFrame([
        {"start": w["baseline_start"], "end": w["baseline_end"], "window": "Before (baseline)"},
        {"start": w["current_start"], "end": w["current_end"], "window": "Now (current)"},
    ])).mark_rect(opacity=0.12).encode(
        x="start:T", x2="end:T",
        fill=alt.Fill("window:N", title=None, legend=alt.Legend(orient="bottom"),
                      scale=alt.Scale(range=["#2a78d6", "#eb6834"])))
    lines = alt.Chart(pd.concat([typical, example])).mark_line(strokeWidth=2).encode(
        x=alt.X("reading_date:T", title=None), y=alt.Y("kwh_import:Q", title="kWh per day"),
        stroke=alt.Stroke("series:N", title=None, legend=alt.Legend(orient="top"),
                          scale=alt.Scale(domain=[f"Meter {meter_id}", "Typical neighbour (same transformer)"],
                                          range=["#1a1a1a", "#8a8984"])),
        tooltip=[alt.Tooltip("reading_date:T", format="%d %b %Y"), alt.Tooltip("kwh_import:Q", format=".2f")])
    layers = [bands, lines]
    # Legend instead of on-chart text: the two dates are often days apart and the labels would overlap.
    marks = []
    if injected_onset is not None:
        marks.append({"d": pd.Timestamp(injected_onset), "event": "Problem injected"})
    if res["status"] == "FLAGGED" and pd.notna(res["onset_date"]):
        marks.append({"d": pd.Timestamp(res["onset_date"]), "event": "⚑ Flagged: drop starts here"})
    if marks:
        layers.append(alt.Chart(pd.DataFrame(marks)).mark_rule(strokeDash=[4, 3], strokeWidth=2.5).encode(
            x="d:T", tooltip=[alt.Tooltip("event:N", title="Event"), alt.Tooltip("d:T", title="Date", format="%d %b %Y")],
            color=alt.Color("event:N", title=None, legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=["Problem injected", "⚑ Flagged: drop starts here"],
                                            range=["#8a8984", "#d62728"]))))
    if pd.notna(res["baseline_daily_kwh"]):
        levels = [{"start": w["baseline_start"], "end": w["baseline_end"], "y": res["baseline_daily_kwh"]}]
        if pd.notna(res["peer_drop_pct"]):
            levels.append({"start": w["current_start"], "end": w["current_end"],
                           "y": res["baseline_daily_kwh"] * (1 - res["peer_drop_pct"] / 100)})
        layers.append(alt.Chart(pd.DataFrame(levels)).mark_rule(strokeWidth=3, color="#2a78d6").encode(
            x="start:T", x2="end:T", y="y:Q"))
    chart = alt.layer(*layers).resolve_scale(color="independent", fill="independent", stroke="independent")
    st.altair_chart(chart.properties(height=300), width="stretch")


def preview_scores(eval_date) -> pd.DataFrame:
    """Detection run used only to explain the example meters in Step 1. Cleared whenever data is regenerated."""
    if ss.get("preview") is None:
        ss.preview = run_pipeline(ss.data["daily"], ss.data["master"], eval_date, cfg)
    return ss.preview


def show_meter_card(m: pd.Series):
    with st.container(border=True):
        st.markdown(f"**Meter {m['meter_id']}** · consumer {m['consumer_id']}")
        c = st.columns(3)
        c[0].markdown(f"Transformer  \n**{m['dt_id']}**")
        c[1].markdown(f"Tariff · load  \n**{m['tariff_category']} · {m['sanctioned_load_kw']:g} kW**")
        c[2].markdown(f"Connection  \n**{m['status']}**")
        c = st.columns(3)
        c[0].markdown(f"Meter installed  \n**{pd.Timestamp(m['meter_installed_on']):%d %b %Y}**")
        c[1].markdown(f"Phase  \n**{m['phase_type']}**")
        c[2].markdown(f"Rooftop solar  \n**{'Yes' if m['has_registered_der'] else 'No'}**")


def show_verdict(r: pd.Series):
    def num(v, fmt):
        return "n/a" if pd.isna(v) else format(v, fmt)
    status = r["status"]
    if status == "FLAGGED":
        st.error(f"**⚑ FLAGGED for inspection.** {r['reason_text']}")
    elif status == "SUPPRESSED":
        st.warning(f"**Not flagged: {r['suppression_code']}.** {SUPPRESSION_CODES[r['suppression_code']]}.")
    else:
        st.success("**Not flagged.** Its drop is in line with the neighbours on its transformer.")
    c = st.columns(4)
    c[0].metric("Used to use (kWh/day)", num(r["baseline_daily_kwh"], ".1f"))
    c[1].metric("Uses now (kWh/day)", num(r["current_daily_kwh"], ".1f"),
                None if pd.isna(r["drop_pct"]) else f"-{r['drop_pct']:.0f}%")
    c[2].metric("Neighbours' drop", num(r["peer_drop_pct"], ".0f") + "%",
                help="Typical drop of the other meters on the same transformer. Mostly the season.")
    c[3].metric("Readings arrived", num(r["comm_availability_pct"], ".0f") + "%",
                help=f"Share of the meter's readings that reached the system. Below "
                     f"{cfg.min_comm_availability_pct:g}% the drop may just be missing data.")
    with st.container(border=True):
        st.markdown("**How the decision was made**  \n" + decision_text(r))


def decision_text(r: pd.Series, html: bool = False) -> str:
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


def show_summary(kind: str, df: pd.DataFrame):
    if kind == "daily":
        m = st.columns(4)
        m[0].metric("Rows", f"{len(df):,}")
        m[1].metric("Meters", f"{df['meter_id'].nunique():,}")
        m[2].metric("From", df["reading_date"].min().strftime("%d %b %Y"))
        m[3].metric("To", df["reading_date"].max().strftime("%d %b %Y"))
    elif kind == "master":
        m = st.columns(4)
        m[0].metric("Meters", f"{len(df):,}")
        m[1].metric("Transformers", f"{df['dt_id'].nunique():,}")
        m[2].metric("Sub-divisions", f"{df['subdivision_id'].nunique():,}")
        m[3].metric("Active", f"{(df['status'].str.upper() == 'ACTIVE').sum():,}")
    else:
        m = st.columns(2)
        m[0].metric("SEs", f"{df['se_email'].nunique():,}")
        m[1].metric("Sub-divisions covered", f"{df['subdivision_id'].nunique():,}")


# ---------------------------------------------------------------- step 2
def show_windows(eval_date, history_start, history_end, lead: str = ""):
    """Every day of the available readings, split into the periods the check uses for this date.

    Period bounds come from windows() itself, so the picture can never disagree with the pipeline, and the
    segments always add up to the full history.
    """
    w = windows(eval_date, cfg)
    one = pd.Timedelta(days=1)
    history_start, history_end = pd.Timestamp(history_start), pd.Timestamp(history_end)
    # (css class, title, start, end, note); empty periods are dropped.
    periods = [
        ("tl-unused", "Not used", history_start, w["baseline_start"] - one, "older readings"),
        ("tl-before", "Before", w["baseline_start"], w["baseline_end"], "usual use"),
        ("tl-gap", "Skipped", w["baseline_end"] + one, w["current_start"] - one, "gap"),
        ("tl-now", "Now", w["current_start"], w["current_end"], "latest use"),
        ("tl-unused", "Not used", w["current_end"] + one, history_end, "later readings"),
    ]
    periods = [(c, t, a, b, n, (b - a).days + 1) for c, t, a, b, n in periods if b >= a]
    total = sum(p[-1] for p in periods)
    segs = "".join(f'<div class="{c}" style="flex:{max(d / total, 0.12):.3f} 1 0"><b>{t}</b>'
                   f'{a:%d %b} – {b:%d %b}<br>{n} ({d} days)</div>' for c, t, a, b, n, d in periods)
    sums = " + ".join(f"{d} {t.lower()}" for c, t, a, b, n, d in periods)
    if lead:
        st.markdown(lead)
    st.html(f"""
<style>
  .tl {{ display: flex; width: 100%; border-radius: 8px; overflow: hidden; font-size: .78rem; line-height: 1.3; }}
  .tl div {{ padding: 8px 8px; min-width: 0; }}
  .tl b {{ display: block; font-size: .85rem; }}
  .tl-unused {{ background: rgba(138,137,132,.10); opacity: .75; }}
  .tl-before {{ background: rgba(42,120,214,.15); border-left: 4px solid #2a78d6; }}
  .tl-gap {{ background: repeating-linear-gradient(45deg, rgba(138,137,132,.12) 0 6px, transparent 6px 12px);
             opacity: .85; }}
  .tl-now {{ background: rgba(235,104,52,.15); border-right: 4px solid #eb6834; }}
  @media (max-width: 640px) {{ .tl {{ flex-direction: column; }} }}
</style>
<div class="tl">{segs}</div>""")
    gap_days = (w["current_start"] - w["baseline_end"]).days - 1
    st.caption(f"**{total} days of readings** ({history_start:%d %b} – {history_end:%d %b %Y}) = {sums}. "
               f"Checking as of **{eval_date:%d %b %Y}**, each meter's typical daily use in the "
               f"**now** period is compared with its typical daily use in the **before** period. "
               + (f"The {gap_days} days in between are skipped, so a theft that starts slowly cannot creep into the "
                  "'before' figure and hide itself. " if gap_days > 0 else "")
               + "The period lengths come from config.yaml (baseline_days, baseline_gap_days, current_window_days).")


def step_detect():
    page_heading(1, "Detect suspected thefts")
    if not all_loaded():
        st.warning("Generate the data first.")
        st.button("← Back to data generation", on_click=go, args=(STEPS[0],))
        return
    daily, master = ss.data["daily"], ss.data["master"]
    max_d = daily["reading_date"].max().date()
    min_d = (daily["reading_date"].min() + pd.Timedelta(days=MIN_DAYS - 1)).date()

    st.html('<p class="page-intro">Pick the date to run the check for. It is the <b>last day of the period being '
            "checked</b>, not a single day: the app looks at each meter's last month of use and compares it with its "
            "usual use before that.</p>")
    c1, c2 = st.columns([1, 3])
    with c1:
        eval_date = st.date_input("Run the check as of", value=max_d, min_value=min(min_d, max_d), max_value=max_d,
                                  help="In live use this would simply be today, run once a week.")
        run = st.button("Run detection", type="primary", width="stretch")
    with c2:
        show_windows(eval_date, daily["reading_date"].min(), max_d)

    st.caption(f"Flag rule: a meter is flagged when it dropped more than {cfg.excess_drop_threshold_pct:g} points "
               f"beyond its neighbours and by at least {cfg.min_drop_kwh_per_day:g} kWh a day (explained in Step 1).")

    if run:
        with st.spinner("Scoring meters…"):
            ss.scores = run_pipeline(daily, master, eval_date, cfg)
            ss.eval_date = eval_date
    scores = ss.scores
    if scores is None:
        st.info("Pick an evaluation date and run detection.")
        return

    flagged = scores[scores["status"] == "FLAGGED"].sort_values("est_unbilled_kwh", ascending=False)
    suppressed = scores[scores["status"] == "SUPPRESSED"]

    st.divider()
    st.header("Inspection results")
    st.caption(f"Checked as of {ss.eval_date:%d %b %Y}. Tap a blue tile to see its meters.")
    result_tiles(scores, flagged, suppressed)
    t1, t2 = st.tabs(["💡 Insights", "📋 Report"])
    with t1:
        insights(flagged, suppressed)
    with t2:
        report(scores, flagged)

    st.divider()
    drilldown(flagged, suppressed)

    if ss.truth is not None:
        st.divider()
        truth_check(scores)


FLAGGED_COLS = {
    "meter_id": st.column_config.TextColumn("Meter"),
    "consumer_id": st.column_config.TextColumn("Consumer"),
    "dt_id": st.column_config.TextColumn("Transformer"),
    "drop_pct": st.column_config.NumberColumn("Drop %", format="%.0f%%"),
    "peer_drop_pct": st.column_config.NumberColumn("Neighbours' drop", format="%.0f%%"),
    "onset_date": st.column_config.DateColumn("Drop began", format="DD MMM YYYY"),
    "est_unbilled_kwh": st.column_config.NumberColumn("kWh at risk", format="%.0f"),
}


def result_tiles(scores: pd.DataFrame, flagged: pd.DataFrame, suppressed: pd.DataFrame):
    """Headline numbers; the suspected and suppressed tiles open their list in a popup."""
    st.html("""
<style>
  .st-key-tile_flagged, .st-key-tile_suppressed { border-color: #2a78d6 !important; }
  .st-key-tile_flagged:hover, .st-key-tile_suppressed:hover { background: rgba(42,120,214,.06); }
</style>""")
    c = st.columns(4)
    with c[0].container(border=True):
        st.metric("Meters checked", f"{len(scores):,}")
        st.caption("every meter in the area")
    with c[1].container(border=True, key="tile_flagged"):
        st.metric("Suspected (flagged)", f"{len(flagged):,}")
        if st.button("See the list →", key="open_flagged", type="tertiary"):
            flagged_dialog(flagged)
    with c[2].container(border=True, key="tile_suppressed"):
        st.metric("Suppressed", f"{len(suppressed):,}")
        if st.button("See the list →", key="open_suppressed", type="tertiary"):
            suppressed_dialog(suppressed)
    with c[3].container(border=True):
        st.metric("kWh at risk", f"{flagged['est_unbilled_kwh'].sum():,.0f}")
        st.caption("estimated units possibly unbilled")


@st.dialog("Suspected thefts", width="large")
def flagged_dialog(flagged: pd.DataFrame):
    st.write(f"**{len(flagged)} meters** recommended for inspection, biggest kWh at risk first.")
    st.dataframe(flagged[list(FLAGGED_COLS)], column_config=FLAGGED_COLS, width="stretch", hide_index=True)
    st.caption("Use Meter drill-down on the page to see why any one of them was flagged.")


@st.dialog("Suppressed meters", width="large")
def suppressed_dialog(suppressed: pd.DataFrame):
    st.write(f"**{len(suppressed)} meters** dropped, or could not be judged, but were set aside for a reason "
             "other than theft. No visit is requested for them.")
    counts = suppressed["suppression_code"].value_counts().rename_axis("Reason").reset_index(name="Meters")
    counts["What it means"] = counts["Reason"].map(SUPPRESSION_CODES)
    st.dataframe(counts, width="stretch", hide_index=True)
    st.dataframe(suppressed[["meter_id", "dt_id", "suppression_code", "drop_pct", "comm_availability_pct"]],
                 column_config={"meter_id": "Meter", "dt_id": "Transformer", "suppression_code": "Reason",
                                "drop_pct": st.column_config.NumberColumn("Drop %", format="%.0f%%"),
                                "comm_availability_pct": st.column_config.NumberColumn("Readings arrived",
                                                                                       format="%.0f%%")},
                 width="stretch", hide_index=True)


def insights(flagged: pd.DataFrame, suppressed: pd.DataFrame):
    points = []
    if flagged.empty:
        st.success("No meter dropped far enough beyond its neighbours to need a visit.")
    else:
        total = flagged["est_unbilled_kwh"].sum()
        top = flagged.head(5)
        per_dt = flagged["dt_id"].value_counts()
        hot = per_dt[per_dt >= 2]
        onsets = pd.to_datetime(flagged["onset_date"]).dropna()
        points += [
            f"**{len(flagged)} meters** are recommended for inspection, with about **{total:,.0f} kWh** "
            "possibly unbilled between them.",
            f"**Start with the top {len(top)}**: they hold **{top['est_unbilled_kwh'].sum() / total:.0%}** of the "
            f"kWh at risk ({', '.join(top['meter_id'])}).",
        ]
        if len(hot):
            points.append(f"**Hotspots**: {', '.join(f'{d} ({n} meters)' for d, n in hot.items())} have more than "
                          "one suspect, so one trip to that transformer area can cover several.")
        if len(onsets):
            points.append(f"The drops began between **{onsets.min():%d %b}** and **{onsets.max():%d %b}**.")
    if len(suppressed):
        reasons = suppressed["suppression_code"].value_counts()
        points.append(f"**{len(suppressed)} meters were set aside** for innocent reasons, most often "
                      f"{reasons.index[0]} ({reasons.iloc[0]}): {SUPPRESSION_CODES[reasons.index[0]].lower()}.")
    st.markdown("\n".join(f"- {p}" for p in points))
    visit_summary(flagged)


def report(scores: pd.DataFrame, flagged: pd.DataFrame):
    st.write("The inspection list as it will be emailed in Step 3, biggest kWh at risk first.")
    if flagged.empty:
        st.info("Nothing to report for this date.")
    else:
        st.dataframe(flagged[list(FLAGGED_COLS) + ["reason_text"]],
                     column_config=FLAGGED_COLS | {"reason_text": st.column_config.TextColumn("Reason", width="large")},
                     width="stretch", hide_index=True)
    c = st.columns(2)
    c[0].download_button("Download inspection list (CSV)", flagged.to_csv(index=False), width="stretch",
                         file_name=f"inspection_list_{ss.eval_date}.csv", mime="text/csv", disabled=flagged.empty)
    c[1].download_button("Download all meter scores (CSV)", scores.to_csv(index=False), width="stretch",
                         file_name=f"scores_{ss.eval_date}.csv", mime="text/csv")


def visit_summary(flagged: pd.DataFrame):
    """Who receives the list, and how many visits it asks for."""
    officers = ss.data["officers"]
    per_sd = (flagged.groupby("subdivision_id")
              .agg(meters=("meter_id", "size"), kwh=("est_unbilled_kwh", "sum"))
              .reindex(officers["subdivision_id"], fill_value=0).reset_index()
              .merge(officers, on="subdivision_id", how="left"))
    for r in per_sd.itertuples():
        st.info(f"**{r.meters} inspection visits** for {hierarchy.label(r.subdivision_id)} go to "
                f"**{r.se_name}** in Step 3 ({r.kwh:,.0f} kWh at risk).", icon="📨")
    unmapped = sorted(set(flagged["subdivision_id"].dropna()) - set(officers["subdivision_id"]))
    if unmapped:
        st.warning(f"No SE for sub-division(s) {', '.join(unmapped)}: their meters cannot be sent.")


def drilldown(flagged: pd.DataFrame, suppressed: pd.DataFrame):
    """Same pattern as the Step 1 example: a faded teaser, opened on request."""
    ss.setdefault("show_drill", False)
    st.subheader("🔍 Meter drill-down")
    pool = pd.concat([flagged, suppressed])
    if pool.empty:
        st.caption("No flagged or suppressed meters to look into.")
    elif not ss.show_drill:
        first = pool.iloc[0]
        ask = "Why was" if first["status"] == "FLAGGED" else "Why wasn't"
        teaser("<p>Pick any meter on the list and see the evidence behind the decision: its readings, its "
               "neighbours, and the day its drop began.</p>"
               f"<p><b>{ask} meter {first['meter_id']} flagged?</b> {decision_text(first, html=True)}</p>")
    c = st.columns(2)
    if not pool.empty:
        if ss.show_drill:
            c[0].button("Hide the drill-down ↑", width="stretch", on_click=lambda: ss.update(show_drill=False))
        else:
            c[0].button("Investigate a meter ↓", width="stretch", on_click=lambda: ss.update(show_drill=True))
    if len(flagged):
        c[1].button("Next: send to SE →", type="primary", width="stretch", on_click=go, args=(STEPS[2],))
    if pool.empty or not ss.show_drill:
        return

    options = [f"Suspected ({len(flagged)})", f"Suppressed ({len(suppressed)})"]
    group = st.segmented_control("Show", options, default=options[0] if len(flagged) else options[1],
                                 key="drill_group") or options[0]
    view = flagged if group.startswith("Suspected") else suppressed
    if view.empty:
        st.caption("No meters in this group.")
        return
    if view is flagged:
        labels = {r.meter_id: f"{r.meter_id} · {r.est_unbilled_kwh:,.0f} kWh at risk" for r in view.itertuples()}
    else:
        labels = {r.meter_id: f"{r.meter_id} · {r.suppression_code}" for r in view.itertuples()}
    meter = st.selectbox("Meter", list(labels), format_func=labels.get, key=f"drill_meter_{view is flagged}")
    meter_drilldown(view[view["meter_id"] == meter].iloc[0], ss.eval_date)


def truth_check(scores: pd.DataFrame):
    """Compare the run with the scenarios planted by the generator. Only possible because the data is synthetic."""
    st.subheader("Demo only: how did detection do against the planted scenarios?")
    st.info("**This section exists only because the data is generated.** We planted known problems in some meters, "
            "so we can check the pipeline's answers against the truth. With live data nobody knows the truth in "
            "advance, so a production version would not have this section; field inspection results would be used "
            "to measure accuracy instead.", icon="🧪")
    s = scores.merge(ss.truth[["meter_id", "injected_type"]], on="meter_id", how="left")
    s["injected_type"] = s["injected_type"].fillna("normal")
    theft = s["injected_type"].isin(["partial_bypass", "full_bypass"])
    flagged = s["status"] == "FLAGGED"
    m = st.columns(3)
    m[0].metric("Bypasses caught", f"{(theft & flagged).sum()} of {theft.sum()}")
    m[1].metric("Innocent meters flagged", f"{(~theft & flagged).sum()}")
    m[2].metric("Flags that were real thefts", f"{(theft & flagged).sum() / max(flagged.sum(), 1):.0%}")
    with st.expander("Scenario by scenario"):
        s["outcome"] = np.where(s["status"] == "SUPPRESSED", "Suppressed: " + s["suppression_code"], s["status"])
        kinds = [k for k in SCENARIOS if k in set(s["injected_type"])]
        table = pd.crosstab(s["injected_type"], s["outcome"]).reindex(kinds)
        table.index = [scenario_label(k) for k in table.index]
        table.insert(0, "Expected", [SCENARIOS[k][1] for k in kinds])
        st.dataframe(table, width="stretch")
        st.caption("Bypasses on very small loads can fall under the absolute kWh floor by design. "
                   "Results depend on the check date: a date before the injections shows nothing to catch.")


# ---------------------------------------------------------------- step 3
def default_officers(sd_id: str) -> pd.DataFrame:
    """The sub-division's SE, addressed to the demo inbox, in the same types an uploaded list is parsed to."""
    officers = hierarchy.officer(sd_id)
    return parse(officers.assign(se_email=DEMO_INBOX) if DEMO_INBOX else officers, "officers")[0]


def step_send():
    page_heading(2, "Send suspected cases to SE")
    if ss.scores is None:
        st.warning("Run detection first.")
        st.button("← Back to detection", on_click=go, args=(STEPS[1],))
        return

    officers = officer_directory()
    if officers is None:
        return
    emails, unmapped = build_emails(ss.scores, officers, ss.eval_date)
    if unmapped:
        st.warning(f"Flagged meters in {', '.join(unmapped)} have no SE in the directory and will not be sent.")
    if not emails:
        st.info("No suspected cases to send for this evaluation date.")
        return

    st.subheader("2 · Choose how to deliver")
    modes = [f"Demo: send every list to {shown(DEMO_INBOX) or 'the demo inbox'}", "Production: send each SE their own list"]
    mode = st.radio("Delivery", modes, key="delivery_mode", label_visibility="collapsed")
    demo = mode == modes[0]
    if demo and not DEMO_INBOX:
        st.info("No demo inbox is configured. Set `MAIL_TO` in the server's settings (Secrets on Streamlit Cloud).")
    elif demo:
        st.caption("Each email is still written for its SE, but delivered to the demo inbox with [TEST] in the "
                   "subject, so the real-time email can be shown without contacting anyone.")
    elif not SE_DELIVERY_ALLOWED:
        st.info("Production delivery is switched off on this public demo, because anyone opening the page could "
                "otherwise send email to any address from the project's account. On a private deployment, set "
                "`ALLOW_SE_DELIVERY=true` in the server's settings to enable it.", icon="🔒")

    st.subheader("3 · Review and send")
    st.write(f"**{len(emails)} email(s)** for the check as of {ss.eval_date:%d %b %Y}.")
    st.dataframe(pd.DataFrame([{"SE": e.se_name, "Delivered to": shown(DEMO_INBOX if demo else e.se_email),
                                "Sub-divisions": ", ".join(e.subdivisions), "Suspected cases": e.n_flagged,
                                "kWh at risk": round(e.kwh_at_risk)} for e in emails]),
                 width="stretch", hide_index=True)
    for e in emails:
        with st.expander(f"Preview: {e.se_name} <{shown(e.se_email)}>"):
            st.write(f"**Subject:** {'[TEST] ' if demo else ''}{e.subject}")
            st.html(e.html)

    # Credentials are server-side only (.env locally, Secrets on Streamlit Cloud), never in a widget.
    user = os.getenv("SMTP_USER", "").strip()
    pwd = os.getenv("SMTP_PASSWORD", "")
    if not (user and pwd):
        st.info("Sending is not configured on this server. The preview above is what would be sent.")
        return

    # Safe to show: the sender masked, and only whether the password has an App Password's shape.
    key = "".join(pwd.split())
    shape = ("looks like a Gmail App Password (16 letters)" if len(key) == 16 and key.isalpha()
             else f"does not look like a Gmail App Password (expected 16 letters, found {len(key)} characters)")
    st.caption(f"Sending from {mask(user)} · password {shape}.")
    used = sends_today()
    over = used + len(emails) > DAILY_SEND_LIMIT
    blocked = over or (demo and not DEMO_INBOX) or (not demo and not SE_DELIVERY_ALLOWED)
    label = f"Send to {shown(DEMO_INBOX)}" if demo else f"Send to {len(emails)} SE(s)"
    st.caption(f"{used} of {DAILY_SEND_LIMIT} sends used today.")
    if st.button(label, type="primary", disabled=blocked):
        with st.spinner("Sending…"):
            ss.last_result = send_emails(emails, ss.eval_date, cfg, user, pwd, LOG_PATH,
                                         override_to=DEMO_INBOX if demo else None)
    if over:
        st.warning("Today's sending limit is reached. Try again tomorrow.")

    if ss.get("last_result") is not None:
        res = ss.last_result
        sent = res["status"].isin(["SENT", "TEST_SENT"]).sum()
        (st.success if sent == len(res) else st.warning)(f"{sent} of {len(res)} sent.")
        st.dataframe(masked(res), width="stretch", hide_index=True)
    if LOG_PATH.exists():
        with st.expander("Full delivery log"):
            st.dataframe(masked(pd.read_csv(LOG_PATH, dtype=str)), width="stretch", hide_index=True)


def officer_directory() -> pd.DataFrame | None:
    """Who receives each sub-division's list: edit in place or upload a CSV. Returns the valid list, or None."""
    st.subheader("1 · SE directory")
    st.write("Who receives each sub-division's list. In production this is the utility's officer list: add, change "
             "or remove rows here, or upload it as a CSV with the columns `subdivision_id`, `se_name`, `se_email`.")
    ss.setdefault("officer_ver", 0)
    if "officer_base" not in ss:
        ss.officer_base = default_officers(ss.sd_id)

    edited = st.data_editor(
        masked_officers(ss.officer_base), num_rows="dynamic", width="stretch", hide_index=True, key=f"officers_{ss.officer_ver}",
        column_config={"subdivision_id": st.column_config.TextColumn("Sub-division", required=True),
                       "se_name": st.column_config.TextColumn("SE name", required=True),
                       "se_email": st.column_config.TextColumn("SE email", required=True)})

    c = st.columns([2, 1, 1], vertical_alignment="bottom")
    up = c[0].file_uploader("Upload officer list (CSV)", type="csv", key=f"officer_upload_{ss.officer_ver}")
    template = default_officers(ss.sd_id).assign(se_email="se.name@utility.example")
    c[1].download_button("Template CSV", template.to_csv(index=False), width="stretch",
                         file_name="officers_template.csv", mime="text/csv")
    if c[2].button("Reset to default", width="stretch"):
        replace_directory(default_officers(ss.sd_id))
    if up is not None:
        df, errors = parse(pd.read_csv(up, dtype=str), "officers")
        if errors:
            st.error("The uploaded list was not used: " + "; ".join(errors))
        else:
            replace_directory(df)

    officers, errors = parse(unmask_officers(edited.astype("string").fillna("")), "officers")
    if errors:
        st.error("Fix the directory before sending: " + "; ".join(errors))
        return None
    if not officers.equals(ss.data["officers"]):
        ss.data["officers"] = officers
        ss.pop("last_result", None)
    return officers


def replace_directory(df: pd.DataFrame):
    ss.officer_base = df.reset_index(drop=True)
    ss.officer_ver += 1  # new editor and uploader keys, so neither replays its old state
    ss.pop("last_result", None)
    st.rerun()


def mask(addr: str) -> str:
    name, _, domain = str(addr).partition("@")
    return f"{name[:2]}***@{domain}" if domain else "***"


def shown(addr: str) -> str:
    """The demo inbox is the project owner's own address: it is shown masked everywhere, and only used to send."""
    return mask(addr) if DEMO_INBOX and str(addr).strip().lower() == DEMO_INBOX.lower() else addr


def masked_officers(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(se_email=df["se_email"].map(shown))


def unmask_officers(df: pd.DataFrame) -> pd.DataFrame:
    """The directory editor shows the demo inbox masked; a row left as that mask still means the demo inbox."""
    return df.assign(se_email=df["se_email"].replace(mask(DEMO_INBOX), DEMO_INBOX)) if DEMO_INBOX else df


def masked(log: pd.DataFrame) -> pd.DataFrame:
    return log.assign(se_email=log["se_email"].map(mask), sent_to=log["sent_to"].map(mask))


def sends_today() -> int:
    """Emails attempted today across all visitors, so a public demo cannot flood the inbox."""
    if not LOG_PATH.exists():
        return 0
    log = pd.read_csv(LOG_PATH, dtype=str)
    return int((log["sent_at"].str[:10] == date.today().isoformat()).sum())


app_styles()
st.button("Consumption-drop detection", key="home", type="tertiary", help="Home: start a fresh analysis", on_click=start_over)
if ss.step == INTRO:
    intro_page()
else:
    # The progress track and footer are drawn after the page so they reflect what this run just did.
    ribbon = st.container(key="stepper_top")
    {STEPS[0]: step_generate, STEPS[1]: step_detect, STEPS[2]: step_send}[ss.step]()
    with ribbon:
        stepper()
    with st.container(key="wizard_footer"):
        wizard_footer()
    st.button("About this project", key="about", type="tertiary", on_click=go, args=(INTRO,))
st.caption("Every output says *suspected* / *recommended for inspection*. "
           "A finding of theft is made by a field officer, not by this tool.")
scroll_to_top()
