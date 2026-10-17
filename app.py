"""Consumption-drop theft detection MVP: generate data -> detect -> send to SE.

Run with:  streamlit run app.py
"""
import os
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
STEPS = ["1. Generate data", "2. Detect suspected thefts", "3. Send to SE"]

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Consumption-drop detection", page_icon="⚡", layout="wide")
cfg = load_config(ROOT / "config.yaml")
ss = st.session_state
ss.setdefault("data", {})       # kind -> parsed, valid dataframe
ss.setdefault("step", STEPS[0])
ss.setdefault("scores", None)
ss.setdefault("truth", None)  # ground_truth of the generated population
ss.setdefault("eval_date", None)
SEED = 7
MIN_DAYS = cfg.baseline_days + cfg.baseline_gap_days + cfg.current_window_days


def go(step: str):
    ss.step = step


def all_loaded() -> bool:
    return all(k in ss.data for k in SCHEMAS)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚡ Consumption-drop detection")
    st.caption("Weekly MVP · peer-normalised drop analysis")
    st.radio("Step", STEPS, key="step", label_visibility="collapsed")
    st.divider()
    for kind, spec in SCHEMAS.items():
        st.write(("✅ " if kind in ss.data else "⬜ ") + spec["label"])
    st.write(("✅ " if ss.scores is not None else "⬜ ") + "Detection run")
    st.divider()
    st.caption("Every output says *suspected* / *recommended for inspection*. "
               "A finding of theft is made by a field officer, not by this tool.")


# ---------------------------------------------------------------- step 1
@st.cache_data(show_spinner=False)
def generate(days: int, end: str, subdivision: str) -> dict[str, pd.DataFrame]:
    # Fixed seed: the same area and settings always give the same data. Real data would come from the MDM database.
    return generate_population(days, SEED, end, subdivision)


def scenario_label(kind: str) -> str:
    return {"der_added": "Rooftop solar (DER) added"}.get(kind, kind.replace("_", " ").capitalize())


def step_generate():
    st.header("Step 1 · Generate data")
    st.write("This is a mock project, so instead of asking you for real utility data, the app builds a "
             "synthetic population of smart meters. A handful of meters have known problems planted in them, "
             "so after detection you can see exactly which ones the pipeline caught and which it correctly ignored.")

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
  .net-flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 4px 0 8px; }}
  .net-box {{ flex: 1 1 140px; min-width: 0; padding: 14px 16px; border-radius: 12px;
             border: 1px solid rgba(42,120,214,.35); background: rgba(42,120,214,.08); }}
  .net-num {{ font-size: 2rem; font-weight: 700; line-height: 1.1; color: #2a78d6; }}
  .net-lbl {{ font-size: .95rem; font-weight: 600; }}
  .net-sub {{ font-size: .8rem; opacity: .75; }}
  .net-arrow {{ font-size: 1.4rem; opacity: .5; }}
  @media (max-width: 640px) {{ .net-flow {{ flex-direction: column; align-items: stretch; }}
                               .net-arrow {{ transform: rotate(90deg); text-align: center; }} }}
</style>
<div class="net-flow">
  <div class="net-box"><div class="net-num">{sd['meters']:,}</div><div class="net-lbl">smart meters</div>
    <div class="net-sub">one in every home or shop</div></div>
  <div class="net-arrow">→</div>
  <div class="net-box"><div class="net-num">{n_feeders}</div><div class="net-lbl">feeders</div>
    <div class="net-sub">11 kV lines that carry power into the area</div></div>
  <div class="net-arrow">→</div>
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
                 "cases. In this demo, Step 3 delivers the email to the project owner's inbox instead.")
        show_summary("officers", ss.data["officers"])
        st.dataframe(ss.data["officers"], width="stretch", hide_index=True)


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


def show_example():
    """Faded teaser by default, so moving on is one obvious click; the full walk-through is opt-in."""
    ss.setdefault("show_example", False)
    st.subheader("See an example meter")
    if not ss.show_example:
        kind = example_kinds()[0]
        row, res = example_for(kind)
        st.html(f"""
<style>
  .teaser {{ -webkit-mask-image: linear-gradient(to bottom, #000 35%, transparent 100%);
            mask-image: linear-gradient(to bottom, #000 35%, transparent 100%); max-height: 170px; overflow: hidden; }}
  .teaser p {{ margin: 0 0 10px; line-height: 1.55; }}
</style>
<div class="teaser">
  <p>Follow one meter from the data, step by step: who it belongs to, what it used to use, what it uses now,
    and <b>exactly where on its chart the pipeline flagged it</b>, or why it decided to leave it alone.</p>
  <p><b>{scenario_label(kind)}, meter {row['meter_id']}:</b> {decision_text(res, html=True)}</p>
</div>""")
        c = st.columns(2)
        c[0].button("Read the full example ↓", width="stretch", on_click=lambda: ss.update(show_example=True))
        c[1].button("Next: detect suspected thefts →", type="primary", width="stretch", on_click=go, args=(STEPS[1],))
        return

    kind = st.selectbox("Pick a scenario", example_kinds(), format_func=scenario_label)
    row, res = example_for(kind)
    meter_id = row["meter_id"]
    daily, master = ss.data["daily"], ss.data["master"]
    eval_date = daily["reading_date"].max().date()
    st.caption(f"Detection result as of the last reading date, {eval_date:%d %b %Y}. "
               "Step 2 lets you run it for another date.")

    show_meter_card(master[master["meter_id"] == meter_id].iloc[0])
    show_verdict(res)
    show_example_chart(kind, row, res, eval_date)

    with st.expander(f"Daily readings for meter {meter_id}"):
        st.dataframe(daily[daily["meter_id"] == meter_id].sort_values("reading_date", ascending=False)
                     .rename(columns={"reading_date": "Date", "kwh_import": "kWh used",
                                      "intervals_expected": "Readings expected", "intervals_received": "Readings arrived",
                                      "voltage_present_pct": "Supply on (%)"}).drop(columns="meter_id"),
                     width="stretch", hide_index=True, height=260)
    c = st.columns(2)
    c[0].button("Hide the example ↑", width="stretch", on_click=lambda: ss.update(show_example=False))
    c[1].button("Next: detect suspected thefts →", type="primary", width="stretch", on_click=go, args=(STEPS[1],))


def show_example_chart(kind: str, row: pd.Series, res: pd.Series, eval_date):
    daily, master = ss.data["daily"], ss.data["master"]
    meter_id = row["meter_id"]
    w = windows(eval_date, cfg)
    normal_ids = set(master["meter_id"]) - set(ss.truth["meter_id"])
    typical = (daily[daily["meter_id"].isin(normal_ids)].groupby("reading_date")["kwh_import"].median()
               .reset_index().assign(series="Typical normal meter"))
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
                          scale=alt.Scale(range=["#1a1a1a", "#8a8984"])),
        tooltip=[alt.Tooltip("reading_date:T", format="%d %b %Y"), alt.Tooltip("kwh_import:Q", format=".2f")])
    # Legend instead of on-chart text: the two dates are often days apart and the labels would overlap.
    marks = [{"d": pd.Timestamp(row["onset_date"]), "event": "Problem injected"}]
    if res["status"] == "FLAGGED" and pd.notna(res["onset_date"]):
        marks.append({"d": pd.Timestamp(res["onset_date"]), "event": "⚑ Flagged: drop starts here"})
    rules = alt.Chart(pd.DataFrame(marks)).mark_rule(strokeDash=[4, 3], strokeWidth=2.5).encode(
        x="d:T", tooltip=[alt.Tooltip("event:N", title="Event"), alt.Tooltip("d:T", title="Date", format="%d %b %Y")],
        color=alt.Color("event:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["Problem injected", "⚑ Flagged: drop starts here"],
                                        range=["#8a8984", "#d62728"])))
    layers = [bands, lines, rules]
    if pd.notna(res["baseline_daily_kwh"]):
        levels = [{"start": w["baseline_start"], "end": w["baseline_end"], "y": res["baseline_daily_kwh"]}]
        if pd.notna(res["peer_drop_pct"]):
            levels.append({"start": w["current_start"], "end": w["current_end"],
                           "y": res["baseline_daily_kwh"] * (1 - res["peer_drop_pct"] / 100)})
        layers.append(alt.Chart(pd.DataFrame(levels)).mark_rule(strokeWidth=3, color="#2a78d6").encode(
            x="start:T", x2="end:T", y="y:Q"))
    chart = alt.layer(*layers).resolve_scale(color="independent", fill="independent", stroke="independent")
    st.altair_chart(chart.properties(height=300), width="stretch")
    st.caption(f"{SCENARIOS[kind][0]} Thick blue lines: what the meter used to use, and what it would use now "
               f"if it had dropped only as much as its neighbours. Expected outcome: {SCENARIOS[kind][1]}.")


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
    c[0].metric("Used to use", num(r["baseline_daily_kwh"], ".1f") + " kWh/day")
    c[1].metric("Uses now", num(r["current_daily_kwh"], ".1f") + " kWh/day",
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
    st.header("Step 2 · Detect suspected thefts")
    if not all_loaded():
        st.warning("Generate the data first.")
        st.button("← Back to data generation", on_click=go, args=(STEPS[0],))
        return
    daily, master = ss.data["daily"], ss.data["master"]
    max_d = daily["reading_date"].max().date()
    min_d = (daily["reading_date"].min() + pd.Timedelta(days=MIN_DAYS - 1)).date()

    st.write("Pick the date to run the check for. It is the **last day of the period being checked**, not a single "
             "day: the app looks at each meter's last month of use and compares it with its usual use before that.")
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

    flagged = scores[scores["status"] == "FLAGGED"]
    suppressed = scores[scores["status"] == "SUPPRESSED"]
    m = st.columns(4)
    m[0].metric("Meters evaluated", f"{len(scores):,}")
    m[1].metric("Suspected (flagged)", f"{len(flagged):,}")
    m[2].metric("Suppressed", f"{len(suppressed):,}")
    m[3].metric("Est. kWh at risk", f"{flagged['est_unbilled_kwh'].sum():,.0f}")
    st.caption(f"Results for evaluation date {ss.eval_date:%d %b %Y}.")
    visit_summary(flagged)

    t1, t2, t3 = st.tabs([f"Suspected thefts ({len(flagged)})", f"Suppressed ({len(suppressed)})", "All meters"])
    with t1:
        sds = sorted(flagged["subdivision_id"].dropna().unique())
        pick = st.multiselect("Sub-division", sds, default=sds)
        view = flagged[flagged["subdivision_id"].isin(pick)]
        st.dataframe(
            view[["meter_id", "consumer_id", "subdivision_id", "dt_id", "baseline_daily_kwh", "current_daily_kwh",
                  "drop_pct", "peer_drop_pct", "excess_drop_pct", "onset_date", "est_unbilled_kwh",
                  "peer_group_size", "reason_text"]],
            width="stretch", hide_index=True,
            column_config={
                "baseline_daily_kwh": st.column_config.NumberColumn("Baseline kWh/day", format="%.1f"),
                "current_daily_kwh": st.column_config.NumberColumn("Current kWh/day", format="%.1f"),
                "drop_pct": st.column_config.NumberColumn("Drop %", format="%.0f"),
                "peer_drop_pct": st.column_config.NumberColumn("Peer drop %", format="%.0f"),
                "excess_drop_pct": st.column_config.NumberColumn("Beyond peers (pts)", format="%.0f"),
                "est_unbilled_kwh": st.column_config.NumberColumn("Est. unbilled kWh", format="%.0f"),
                "reason_text": st.column_config.TextColumn("Reason", width="large"),
            },
        )
        if len(view):
            meter_chart(view)
    with t2:
        counts = (suppressed["suppression_code"].value_counts().rename_axis("code").reset_index(name="meters"))
        counts["meaning"] = counts["code"].map(SUPPRESSION_CODES)
        st.dataframe(counts, width="stretch", hide_index=True)
        st.dataframe(suppressed[["meter_id", "subdivision_id", "suppression_code", "drop_pct",
                                 "comm_availability_pct", "reason_text"]],
                     width="stretch", hide_index=True)
    with t3:
        st.dataframe(scores, width="stretch", hide_index=True)
    if ss.truth is not None:
        truth_check(scores)

    st.download_button("Download all scores (CSV)", scores.to_csv(index=False),
                       file_name=f"scores_{ss.eval_date}.csv", mime="text/csv")
    if len(flagged):
        st.button("Next: send to SE →", type="primary", on_click=go, args=(STEPS[2],))


def visit_summary(flagged: pd.DataFrame):
    """How many inspection visits each sub-division's SE will be asked for."""
    st.subheader(f"Inspection visits to request: {len(flagged)} meters")
    officers = ss.data["officers"]
    per_sd = (flagged.groupby("subdivision_id")
              .agg(meters=("meter_id", "size"), kwh=("est_unbilled_kwh", "sum"))
              .reindex(officers["subdivision_id"], fill_value=0).reset_index()
              .merge(officers, on="subdivision_id", how="left"))
    cols = st.columns(len(per_sd))
    for col, r in zip(cols, per_sd.itertuples()):
        with col.container(border=True):
            st.metric(f"Sub-division {r.subdivision_id}", f"{r.meters} visits")
            st.caption(f"Goes to **{r.se_name}** ({r.se_email}) · {r.kwh:,.0f} kWh at risk")
    unmapped = sorted(set(flagged["subdivision_id"].dropna()) - set(officers["subdivision_id"]))
    if unmapped:
        st.warning(f"No SE for sub-division(s) {', '.join(unmapped)}: their meters cannot be sent.")


def truth_check(scores: pd.DataFrame):
    """Compare the run with the scenarios planted by the generator. Only possible because the data is synthetic."""
    with st.expander("How did detection do against the planted scenarios?", expanded=True):
        s = scores.merge(ss.truth[["meter_id", "injected_type"]], on="meter_id", how="left")
        s["injected_type"] = s["injected_type"].fillna("normal")
        s["outcome"] = np.where(s["status"] == "SUPPRESSED", "Suppressed: " + s["suppression_code"], s["status"])
        table = pd.crosstab(s["injected_type"], s["outcome"]).reindex([k for k in SCENARIOS if k in set(s["injected_type"])])
        table.index = [scenario_label(k) for k in table.index]
        table.insert(0, "Expected", [SCENARIOS[k][1] for k in SCENARIOS if k in set(s["injected_type"])])
        st.dataframe(table, width="stretch")
        theft = s["injected_type"].isin(["partial_bypass", "full_bypass"])
        flagged = s["status"] == "FLAGGED"
        m = st.columns(3)
        m[0].metric("Bypasses caught", f"{(theft & flagged).sum()} of {theft.sum()}")
        m[1].metric("Innocent meters flagged", f"{(~theft & flagged).sum()}")
        m[2].metric("Precision of the list", f"{(theft & flagged).sum() / max(flagged.sum(), 1):.0%}")
        st.caption("Bypasses on very small loads can fall under the absolute kWh floor by design. "
                   "Results depend on the evaluation date: a date before the injections shows nothing to catch.")


def meter_chart(view: pd.DataFrame):
    st.subheader("Meter drill-down")
    meter = st.selectbox("Meter", view["meter_id"],
                         format_func=lambda m: f"{m} · {view.set_index('meter_id').at[m, 'est_unbilled_kwh']:,.0f} kWh at risk")
    r = view.set_index("meter_id").loc[meter]
    w = windows(ss.eval_date, cfg)
    d = ss.data["daily"]
    d = d[(d["meter_id"] == meter) & d["reading_date"].between(w["baseline_start"], w["current_end"])]
    series = pd.DataFrame({"date": d["reading_date"], "kWh": d["kwh_import"], "series": "Daily kWh"})
    refs = pd.DataFrame([
        {"date": w["baseline_start"], "kWh": r["baseline_daily_kwh"], "series": "Baseline (median)"},
        {"date": w["baseline_end"], "kWh": r["baseline_daily_kwh"], "series": "Baseline (median)"},
        {"date": w["current_start"], "kWh": r["baseline_daily_kwh"] * (1 - r["peer_drop_pct"] / 100), "series": "Expected from peers"},
        {"date": w["current_end"], "kWh": r["baseline_daily_kwh"] * (1 - r["peer_drop_pct"] / 100), "series": "Expected from peers"},
    ])
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["Daily kWh", "Baseline (median)", "Expected from peers"],
                                      range=["#2a78d6", "#8a8984", "#eb6834"]))
    line = alt.Chart(series).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("kWh:Q", title="kWh per day"), color=color)
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    pts = alt.Chart(series).mark_point(size=60, filled=True).encode(
        x="date:T", y="kWh:Q", color=color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip("kWh:Q", format=".2f")],
    ).add_params(hover)
    ref = alt.Chart(refs).mark_line(strokeWidth=2, strokeDash=[6, 4]).encode(
        x="date:T", y="kWh:Q", color=color, detail="series:N")
    onset = alt.Chart(pd.DataFrame({"date": [pd.Timestamp(r["onset_date"])]})).mark_rule(
        strokeDash=[2, 2], color="#8a8984").encode(x="date:T")
    st.altair_chart((line + ref + onset + pts).properties(height=300), width="stretch")
    st.caption(r["reason_text"] + "  Dotted vertical line marks the estimated onset.")


# ---------------------------------------------------------------- step 3
def step_send():
    st.header("Step 3 · Send suspected cases to SE")
    if ss.scores is None:
        st.warning("Run detection first.")
        st.button("← Back to detection", on_click=go, args=(STEPS[1],))
        return

    st.subheader("Recipient")
    se = ss.data["officers"].iloc[0]
    st.caption(f"The list for {hierarchy.label(ss.sd_id)} is addressed to {se['se_name']}. This is a demo, so every "
               "email is delivered to the project owner's inbox, never to the SE address.")

    emails, unmapped = build_emails(ss.scores, ss.data["officers"], ss.eval_date)
    if unmapped:
        st.warning(f"Flagged meters in {', '.join(unmapped)} have no SE in the directory and will not be sent.")
    if not emails:
        st.info("No suspected cases to send for this evaluation date.")
        return

    st.write(f"**{len(emails)} email(s)** ready for evaluation date {ss.eval_date:%d %b %Y}.")
    st.dataframe(pd.DataFrame([{"SE": e.se_name, "Email": e.se_email, "Sub-divisions": ", ".join(e.subdivisions),
                                "Suspected cases": e.n_flagged, "kWh at risk": round(e.kwh_at_risk)} for e in emails]),
                 width="stretch", hide_index=True)
    for e in emails:
        with st.expander(f"Preview: {e.se_name} <{e.se_email}>"):
            st.write(f"**Subject:** {e.subject}")
            st.html(e.html)

    # Credentials and recipient are server-side only (.env locally, st.secrets on Streamlit Cloud).
    # They are never put in a widget: widget values are sent to the visitor's browser.
    user = os.getenv("SMTP_USER", "").strip()
    pwd = os.getenv("SMTP_PASSWORD", "")
    mail_to = os.getenv("MAIL_TO", "").strip() or user
    st.subheader("Send")
    if not (user and pwd):
        st.info("Sending is not configured on this server. The preview above is what would be sent.")
        return

    used = sends_today()
    st.caption(f"Delivered to {mask(mail_to)} · {used} of {DAILY_SEND_LIMIT} sends used today.")
    if st.button("Send to my inbox", type="primary", disabled=used + len(emails) > DAILY_SEND_LIMIT):
        with st.spinner("Sending…"):
            ss.last_result = send_emails(emails, ss.eval_date, cfg, user, pwd, LOG_PATH, override_to=mail_to)
    if used + len(emails) > DAILY_SEND_LIMIT:
        st.warning("Today's sending limit is reached. Try again tomorrow.")

    if ss.get("last_result") is not None:
        res = ss.last_result
        sent = res["status"].isin(["SENT", "TEST_SENT"]).sum()
        (st.success if sent == len(res) else st.warning)(f"{sent} of {len(res)} sent.")
        st.dataframe(masked(res), width="stretch", hide_index=True)
    if LOG_PATH.exists():
        with st.expander("Full delivery log"):
            st.dataframe(masked(pd.read_csv(LOG_PATH, dtype=str)), width="stretch", hide_index=True)


def mask(addr: str) -> str:
    name, _, domain = str(addr).partition("@")
    return f"{name[:2]}***@{domain}" if domain else "***"


def masked(log: pd.DataFrame) -> pd.DataFrame:
    return log.assign(se_email=log["se_email"].map(mask), sent_to=log["sent_to"].map(mask))


def sends_today() -> int:
    """Emails attempted today across all visitors, so a public demo cannot flood the inbox."""
    if not LOG_PATH.exists():
        return 0
    log = pd.read_csv(LOG_PATH, dtype=str)
    return int((log["sent_at"].str[:10] == date.today().isoformat()).sum())


{STEPS[0]: step_generate, STEPS[1]: step_detect, STEPS[2]: step_send}[ss.step]()
