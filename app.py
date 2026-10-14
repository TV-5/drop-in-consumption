"""Consumption-drop theft detection MVP: generate data -> detect -> send to SE.

Run with:  streamlit run app.py
"""
import os
import re
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
    st.write(f"Data is generated for all **{hierarchy.SUBDIVISIONS[sd_id]['meters']:,} smart meters** installed in "
             f"{hierarchy.SUBDIVISIONS[sd_id]['name']}. Choose how much history to create.")
    with st.form("gen"):
        c = st.columns(2)
        days = c[0].number_input("Days of history", MIN_DAYS, 365, 180, step=10,
                                 help=f"At least {MIN_DAYS}: {cfg.baseline_days}-day baseline + "
                                      f"{cfg.baseline_gap_days}-day gap + {cfg.current_window_days}-day current window.")
        end = c[1].date_input("Last reading date", value=pd.Timestamp("2026-10-04"),
                              help="The most recent day of readings. Detection is normally run for this date.")
        submitted = st.form_submit_button(f"Generate data for {hierarchy.SUBDIVISIONS[sd_id]['name']}", type="primary")

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
    st.button("Next: detect suspected thefts →", type="primary", on_click=go, args=(STEPS[1],))


def show_hierarchy(sd_id: str):
    sd = hierarchy.SUBDIVISIONS[sd_id]
    div = hierarchy.DIVISIONS[sd["division"]]
    net = hierarchy.network(sd_id)
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Office hierarchy**")
        st.metric("Installed smart meters", f"{sd['meters']:,}")
        st.markdown("\n".join([
            f"- **Circle** · {hierarchy.CIRCLE['name']} ({hierarchy.CIRCLE['id']}) · {hierarchy.CIRCLE['se_name']}",
            f"    - **Division** · {div['name']} ({sd['division']}) · {div['ee_name']}",
            f"        - **Sub-division** · {sd['name']} ({sd_id}) · {sd['ae_name']}",
        ]))
        st.caption("The inspection list for this sub-division is emailed to the circle SE in Step 3.")
    with c2.container(border=True):
        st.markdown("**Network hierarchy**")
        lines = []
        for ss_id, feeders in net.groupby("substation_id", sort=False):
            lines.append(f"- **{ss_id}** (33/11 kV substation): {feeders['meters'].sum():,} meters")
            for feeder, dts in feeders.groupby("feeder_id", sort=False):
                lines.append(f"    - **{feeder}** (11 kV feeder): {dts['meters'].sum():,} meters on {len(dts)} DTs, "
                             f"{dts['dt_id'].iloc[0]} to {dts['dt_id'].iloc[-1]}")
        st.markdown("\n".join(lines))
        st.caption(f"{net['substation_id'].nunique()} substations · {net['feeder_id'].nunique()} feeders · "
                   f"{len(net)} distribution transformers · {net['meters'].sum():,} meters")
        with st.expander("Meters per transformer"):
            st.dataframe(net.rename(columns={"dt_id": "DT", "feeder_id": "Feeder", "substation_id": "Substation",
                                             "meters": "Meters"}), width="stretch", hide_index=True)


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
                 "cases. You can change the address in Step 3 before sending.")
        show_summary("officers", ss.data["officers"])
        st.dataframe(ss.data["officers"], width="stretch", hide_index=True)


def explain_scenarios():
    st.subheader("Scenarios the data covers")
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

    kinds = [k for k in SCENARIOS if k != "normal" and k in counts]
    kind = st.selectbox("See an example meter", kinds, format_func=scenario_label)
    row = truth[truth["injected_type"] == kind].iloc[0]
    daily = ss.data["daily"]
    normal_ids = set(ss.data["master"]["meter_id"]) - set(truth["meter_id"])
    typical = (daily[daily["meter_id"].isin(normal_ids)].groupby("reading_date")["kwh_import"].median()
               .reset_index().assign(series="Typical normal meter (median)"))
    example = daily.loc[daily["meter_id"] == row["meter_id"], ["reading_date", "kwh_import"]].assign(
        series=f"Meter {row['meter_id']}")
    chart = alt.Chart(pd.concat([typical, example])).mark_line(strokeWidth=2).encode(
        x=alt.X("reading_date:T", title=None), y=alt.Y("kwh_import:Q", title="kWh per day"),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(range=["#2a78d6", "#8a8984"])),
        tooltip=[alt.Tooltip("reading_date:T", format="%d %b %Y"), alt.Tooltip("kwh_import:Q", format=".2f")])
    onset = alt.Chart(pd.DataFrame({"d": [pd.Timestamp(row["onset_date"])]})).mark_rule(
        strokeDash=[2, 2], color="#eb6834").encode(x="d:T")
    st.altair_chart((chart + onset).properties(height=260), width="stretch")
    st.caption(f"{SCENARIOS[kind][0]} The dotted line marks when it was injected. Expected outcome: {SCENARIOS[kind][1]}.")


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
def step_detect():
    st.header("Step 2 · Detect suspected thefts")
    if not all_loaded():
        st.warning("Generate the data first.")
        st.button("← Back to data generation", on_click=go, args=(STEPS[0],))
        return
    daily, master = ss.data["daily"], ss.data["master"]
    max_d = daily["reading_date"].max().date()
    min_d = (daily["reading_date"].min() + pd.Timedelta(days=cfg.baseline_days + cfg.baseline_gap_days)).date()

    c1, c2 = st.columns([1, 3])
    with c1:
        eval_date = st.date_input("Evaluation date", value=max_d, min_value=min(min_d, max_d), max_value=max_d)
        run = st.button("Run detection", type="primary")
    with c2:
        w = windows(eval_date, cfg)
        st.caption(f"Baseline window: {w['baseline_start']:%d %b} – {w['baseline_end']:%d %b %Y} "
                   f"(median of {cfg.baseline_days} days)  ·  Current window: {w['current_start']:%d %b} – "
                   f"{w['current_end']:%d %b %Y}")
        with st.expander("Thresholds (edit config.yaml to change)"):
            st.json({k: v for k, v in vars(cfg).items() if k != "smtp"})

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
    officers = ss.data["officers"]
    se = officers.iloc[0]
    st.caption(f"The list for {hierarchy.label(ss.sd_id)} goes to {se['se_name']}. "
               "Put your own address here if you want to receive it.")
    addr = st.text_input("SE email", value=se["se_email"]).strip()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", addr):
        st.error("Enter a valid email address.")
        return
    if addr != se["se_email"]:
        ss.data["officers"] = officers.assign(se_email=addr)
        ss.pop("last_result", None)

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

    st.subheader("SMTP")
    user = os.getenv("SMTP_USER", "")
    pwd = os.getenv("SMTP_PASSWORD", "")
    st.caption(f"Server {cfg.smtp.host}:{cfg.smtp.port} (config.yaml). Credentials from .env "
               "(SMTP_USER, SMTP_PASSWORD); for Gmail use an App Password.")
    c1, c2 = st.columns(2)
    user = c1.text_input("SMTP user (sender address)", value=user)
    pwd = c2.text_input("SMTP password", value=pwd, type="password")
    if not (user and pwd):
        st.info("Enter SMTP credentials (or set them in .env) to send.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Test send**: every SE's email goes to one address you choose.")
        test_to = st.text_input("Test recipient", value=user)
        if st.button("Send test"):
            with st.spinner("Sending test emails…"):
                ss.last_result = send_emails(emails, ss.eval_date, cfg, user, pwd, LOG_PATH, override_to=test_to)
    with c2:
        st.markdown("**Send to SEs**: each SE gets their own list. Already-sent SEs are skipped.")
        ok = st.checkbox("I have reviewed the list and want to send it to the SEs")
        if st.button("Send to all SEs", type="primary", disabled=not ok):
            with st.spinner("Sending…"):
                ss.last_result = send_emails(emails, ss.eval_date, cfg, user, pwd, LOG_PATH)

    if ss.get("last_result") is not None:
        res = ss.last_result
        sent = res["status"].isin(["SENT", "TEST_SENT"]).sum()
        (st.success if sent == len(res) else st.warning)(f"{sent} of {len(res)} sent.")
        st.dataframe(res, width="stretch", hide_index=True)
    if LOG_PATH.exists():
        with st.expander("Full delivery log"):
            st.dataframe(pd.read_csv(LOG_PATH, dtype=str), width="stretch", hide_index=True)


{STEPS[0]: step_generate, STEPS[1]: step_detect, STEPS[2]: step_send}[ss.step]()
