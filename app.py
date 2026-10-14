"""Consumption-drop theft detection MVP: upload -> detect -> send to SE.

Run with:  streamlit run app.py
"""
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.baseline import windows
from src.config import load_config
from src.deliver import build_emails, send_emails
from src.pipeline import run_pipeline
from src.schemas import SCHEMAS, parse, template
from src.suppress import SUPPRESSION_CODES

ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "sample_data"
LOG_PATH = ROOT / "delivery_log.csv"
STEPS = ["1. Upload data", "2. Detect suspected thefts", "3. Send to SE"]

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Consumption-drop detection", page_icon="⚡", layout="wide")
cfg = load_config(ROOT / "config.yaml")
ss = st.session_state
ss.setdefault("data", {})       # kind -> parsed, valid dataframe
ss.setdefault("step", STEPS[0])
ss.setdefault("scores", None)
ss.setdefault("eval_date", None)


def go(step: str):
    ss.step = step


def all_loaded() -> bool:
    return all(k in ss.data for k in SCHEMAS)


def load_csv(source, kind: str) -> list[str]:
    raw = pd.read_csv(source, dtype=str, keep_default_na=False, na_values=[""])
    df, errors = parse(raw, kind)
    if errors:
        ss.data.pop(kind, None)
    else:
        ss.data[kind] = df
    ss.scores = None  # new data invalidates any previous run
    return errors


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
def step_upload():
    st.header("Step 1 · Upload data")
    st.write("Upload the three CSV files in the formats below. Each file is checked against its "
             "data contract before it can be used.")

    with st.expander("No data yet? Load the synthetic sample population"):
        st.write("500 meters, 180 days, with injected bypasses, vacancies, comms failures, DER, "
                 "meter replacements and a seasonal swing on every meter.")
        if st.button("Load sample data"):
            if not (SAMPLE_DIR / "daily_consumption.csv").exists():
                from sample_data.generate import generate_population
                for name, df in generate_population().items():
                    df.to_csv(SAMPLE_DIR / f"{name}.csv", index=False)
            for kind, spec in SCHEMAS.items():
                load_csv(SAMPLE_DIR / spec["filename"], kind)
            st.rerun()

    for kind, spec in SCHEMAS.items():
        st.subheader(spec["label"])
        c1, c2 = st.columns([3, 1])
        with c2:
            st.caption("Required columns")
            st.code(", ".join(spec["columns"]), language=None, wrap_lines=True)
            st.download_button("Download blank template", template(kind).to_csv(index=False),
                               file_name=spec["filename"], mime="text/csv", key=f"tpl_{kind}")
        with c1:
            up = st.file_uploader(spec["filename"], type="csv", key=f"up_{kind}")
            if up is not None and ss.get(f"last_{kind}") != (up.name, up.size):
                ss[f"last_{kind}"] = (up.name, up.size)
                ss[f"err_{kind}"] = load_csv(up, kind)
            for e in ss.get(f"err_{kind}", []):
                st.error(e)
            if kind in ss.data:
                df = ss.data[kind]
                show_summary(kind, df)
                st.dataframe(df.head(500), width="stretch", hide_index=True, height=220)
        st.divider()

    if all_loaded():
        cross_checks()
        st.button("Next: detect suspected thefts →", type="primary", on_click=go, args=(STEPS[1],))
    else:
        st.info("Upload all three files to continue.")


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


def cross_checks():
    daily, master, officers = ss.data["daily"], ss.data["master"], ss.data["officers"]
    no_master = set(daily["meter_id"]) - set(master["meter_id"])
    no_se = set(master["subdivision_id"]) - set(officers["subdivision_id"])
    span = (daily["reading_date"].max() - daily["reading_date"].min()).days + 1
    need = cfg.baseline_days + cfg.baseline_gap_days
    if no_master:
        st.warning(f"{len(no_master)} meters in daily consumption are missing from the meter master; "
                   "they will be reported as NO_MASTER_RECORD.")
    if no_se:
        st.warning(f"No SE mapped for sub-division(s): {', '.join(sorted(no_se))}. "
                   "Their suspected cases cannot be emailed.")
    if span < need:
        st.error(f"Daily data covers {span} days; at least {need} are needed "
                 f"({cfg.baseline_days}-day baseline + {cfg.baseline_gap_days}-day gap).")
    else:
        st.success("All three files are valid.")


# ---------------------------------------------------------------- step 2
def step_detect():
    st.header("Step 2 · Detect suspected thefts")
    if not all_loaded():
        st.warning("Upload all three files first.")
        st.button("← Back to upload", on_click=go, args=(STEPS[0],))
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

    st.download_button("Download all scores (CSV)", scores.to_csv(index=False),
                       file_name=f"scores_{ss.eval_date}.csv", mime="text/csv")
    if len(flagged):
        st.button("Next: send to SE →", type="primary", on_click=go, args=(STEPS[2],))


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


{STEPS[0]: step_upload, STEPS[1]: step_detect, STEPS[2]: step_send}[ss.step]()
