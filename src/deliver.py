"""Build one email per SE from the flagged meters, and send them over SMTP."""
import smtplib
import ssl
from dataclasses import dataclass
from datetime import date, datetime
from email.message import EmailMessage
from html import escape
from pathlib import Path

import pandas as pd

from .config import Config

EMAIL_COLUMNS = {
    "meter_id": "Meter",
    "consumer_id": "Consumer",
    "subdivision_id": "Sub-division",
    "dt_id": "Transformer",
    "drop_pct": "Drop %",
    "excess_drop_pct": "Beyond peers (pts)",
    "onset_date": "Since",
    "est_unbilled_kwh": "Est. unbilled kWh",
    "reason_text": "Reason",
}

LOG_COLUMNS = ["eval_date", "se_name", "se_email", "sent_to", "n_flagged", "status", "error", "sent_at"]


@dataclass
class Email:
    se_name: str
    se_email: str
    subdivisions: list[str]
    subject: str
    html: str
    table: pd.DataFrame
    n_flagged: int
    kwh_at_risk: float


def _table(flagged: pd.DataFrame) -> pd.DataFrame:
    t = flagged[list(EMAIL_COLUMNS)].rename(columns=EMAIL_COLUMNS).copy()
    t["Drop %"] = t["Drop %"].round(0).astype(int)
    t["Beyond peers (pts)"] = t["Beyond peers (pts)"].round(0).astype(int)
    t["Est. unbilled kWh"] = t["Est. unbilled kWh"].round(0).astype(int)
    return t


def build_emails(scores: pd.DataFrame, officers: pd.DataFrame, eval_date: date) -> tuple[list[Email], list[str]]:
    """One email per SE covering all flagged meters in their sub-divisions.

    Returns (emails, unmapped_subdivisions): sub-divisions with flagged meters but no SE.
    """
    flagged = scores[scores["status"] == "FLAGGED"].sort_values("est_unbilled_kwh", ascending=False)
    mapped = flagged.merge(officers, on="subdivision_id", how="left")
    unmapped = sorted(mapped.loc[mapped["se_email"].isna(), "subdivision_id"].dropna().unique())
    week = pd.Timestamp(eval_date).strftime("%d %b %Y")

    emails = []
    for (name, addr), grp in mapped.dropna(subset=["se_email"]).groupby(["se_name", "se_email"]):
        table = _table(grp)
        n, kwh = len(grp), float(grp["est_unbilled_kwh"].sum())
        sds = sorted(grp["subdivision_id"].unique())
        subject = f"Suspected consumption-drop cases, week to {week}: {n} flagged for inspection, {kwh:,.0f} kWh at risk"
        rows = "".join(
            "<tr>" + "".join(f"<td style='padding:4px 8px;border-bottom:1px solid #ddd'>{escape(str(v))}</td>" for v in r) + "</tr>"
            for r in table.itertuples(index=False)
        )
        head = "".join(f"<th style='text-align:left;padding:4px 8px;border-bottom:2px solid #999'>{escape(c)}</th>" for c in table.columns)
        html = f"""<div style="font-family:Arial,sans-serif;font-size:14px;color:#222">
<p>Dear {escape(name)},</p>
<p>The weekly consumption-drop analysis for the week to <b>{week}</b> found
<b>{n} connections</b> in {escape(', '.join(sds))} whose consumption fell much more steeply than comparable
connections on the same transformer. They are <b>recommended for inspection</b>, with an estimated
<b>{kwh:,.0f} kWh</b> possibly unbilled.</p>
<p>These are <b>suspected</b> cases only. A finding of theft is made by a field officer after inspection,
not by this analysis.</p>
<table style="border-collapse:collapse;font-size:13px"><tr>{head}</tr>{rows}</table>
<p style="color:#666;font-size:12px">The full list is attached as CSV. Generated automatically by the consumption-drop analytics pipeline.</p>
</div>"""
        emails.append(Email(name, addr, sds, subject, html, table, n, kwh))
    return emails, unmapped


AUTH_HELP = ("Login rejected (535): username or password not accepted. For Gmail, use a 16-character "
             "App Password from myaccount.google.com/apppasswords, not your normal password.")


def _login(server: smtplib.SMTP, user: str, password: str):
    """Log in with AUTH PLAIN only. smtplib.login() falls through to AUTH LOGIN after a
    rejection, and Gmail drops the connection, hiding the real 535 error."""
    server.ehlo_or_helo_if_needed()
    server.user, server.password = user, password
    try:
        server.auth("PLAIN", server.auth_plain)
    except smtplib.SMTPException:
        if "plain" in server.esmtp_features.get("auth", "").lower():
            raise
        server.login(user, password)  # server without PLAIN: let smtplib pick a method


def _read_log(log_path: Path) -> pd.DataFrame:
    if log_path.exists():
        return pd.read_csv(log_path, dtype=str)
    return pd.DataFrame(columns=LOG_COLUMNS)


def send_emails(emails: list[Email], eval_date: date, cfg: Config, smtp_user: str, smtp_password: str,
                log_path: Path, override_to: str | None = None) -> pd.DataFrame:
    """Send each email and append one log row per recipient.

    Idempotent on (eval_date, se_email): an SE already marked SENT for this eval_date is
    skipped, so a retry after a partial failure never double-sends. Test sends
    (override_to) are always sent and logged as TEST_SENT.
    """
    ev = str(pd.Timestamp(eval_date).date())
    log = _read_log(log_path)
    already = set(log.loc[(log["eval_date"] == ev) & (log["status"] == "SENT"), "se_email"])
    results = []

    ctx = ssl.create_default_context()
    server = None
    try:
        for e in emails:
            row = {"eval_date": ev, "se_name": e.se_name, "se_email": e.se_email,
                   "sent_to": override_to or e.se_email, "n_flagged": e.n_flagged,
                   "error": "", "sent_at": datetime.now().isoformat(timespec="seconds")}
            if not override_to and e.se_email in already:
                results.append({**row, "status": "SKIPPED_ALREADY_SENT"})
                continue
            msg = EmailMessage()
            msg["Subject"] = ("[TEST] " if override_to else "") + e.subject
            msg["From"] = f"{cfg.smtp.sender_name} <{smtp_user}>"
            msg["To"] = override_to or e.se_email
            msg.set_content("This email needs an HTML-capable mail client. The flagged list is attached as CSV.")
            msg.add_alternative(e.html, subtype="html")
            msg.add_attachment(e.table.to_csv(index=False).encode("utf-8"), maintype="text", subtype="csv",
                               filename=f"suspected_cases_{ev}_{e.se_email.split('@')[0]}.csv")
            try:
                if server is None:
                    if cfg.smtp.port == 465:  # implicit SSL
                        server = smtplib.SMTP_SSL(cfg.smtp.host, cfg.smtp.port, timeout=30, context=ctx)
                    else:  # 587: plain connection upgraded with STARTTLS
                        server = smtplib.SMTP(cfg.smtp.host, cfg.smtp.port, timeout=30)
                        server.starttls(context=ctx)
                    _login(server, smtp_user.strip(), "".join(smtp_password.split()))
                server.send_message(msg)
                results.append({**row, "status": "TEST_SENT" if override_to else "SENT"})
            except smtplib.SMTPAuthenticationError:
                results.append({**row, "status": "FAILED", "error": AUTH_HELP})
                server = None
            except Exception as exc:  # log every failure per recipient and carry on
                results.append({**row, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
                if isinstance(exc, (smtplib.SMTPAuthenticationError, smtplib.SMTPConnectError, OSError)):
                    server = None
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass

    out = pd.DataFrame(results, columns=LOG_COLUMNS)
    pd.concat([log, out.astype(str)], ignore_index=True).to_csv(log_path, index=False)
    return out
