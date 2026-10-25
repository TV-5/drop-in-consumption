"""Building and sending emails end to end, against a fake mail server (no real email is ever sent)."""
import datetime
import io

import pandas as pd
import pytest

from src.config import Config
from src.deliver import build_emails, send_emails
from tests.conftest import log_rows
from tests.helpers import OWNER

pytestmark = pytest.mark.integration
EVAL = datetime.date(2026, 10, 4)


@pytest.fixture
def emails():
    base = {"consumer_id": "C", "dt_id": "DT-1", "drop_pct": 60.0, "excess_drop_pct": 40.0,
            "onset_date": datetime.date(2026, 9, 20), "reason_text": "Suspected", "status": "FLAGGED"}
    scores = pd.DataFrame([{**base, "meter_id": "1", "subdivision_id": "SD-1", "est_unbilled_kwh": 100.0},
                           {**base, "meter_id": "2", "subdivision_id": "SD-2", "est_unbilled_kwh": 50.0}])
    officers = pd.DataFrame({"subdivision_id": ["SD-1", "SD-2"], "se_name": ["SE A", "SE B"],
                             "se_email": ["a@example.com", "b@example.com"]})
    return build_emails(scores, officers, EVAL)[0]


def send(emails, log, **kw):
    return send_emails(emails, EVAL, Config(), "sender@example.com", "abcd efgh ijkl mnop", log, **kw)


def test_demo_mode_sends_everything_to_the_demo_inbox(fake_smtp, emails, tmp_path):
    log = tmp_path / "log.csv"
    res = send(emails, log, override_to=OWNER)
    assert fake_smtp.to() == [OWNER, OWNER]
    assert all(m["Subject"].startswith("[TEST] ") for m in fake_smtp.sent)
    assert set(res["status"]) == {"TEST_SENT"}
    assert set(log_rows(log)["sent_to"]) == {OWNER}


def test_production_mode_sends_each_se_their_own_list(fake_smtp, emails, tmp_path):
    res = send(emails, tmp_path / "log.csv")
    assert sorted(fake_smtp.to()) == ["a@example.com", "b@example.com"]
    assert set(res["status"]) == {"SENT"}
    assert not any(m["Subject"].startswith("[TEST]") for m in fake_smtp.sent)


def test_resending_skips_ses_already_sent_for_that_date(fake_smtp, emails, tmp_path):
    log = tmp_path / "log.csv"
    send(emails, log)
    res = send(emails, log)
    assert set(res["status"]) == {"SKIPPED_ALREADY_SENT"} and len(fake_smtp.sent) == 2
    assert len(log_rows(log)) == 4  # every attempt is logged


def test_demo_sends_are_never_skipped(fake_smtp, emails, tmp_path):
    log = tmp_path / "log.csv"
    send(emails, log, override_to=OWNER)
    send(emails, log, override_to=OWNER)
    assert len(fake_smtp.sent) == 4


def test_one_refused_recipient_does_not_stop_the_others(fake_smtp, emails, tmp_path):
    fake_smtp.refuse = {"a@example.com"}
    res = send(emails, tmp_path / "log.csv").set_index("se_email")
    assert res.at["a@example.com", "status"] == "FAILED" and "SMTPRecipientsRefused" in res.at["a@example.com", "error"]
    assert res.at["b@example.com", "status"] == "SENT"


def test_login_failure_keeps_the_mail_servers_reply(fake_smtp, emails, tmp_path):
    fake_smtp.fail_login = True
    res = send(emails, tmp_path / "log.csv")
    assert set(res["status"]) == {"FAILED"} and fake_smtp.sent == []
    assert res["error"].str.contains("App Password").all()
    assert res["error"].str.contains("Server said: 535 5.7.8 Username and Password not accepted").all()


def test_login_uses_trimmed_user_and_password_without_spaces(fake_smtp, emails, tmp_path):
    send_emails(emails[:1], EVAL, Config(), "  sender@example.com ", "abcd efgh ijkl mnop", tmp_path / "log.csv")
    assert fake_smtp.logins == [("sender@example.com", "abcdefghijklmnop")]


def test_attachment_is_the_inspection_table(fake_smtp, emails, tmp_path):
    send(emails[:1], tmp_path / "log.csv")
    (msg,) = fake_smtp.sent
    (att,) = list(msg.iter_attachments())
    assert att.get_filename() == "suspected_cases_2026-10-04_a.csv"
    table = pd.read_csv(io.StringIO(att.get_content()), dtype=str)
    assert table["Meter"].tolist() == ["1"]


def test_starttls_port_also_works(fake_smtp, emails, tmp_path):
    cfg = Config()
    cfg.smtp.port = 587
    res = send_emails(emails, EVAL, cfg, "sender@example.com", "pw", tmp_path / "log.csv")
    assert set(res["status"]) == {"SENT"}
