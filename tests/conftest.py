"""Shared fixtures: generated population, fake mail server, and a safe environment for app tests."""
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage

import pandas as pd
import pytest

from sample_data.generate import generate_population
from src.config import load_config
from src.pipeline import run_pipeline
from src.schemas import SCHEMAS, parse
from tests.helpers import EVAL, OWNER, ROOT


@pytest.fixture(scope="session")
def real_cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def generated(real_cfg):
    """The app's default population (Vidyadhar Nagar, 180 days to 4 Oct 2026), parsed and scored once per run."""
    raw = generate_population(180, 7, "2026-10-04", "SD-JPR-07")
    data = {k: parse(raw[s["filename"].removesuffix(".csv")].astype(str), k)[0] for k, s in SCHEMAS.items()}
    scores = run_pipeline(data["daily"], data["master"], EVAL.date(), real_cfg)
    return {**data, "truth": raw["ground_truth"], "scores": scores}


# ------------------------------------------------------------------ fake mail server
@dataclass
class Mailbox:
    """What the fake server saw, and how it should misbehave."""
    sent: list[EmailMessage] = field(default_factory=list)
    logins: list[tuple[str, str]] = field(default_factory=list)
    fail_login: bool = False
    refuse: set[str] = field(default_factory=set)  # recipients the server rejects

    def to(self) -> list[str]:
        return [m["To"] for m in self.sent]


@pytest.fixture
def fake_smtp(monkeypatch):
    box = Mailbox()

    class FakeServer:
        def __init__(self, host, port, timeout=None, context=None):
            self.esmtp_features = {"auth": "PLAIN LOGIN"}
            self.user = self.password = None

        def ehlo_or_helo_if_needed(self):
            pass

        def starttls(self, context=None):
            pass

        def auth_plain(self, challenge=None):
            return ""

        def auth(self, mechanism, authobject):
            if box.fail_login:
                raise smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")
            box.logins.append((self.user, self.password))

        def login(self, user, password):
            self.user, self.password = user, password
            self.auth("PLAIN", None)

        def send_message(self, msg):
            if msg["To"] in box.refuse:
                raise smtplib.SMTPRecipientsRefused({msg["To"]: (550, b"mailbox unavailable")})
            box.sent.append(msg)

        def quit(self):
            pass

    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeServer)
    monkeypatch.setattr(smtplib, "SMTP", FakeServer)
    return box


# ------------------------------------------------------------------ app environment
@pytest.fixture
def app_env(monkeypatch, tmp_path):
    """Mail settings for the app under test. Every key is set explicitly so a developer's own .env is never used,
    and the delivery log goes to a temp file."""
    log = tmp_path / "delivery_log.csv"
    for key, value in {"SMTP_USER": "sender.demo@example.com", "SMTP_PASSWORD": "abcdefghijklmnop",
                       "MAIL_TO": OWNER, "ALLOW_SE_DELIVERY": "", "DELIVERY_LOG": str(log)}.items():
        monkeypatch.setenv(key, value)
    return log


def log_rows(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame()
