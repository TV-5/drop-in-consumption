"""The Streamlit app driven like a user would, with AppTest. Slow: skip with -m "not app"."""
import datetime

import pandas as pd
import pytest

from tests.conftest import log_rows
from tests.helpers import (OWNER, ROOT, app_test, button, errors, generate, page_bytes, run_detection, to_step1,
                           to_step3)

pytestmark = [pytest.mark.integration, pytest.mark.app]
MASKED = "ow***@example.com"


@pytest.fixture
def at(app_env, fake_smtp):
    return app_test().run()


def step_buttons(at):
    return {b.key.split("_")[0]: b.disabled for b in at.button if b.key and b.key.startswith("step")}


# ------------------------------------------------------------------ intro, navigation, legal
def test_opens_on_the_intro_without_the_progress_track(at):
    assert errors(at) == []
    assert step_buttons(at) == {}
    assert {"intro_start", "intro_start_bottom", "terms"} <= {b.key for b in at.button}


def test_intro_content_is_filled_in_and_drawings_survive(at):
    body = next(h.value for h in at.get("html") if "sec-head" in h.value)
    assert "{THRESHOLD}" not in body and "{FLOOR}" not in body and "20 pts" in body
    assert "<svg" not in body and body.count("data:image/svg+xml") == 5
    assert body.count('class="sec-head"') == 7 and body.count('class="read-track"') == 1


def test_steps_unlock_in_order(at):
    at = to_step1(at)
    assert step_buttons(at) == {"step0": False, "step1": True, "step2": True}
    at = generate(at)
    assert step_buttons(at) == {"step0": False, "step1": False, "step2": True}
    at = run_detection(at)
    assert errors(at) == [] and step_buttons(at) == {"step0": False, "step1": False, "step2": False}


def test_back_and_next(at):
    at = generate(to_step1(at))
    at = button(at, key="wiz_next").click().run()
    assert at.session_state["step"].startswith("2.")
    at = button(at, key="wiz_back").click().run()
    assert at.session_state["step"].startswith("1.")


def test_title_starts_fresh_on_the_intro(at):
    at = run_detection(generate(to_step1(at)))
    at = button(at, key="home").click().run()
    assert at.session_state["step"] == "intro" and at.session_state["data"] == {} and at.session_state["scores"] is None


def test_terms_page_and_back(at):
    at = generate(to_step1(at))
    at = button(at, key="terms").click().run()
    assert any("Terms &amp; disclaimer" in h.value for h in at.get("html"))
    at = button(at, key="terms_back").click().run()
    assert at.session_state["step"].startswith("1.") and at.session_state["data"]


def test_about_link_keeps_the_analysis(at):
    at = generate(to_step1(at))
    at = button(at, key="about").click().run()
    assert at.session_state["step"] == "intro" and at.session_state["data"]


# ------------------------------------------------------------------ detection results
def test_result_popups_and_drill_down_for_every_reason(at):
    at = run_detection(generate(to_step1(at)))
    for key in ("open_flagged", "open_suppressed"):
        at = button(at, key=key).click().run()
        assert errors(at) == [], key
    at = button(at, starts="Investigate a meter").click().run()
    scores = at.session_state["scores"]
    sup = scores[scores["status"] == "SUPPRESSED"]
    at.segmented_control[0].set_value(f"Suppressed ({len(sup)})").run()
    for code, meter in sup.groupby("suppression_code")["meter_id"].first().items():
        at.session_state["drill_meter_False"] = meter
        at.run()
        assert errors(at) == [], code
        assert any(code in w.value for w in at.warning), code


# ------------------------------------------------------------------ step 3: sending
def test_demo_send_goes_to_the_demo_inbox_and_the_temp_log(at, fake_smtp, app_env):
    real_log = ROOT / "delivery_log.csv"
    before = real_log.stat().st_mtime if real_log.exists() else None
    at = to_step3(at)
    at = button(at, starts=f"Send to {MASKED}").click().run()
    assert errors(at) == [] and fake_smtp.to() == [OWNER]
    assert fake_smtp.sent[0]["Subject"].startswith("[TEST] ")
    assert log_rows(app_env)["status"].tolist() == ["TEST_SENT"]
    assert (real_log.stat().st_mtime if real_log.exists() else None) == before


def test_production_is_locked_on_a_public_deployment(at):
    at = to_step3(at)
    at.radio(key="delivery_mode").set_value("Production: send each SE their own list").run()
    assert button(at, starts="Send to 1 SE").disabled
    assert any("ALLOW_SE_DELIVERY" in i.value for i in at.info)


def test_production_sends_when_the_server_allows_it(app_env, fake_smtp, monkeypatch):
    monkeypatch.setenv("ALLOW_SE_DELIVERY", "true")
    at = to_step3(app_test())
    at.radio(key="delivery_mode").set_value("Production: send each SE their own list").run()
    at = button(at, starts="Send to 1 SE").click().run()
    assert log_rows(app_env)["status"].tolist() == ["SENT"] and len(fake_smtp.sent) == 1


def test_daily_limit_blocks_sending(app_env, fake_smtp):
    today = datetime.datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([{"eval_date": "2026-10-04", "se_name": "x", "se_email": "x@example.com", "sent_to": "x@example.com",
                   "n_flagged": 1, "status": "TEST_SENT", "error": "", "sent_at": today}] * 20).to_csv(app_env, index=False)
    at = to_step3(app_test())
    assert button(at, starts=f"Send to {MASKED}").disabled
    assert any("limit" in w.value for w in at.warning)


def test_invalid_directory_blocks_sending(at):
    at = to_step3(at)
    at.session_state["officer_base"] = pd.DataFrame(
        {"subdivision_id": ["SD-JPR-07"], "se_name": ["SE"], "se_email": ["not-an-email"]})
    at.run()
    assert any("Fix the directory" in e.value for e in at.error)
    assert not [b for b in at.button if b.label.startswith("Send to")]


def test_owner_address_never_reaches_the_browser(at):
    at = to_step3(at)
    page = page_bytes(at)
    assert OWNER.encode() not in page and MASKED.encode() in page
    at = button(at, key="step0_done_ok").click().run()  # step 1 also lists the directory
    assert OWNER.encode() not in page_bytes(at)
