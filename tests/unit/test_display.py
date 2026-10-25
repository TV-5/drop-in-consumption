import base64
import re

import numpy as np
import pandas as pd

from src.display import decision_text, mask, masked_officers, shown, svg_as_img, unmask_officers
from tests.helpers import CFG

INBOX = "owner.demo@example.com"


def test_mask():
    assert mask("tulip.someone@gmail.com") == "tu***@gmail.com"
    assert mask("not an address") == "***"


def test_shown_masks_only_the_demo_inbox():
    assert shown(INBOX, INBOX) == "ow***@example.com"
    assert "Owner.Demo" not in shown("Owner.Demo@Example.com", INBOX)  # case-insensitive match
    assert shown("se@utility.example", INBOX) == "se@utility.example"
    assert shown(INBOX, "") == INBOX


def test_directory_masking_round_trip():
    df = pd.DataFrame({"subdivision_id": ["A", "B"], "se_name": ["x", "y"], "se_email": [INBOX, "se@u.example"]})
    shown_df = masked_officers(df, INBOX)
    assert shown_df["se_email"].tolist() == ["ow***@example.com", "se@u.example"]
    assert unmask_officers(shown_df, INBOX)["se_email"].tolist() == [INBOX, "se@u.example"]
    assert unmask_officers(shown_df, "")["se_email"].tolist() == ["ow***@example.com", "se@u.example"]


def result(**over):
    r = {"status": "FLAGGED", "suppression_code": "", "drop_pct": 72.0, "peer_drop_pct": 19.0,
         "excess_drop_pct": 53.0, "baseline_daily_kwh": 8.6, "current_daily_kwh": 2.4, "drop_kwh_per_day": 6.2}
    return pd.Series({**r, **over})


def test_decision_text_for_a_flagged_meter():
    t = decision_text(result(), CFG)
    assert "**72% less**" in t and "**19% less**" in t and "**53 percentage points more**" in t
    assert "over the limit" in t and "flagged for inspection" in t


def test_decision_text_for_a_set_aside_meter():
    t = decision_text(result(status="SUPPRESSED", suppression_code="DER_PRESENT"), CFG)
    assert "But it is **not flagged**" in t and "solar" in t.lower()


def test_decision_text_without_history():
    t = decision_text(result(drop_pct=np.nan, status="SUPPRESSED", suppression_code="NEW_CONNECTION"), CFG)
    assert t.startswith("There is not enough history")


def test_decision_text_html_mode():
    assert "<b>72% less</b>" in decision_text(result(), CFG, html=True)


def test_svg_becomes_an_image_that_keeps_its_description():
    html = ('<p>x</p><svg aria-hidden="true" width="28" height="28" viewBox="0 0 24 24" stroke-width="1.8">'
            '<path d="M3 21h18"></path></svg>'
            '<svg role="img" aria-label="A chart" viewBox="0 0 600 300" style="width: 100%"><text>Hi</text></svg>')
    out = svg_as_img(html)
    assert "<svg" not in out and out.startswith("<p>x</p>")
    imgs = re.findall(r'<img src="data:image/svg\+xml;base64,([^"]+)" alt="([^"]*)" style="([^"]*)">', out)
    assert [(alt, style) for _, alt, style in imgs] == [("", "width:28px;height:28px;"), ("A chart", "width: 100%")]
    assert base64.b64decode(imgs[0][0]).decode().startswith('<svg xmlns="http://www.w3.org/2000/svg"')
