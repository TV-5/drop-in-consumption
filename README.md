# Consumption-drop detection

[![tests](https://github.com/TV-5/drop-in-consumption/actions/workflows/tests.yml/badge.svg)](https://github.com/TV-5/drop-in-consumption/actions/workflows/tests.yml)

Find homes where electricity may be going unbilled. This app spots smart meters whose use dropped far
more than their neighbours’, rules out innocent explanations, and emails a short, explained inspection
list to the engineer responsible.

**[Try the live demo →](https://drop-in-consumption-6uxgnwbsd6rg3e52fk4wqt.streamlit.app/)**
It runs on generated data, takes about three minutes, and needs no sign-up.

## The problem

A common way to steal power is to rewire around the meter (a “bypass”), so it records only part of what
a home uses. Utilities have far more meters than inspectors, so the useful question is not “who might be
stealing?” but “which few homes are worth a visit this week?”.

## How it decides

1. **Compare each home with its own past.** Typical daily use over the last 30 days (“now”) against the
   90 days before that (“before”), with a 30-day gap in between so a slowly starting bypass cannot hide
   in its own baseline.
2. **Compare it with its neighbours.** Homes on the same transformer share weather, seasons and
   festivals. A meter is flagged only when its drop is more than 20 percentage points steeper than its
   neighbours’ and at least 2 kWh a day.
3. **Rule out innocent explanations.** Rooftop solar, vacant homes, meters that stopped reporting, new or
   replaced meters, disconnections and tiny drops are set aside, each with a recorded reason.

Flagged homes are ranked by estimated unbilled energy, and every flag carries a plain-language reason.
Outputs always say *suspected* or *recommended for inspection*: only a field inspection can establish theft.

## The demo

1. **Generate data** for one sub-division: a realistic population of smart meters with known problems
   planted in some of them.
2. **Detect suspected thefts**, then compare the results with the planted scenarios.
3. **Send to the engineer**: the inspection list is emailed (in the demo, to a single demo inbox).

## Product decisions

- **Explainable, rule-based v1.** No trained model, so every flag can be explained in one sentence.
- **No weather data needed.** The neighbour comparison absorbs seasonal and local effects.
- **Every threshold in config.** `config.yaml` holds all limits, so a pilot can tune them without code changes.
- **Measured by inspections.** Accuracy can only be known from inspection outcomes, so recording those
  outcomes is the next thing to build.

## Project structure

```
app.py               Streamlit app: intro, the three steps, emails
content/             Intro and Terms page text
config.yaml          Every detection threshold
src/
  baseline.py        "Before" and "now" periods and each meter's baseline
  drop.py            Each meter's drop, in % and kWh/day
  peers.py           Neighbours' drop and how far beyond it each meter fell
  suppress.py        Innocent explanations, each with a reason code
  score.py           Flag, rank and write the reason text
  pipeline.py        Runs the steps above in order
  deliver.py         Builds and sends the emails
  schemas.py         Input data contracts and validation
  config.py          Loads config.yaml
  legal.py           Copyright details used by the app and emails
sample_data/         Synthetic population generator and mock network hierarchy
tests/unit/          One test file per module
tests/integration/   Pipeline flows, email (fake mail server), the app driven end to end
```

## Run it locally

Requires Python 3.11 or later.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open http://localhost:8501.

To send email, copy `.env.example` to `.env` and fill in `SMTP_USER`, `SMTP_PASSWORD` (a Gmail App
Password) and `MAIL_TO`. Without them, the app still shows a preview of every email. On Streamlit
Community Cloud, put the same keys in the app’s Secrets.

## Tests

About 120 automated tests run on every push (GitHub Actions), with a coverage floor of 90%; the detection code is
covered almost completely.

- **Unit tests** check each module on its own: the before/skipped/now periods, drops, neighbour comparison, every
  innocent-explanation rule, scoring and ranking, data validation, config, email building and the display helpers.
- **Integration tests** run the whole pipeline on generated data (every planted scenario must get its expected
  outcome), send email through a fake mail server (no real email), and drive the app from the intro to sending,
  including step locking, the production-delivery lock, the daily limit and the guarantee that the demo inbox
  address never reaches the browser.

```bash
pip install -r requirements-dev.txt
python -m pytest                      # everything (about 3 minutes)
python -m pytest -m "not app"         # skip the slower app tests
python -m pytest --cov=src --cov=sample_data --cov-report=term-missing
```

## Licence

Copyright © 2026 Tushar Varma. All rights reserved.

The code is public so it can be read and evaluated, and you may run it locally for that purpose. It
may not be copied, modified, redistributed or deployed without written permission. See [LICENSE](LICENSE), and ask through
[GitHub](https://github.com/TV-5). Third-party software keeps its own licences; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

All data in the app is generated. Place names are used for realism only, and the project is not
affiliated with any electricity utility or government body.
