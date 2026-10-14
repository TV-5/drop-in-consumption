# Consumption-drop theft detection: code execution plan

Oct 6, 2026 · @Tushar

## Scope and success criteria

The MVP produces one ranked inspection list per week, per sub-division, built from a single signal: a consumption drop that is steeper than the drop seen by comparable neighbours on the same transformer.

**In scope**

- Detection of partial and full meter bypass through consumption-drop analysis
- Peer-normalised comparison to absorb seasonality
- A suppression layer that removes data-quality artefacts before scoring
- Hierarchy rollup and automated email reports to the responsible engineer
- Capture of field inspection outcomes

**Explicitly out of scope for v1**

- Load-shape correlation (the rhythm check). Add in v2 once the drop calculation is stable.
- DT energy balance and direct hooking detection. Needs reliable consumer-to-DT mapping.
- Any trained model. v1 is deterministic and fully explainable.
- Tamper event correlation, CT manipulation, tariff misuse.
- Real-time scoring. Weekly batch only.

**Done means**

| Criterion | Target |
| --- | --- |
| Pipeline runs end to end on one division's real data | Yes |
| Weekly run completes | Under 30 minutes |
| Suppressed meters carry a logged reason | 100% |
| Every flagged meter has readable reason text | 100% |
| Reports delivered to configured recipients | Logged per recipient |
| Inspection outcomes captured against score id | Schema live, form shipped |

Detection accuracy is deliberately not a launch criterion. It cannot be measured until inspections run, which is why the outcome capture ships in the same release rather than later.

## Architecture

&#91;embedded content: pipeline architecture · 8 stages, 2 rows\]

The accent marks the stage that does the real work. Peer comparison is what absorbs seasonality, and without it the other seven stages produce a list of everyone who owns an air conditioner.

```
src/
  config.py        Config dataclass, YAML loader
  baseline.py      build_baselines
  drop.py          compute_drop
  peers.py         compute_peer_excess
  suppress.py      apply_suppressions
  score.py         score_and_explain
  rollup.py        rollup, report rendering
  deliver.py       dispatch, templates
  runner.py        orchestration, all file and DB access
tests/
  generate.py      synthetic population
  test_*.py        one file per module
```

Pandas for the MVP. One division fits comfortably in memory at daily grain. Build against the dataframe interface so a move to Spark later touches the runner and nothing else.

The runner owning all input and output is the constraint that makes everything else cheap to test. No module opens a file, reads a database or sends an email.

## Data contracts

Fix these before writing any logic. Every module reads and writes against these shapes, so changing a scorer later touches no other module.

**Input: daily consumption (derived from interval data upstream)**

```csv
meter_id,reading_date,kwh_import,intervals_expected,intervals_received,voltage_present_pct
4471028,2026-09-14,3.10,48,47,100.0
4471028,2026-09-15,2.95,48,48,100.0
```

Rolling intervals up to daily before the pipeline keeps the detection code small. The interval table stays the source of truth; the daily table is a materialised view.

**Input: meter master**

```csv
meter_id,consumer_id,dt_id,subdivision_id,tariff_category,sanctioned_load_kw,phase_type,status,status_changed_on,meter_installed_on,has_registered_der
4471028,C8812004,DT-JPR-0142,SD-JPR-07,LT-1 Domestic,3.0,1P,ACTIVE,2023-04-11,2023-04-11,false
```

**Output: one row per meter per evaluation date**

```csv
score_id,meter_id,eval_date,baseline_daily_kwh,current_daily_kwh,drop_pct,peer_drop_pct,excess_drop_pct,drop_kwh_per_day,onset_date,est_unbilled_kwh,peer_group_size,comm_availability_pct,status,suppression_code,reason_text,model_version
```

`status` takes one of three values and every meter gets a row:

| Status | Meaning |
| --- | --- |
| `FLAGGED` | On the inspection list |
| `CLEAR` | Scored, below threshold |
| `SUPPRESSED` | Not scored, `suppression_code` says why |

Suppressed meters keep a row rather than disappearing. Without that you cannot answer how many high drops were discarded and why, and that question is where data-quality problems surface.

**Constants, in config not code**

```yaml
baseline_days: 90
baseline_gap_days: 30
current_window_days: 30
min_baseline_days_covered: 60
min_peer_group_size: 20
min_comm_availability_pct: 85
min_drop_kwh_per_day: 2.0
excess_drop_threshold_pct: 20.0
```

Every threshold lives here. Tuning during the pilot must never require a code change.

## Pipeline modules

Five pure functions. Each takes a dataframe and config, returns a dataframe. No module reads from disk or writes to it; the runner handles all I/O. This is what makes the whole thing testable without a database.

### 1. Baseline builder

```python
def build_baselines(daily: pd.DataFrame, eval_date: date, cfg: Config) -> pd.DataFrame:
    """One row per meter: baseline_daily_kwh, days_covered, window bounds.

    Window ends cfg.baseline_gap_days before eval_date and runs back
    cfg.baseline_days. The gap is load-bearing: without it a slow-onset
    bypass contaminates its own reference.

    Uses median, not mean. Returns days_covered so the caller can reject
    thin baselines rather than trusting a number built on 12 days.
    """
```

### 2. Drop calculation

```python
def compute_drop(daily: pd.DataFrame, baselines: pd.DataFrame,
                 eval_date: date, cfg: Config) -> pd.DataFrame:
    """Adds current_daily_kwh, drop_pct, drop_kwh_per_day, comm_availability_pct.

    Both the percentage and the absolute kWh are carried forward. A 70% drop
    on a consumer using one unit a day is noise.
    """
```

### 3. Peer comparison

```python
def compute_peer_excess(drops: pd.DataFrame, master: pd.DataFrame,
                        cfg: Config) -> pd.DataFrame:
    """Adds peer_drop_pct, excess_drop_pct, peer_group_size.

    Peer group: same dt_id, same tariff_category, sanctioned load within
    one band. Falls back to feeder level when the group is smaller than
    cfg.min_peer_group_size. Widen the geography before widening the tariff.

    excess_drop_pct = drop_pct - peer_drop_pct

    This absorbs seasonality, local load shedding and festival effects with
    no weather data and no second year of history.
    """
```

### 4. Suppression

```python
def apply_suppressions(scored: pd.DataFrame, master: pd.DataFrame,
                       cfg: Config) -> pd.DataFrame:
    """Sets status=SUPPRESSED and a suppression_code. Drops no rows."""
```

Codes, evaluated in this order so the first match wins and the reason is the most specific one:

| Code | Condition |
| --- | --- |
| `COMM_POOR` | comm\_availability\_pct below threshold |
| `DISCONNECTED` | master status not ACTIVE |
| `METER_REPLACED` | meter\_installed\_on falls inside either window |
| `NEW_CONNECTION` | baseline days\_covered below minimum |
| `LOAD_REVISED` | sanctioned\_load\_kw changed in window |
| `DER_PRESENT` | has\_registered\_der is true |
| `BELOW_FLOOR` | drop\_kwh\_per\_day under the absolute floor |

### 5. Scoring and reason text

```python
def score_and_explain(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Sets status, est_unbilled_kwh, reason_text.

    FLAGGED when excess_drop_pct exceeds threshold and status is not
    already SUPPRESSED.

    est_unbilled_kwh = drop_kwh_per_day * days since onset_date

    reason_text is generated from a template, never free-form:
      "Consumption down {drop_pct}% since {onset_date}, {excess_drop_pct}
       points steeper than {peer_group_size} comparable connections on the
       same transformer. About {est_unbilled_kwh} units unbilled.
       Meter reporting normally ({comm_availability_pct}%)."

    Ranking is by est_unbilled_kwh, not by excess_drop_pct. A 10 kW
    connection at 40% excess outranks a 2 kW connection at 60%.
    """
```

The word "suspected" or "flagged for inspection" appears in every piece of generated text. No output ever states that theft occurred.

## Reporting and distribution

Three report cuts from one set of scores. Each hierarchy level gets a different document because a Superintending Engineer and a Junior Engineer want opposite things from the same data.

| Level | Recipient | Content |
| --- | --- | --- |
| Circle | Superintending Engineer | Prevalence per 1,000 consumers, week-on-week delta, worst three divisions |
| Division | Executive Engineer | Transformer hotspots ranked by kWh at risk |
| Sub-division | Assistant Engineer | Named inspection list with reason text |

### Rollup rules

Aggregate counts and energy. Never aggregate scores: an average excess-drop across a circle is a meaningless number that looks authoritative, and someone will put it on a dashboard and then argue about it.

```python
def rollup(scores: pd.DataFrame, hierarchy: pd.DataFrame,
           level: str) -> pd.DataFrame:
    """Per node: flagged_count, consumers_total, flagged_per_1000,
    est_unbilled_kwh, est_revenue_at_risk, delta_vs_last_cycle,
    last_cycle_hit_rate.
    """
```

The headline is prevalence plus rupees, not count. "212 flagged" means nothing without a denominator. "18 per 1,000 against a circle average of 11" means something immediately.

### Configuration tables

```sql
dim_hierarchy      node_id PK, parent_node_id, level, name, is_active
dim_officer        officer_id PK, name, designation, email, mobile, status
map_officer_scope  officer_id FK, node_id FK, role,
                   effective_from, effective_to
cfg_schedule       schedule_id PK, hierarchy_level, frequency, day_of_week,
                   send_time, channels, template_id, min_threshold, is_active
fact_report_run    run_id PK, schedule_id, node_id, period_start, period_end,
                   generated_at, status, artifact_path
fact_delivery_log  delivery_id PK, run_id FK, officer_id FK, channel,
                   status, sent_at, error_code, retry_count
```

Three decisions worth defending in review:

1. **Scope mapping is time-bounded.** Engineers transfer constantly. Without `effective_from` and `effective_to`, reports keep going to someone who left in March and nobody notices until the numbers come up in a review.
2. **Route to the scope, not the person.** A schedule says "every circle, weekly, Monday 07:00". The system resolves the current SE at send time. A transfer updates one row, not forty schedules.
3. **`min_threshold` suppresses empty reports.** Weekly mail saying "nothing to report" trains people to filter the sender, and then the one that mattered gets missed too.

### Delivery

```python
def dispatch(run_id: str) -> list[DeliveryResult]:
    """Resolves recipients, renders per level, sends, logs per recipient.

    Idempotent on (run_id, officer_id): a retry after partial failure
    must never double-send.

    Three consecutive bounces to an officer escalates: next cycle also
    copies their superior and raises a flag in the admin UI. A silently
    broken address means an entire circle goes dark.
    """
```

Subject line carries the numbers, because for most recipients it is the only thing they will read:

> Jaipur Circle, week to 5 Oct: 212 flagged, 48,300 kWh at risk, up 14%

Email body is one screen: three headline numbers, top five hotspot transformers, last cycle's result, a link to the dashboard. SMS to the configured mobile carries a 160-character nudge pointing at the email; it is not a delivery channel for the report itself.

### The section that makes it trusted

Every report carries the previous cycle's outcome:

> Of 180 flagged on 28 Sep: 94 inspected, 38 confirmed, hit rate 40%, Rs 6.2 lakh assessed.

This is the highest-leverage element in the design and the one usually left out. Without it the recipient has no basis to believe this week's list and the report becomes another automated mail nobody opens. It requires `fact_inspection` to be populated, which is why the field outcome form ships in the same release. They are one feature, not two.

### Data protection

The sub-division report is the only one carrying consumer names and addresses. Link it to an authenticated portal view rather than attaching it. Circle and division reports stay aggregate and can be attached as PDF. Official mailboxes are often NIC-hosted with tight attachment limits, which points the same way.

## Testing and validation

You cannot measure detection accuracy before inspections run. So validation splits into two things: correctness, which you can test today, and effectiveness, which only the field can answer.

### Synthetic data generator

Build this before the pipeline. It is the only way to test logic when no labelled real data exists.

```python
def generate_population(n_meters: int, days: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (daily_consumption, ground_truth).

    Injects, with known meter ids and onset dates:
      - partial bypass: step change to 30-50% of normal
      - full bypass: step to near zero, voltage still present
      - vacancy: step to zero, no residual load
      - comms failure: missing intervals, consumption unchanged
      - seasonal swing: applied to every meter, so peer logic is exercised
      - meter replacement, new connections, DER additions

    ground_truth carries meter_id, injected_type, onset_date, true_unbilled_kwh.
    """
```

The seasonal swing matters most. Without it a naive drop calculation passes every test and then flags half a circle in September.

### Unit tests that must exist

| Test | Asserts |
| --- | --- |
| Baseline gap | A bypass starting inside the gap does not shrink the computed drop |
| Median robustness | One 10x outlier day moves the baseline under 2% |
| Thin baseline | Under 60 days covered returns null, not a number |
| Seasonal neutrality | Uniform 40% seasonal drop flags nobody |
| Peer fallback | Group under 20 widens to feeder, not to another tariff |
| Suppression order | A disconnected meter with poor comms reports `COMM_POOR` first |
| Absolute floor | 70% drop on 0.9 kWh/day does not flag |
| Idempotency | Re-running the same eval\_date produces identical score ids |

### Backtest

On real data, run the pipeline against a historical date and compare the flagged list to inspections the DISCOM already carried out in the following weeks. This is weak evidence, since past inspections were complaint-driven and biased, but it answers the question that kills pilots: does the model flag people the utility already knew about, or does it surface anyone new. If the overlap is near total, the model adds nothing beyond the existing process.

### Shadow run

Four weeks generating the list without sending it anywhere. Compare each week's output to the last. High week-to-week churn in the flagged set means the thresholds are unstable and the field will lose confidence fast. A stable list with gradual turnover is what you want before the first real send.

## Build sequence

Ten weeks, sequential. Each step has a checkpoint that gates the next one.

1. **Week 1: data readiness audit.** No code. On one real circle, measure consumer-to-DT mapping completeness, daily load-profile availability per meter, and the current inspection hit rate. Output is a one-page scorecard.
   - *Gate:* load-profile availability above 85% on at least one division. Below that, stop and fix ingestion first. Everything downstream inherits this ceiling.
2. **Week 2: synthetic generator and config scaffold.** Generator, `Config` loader, project skeleton, CI running tests on every commit.
   - *Gate:* generator produces a population whose injected bypasses are visible by eye in a plot.
3. **Week 3: baseline builder and drop calculation.** Modules 1 and 2, with their unit tests.
   - *Gate:* baseline gap test and median robustness test pass.
4. **Week 4: peer comparison.** Module 3 and the fallback logic.
   - *Gate:* seasonal neutrality test passes. A uniform 40% seasonal drop flags nobody.
5. **Week 5: suppression and scoring.** Modules 4 and 5, reason-text templates, the CSV output contract.
   - *Gate:* full pipeline runs end to end on synthetic data and recovers the injected bypasses without flagging the injected vacancies.
6. **Week 6: real data run and backtest.** Point the pipeline at one division's actual history. Expect this week to be mostly data cleaning, not modelling.
   - *Gate:* run completes under 30 minutes; suppression rates are explainable rather than surprising.
7. **Week 7: hierarchy rollup and report rendering.** Rollup function, three report templates, PDF and portal view.
   - *Gate:* a real SE reads the circle report and can say what action it implies.
8. **Week 8: configuration UI and delivery.** Officer directory, scope assignment against the hierarchy tree, schedule config, delivery log, test send.
   - *Gate:* a transfer simulated in the UI reroutes the next run's report correctly.
9. **Week 9: inspection outcome capture.** The field form, the `fact_inspection` write path, the random-sample allocation.
   - *Gate:* an inspection recorded in the field appears against its originating `score_id`.
10. **Weeks 10 to 13: shadow run.** Pipeline and reports generated weekly, delivered to nobody. Review list stability and suppression reasons each week.
    - *Gate:* two consecutive weeks with stable output and no unexplained suppressions, then go live with one sub-division.

The sequencing principle: nothing that touches a real engineer's inbox ships until the list has been stable for two weeks. A field team that gets one bad list in week one is harder to win back than a team that waits three weeks.

## Open decisions and risks

### Needs an answer before week 3

- [ ] Is this a Genus internal pilot or a portfolio artifact? Internal means real schemas and real constraints. Portfolio means synthetic data throughout and a sharper focus on the scorecard and the contracts.
- [ ] Which division, and can its daily consumption history be extracted for 180 days?
- [ ] Pandas or Spark. Pandas handles a division comfortably; a full DISCOM at interval grain does not fit. Build against the dataframe interface so the swap is contained.
- [ ] Does the hierarchy tree exist as data, or does it have to be built from naming conventions in the meter master?

### Needs an answer before week 8

- [ ] Email transport: SMTP relay, a provider API, or the utility's existing notification service.
- [ ] Does the sub-division report link to a portal that exists, or does one have to be built? This is the difference between a two-week and a five-week reporting phase.
- [ ] Who owns officer contact data, and how does it stay current against transfers.

### Risks, with the mitigation already in the plan

| Risk | Mitigation |
| --- | --- |
| Consumer-to-DT mapping too poor for peer groups | Week 1 gate; fall back to feeder-level peers |
| Seasonality floods the list | Peer normalisation; seasonal neutrality test |
| Comms failures masquerade as drops | `COMM_POOR` suppression ahead of all other codes |
| Field team loses trust after one bad week | Four-week shadow run before any send |
| Nobody fills the inspection form | Form ships with the report, not after; hit rate appears in every report so the loop is visibly used |
| Training labels become self-confirming | Random-sample allocation budgeted from week 9, not retrofitted |

### The one that is not mitigated

Consumption drop alone is a weak signal. It will flag people who lost a job, had a death in the family, or sent a family member away. The peer comparison reduces this but does not remove it.

Two things follow. First, add the load-shape correlation in v2 as the priority enhancement; it is the cheapest available discriminator between bypass and genuine hardship. Second, every artefact the system produces says *suspected* and *recommended for inspection*. A weekly email telling a government officer that 212 named citizens are committing theft, on a signal with a 40% hit rate, is a problem waiting to surface. The finding of theft is made by a field officer under the Electricity Act, not by this pipeline.
