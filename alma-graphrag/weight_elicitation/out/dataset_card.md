# ALMA hotel-choice dataset

Generated 2026-09-01T18:45:40+00:00

## Shape

| | |
|---|---|
| Participants | 5 |
| Queries (participant x task) | 18 |
| Rows (candidates) | 625 |
| Positive labels | 18 |
| Rows per query | up to 34 |

Attention-check tasks excluded from the dataset: 1.
Participants who failed the attention check: 1
(rows dropped: 0).

## Files

| File | Format |
|---|---|
| `dataset.csv` | one row per candidate, all columns |
| `dataset.jsonl` | same rows as JSON |
| `dataset.ltr.txt` | `<label> qid:<n> 1:.. 6:.. # <hotel_id>` (RankLib / XGBoost) |
| `groups.txt` | group sizes in qid order |
| `rooms.csv` / `rooms.jsonl` | stage 2: one row per room offer |
| `rooms.ltr.txt` / `rooms_groups.txt` | stage 2 in ranking format |

## Features (in `dataset.ltr.txt` order)

1. `spatial`
2. `accessibility`
3. `facility`
4. `economic`
5. `disruption`
6. `position`

`position` is the 1-based rank the hotel was displayed at, or 0 if the
participant had filtered it out. `shown` in the CSV distinguishes the two.

## Stage 2 — room choice within the chosen hotel

| | |
|---|---|
| Queries (participant x task) | 5 |
| Rows (offers) | 16 |
| Positive labels | 5 |
| Responses with no room recorded | 13 |

Features (in `rooms.ltr.txt` order):

1. `price_delta_lkr`
2. `breakfast`
3. `refundable`
4. `size_sqm`
5. `max_occupancy`
6. `room_position`

Its own qid space, unrelated to the hotel `qid`. Groups are small (2-5) and
vary in size, because the choice set is whatever the chosen hotel actually
offered. `price_delta_lkr` is measured against the cheapest offer in the SAME
hotel, so it is a within-hotel contrast and carries no hotel-level price
information — that lives in the stage-1 `economic` component.

The reason to fit this: `price_delta_lkr` is in rupees, so the ratio of any
other coefficient to the price coefficient is a willingness-to-pay. That is
what lets the five stage-1 weights be reported on a monetary scale rather than
only up to an arbitrary scale factor.

## Caveats a user of this data must know

- **Position is confounded with proximity.** Each task's list opens sorted by
  distance to the task anchor, so the nearest hotels are also the ones at the
  top. `position` is exported so this can be modelled or corrected; a model
  trained without it will attribute the effect of rank to the spatial feature.
- **`spatial` and `accessibility` are strongly correlated** (~0.9 across the
  pool: in Colombo, being close and being quick to reach mostly coincide).
  Treat them as one location signal unless you have a design that separates
  them. Observed correlations at or above 0.5 in this export:
  - `spatial` vs `accessibility`: r = 0.90
  - `facility` vs `economic`: r = -0.69
- **Choice sets are participant-dependent.** Search and filters were available,
  so the candidates actually on screen differ between participants for the same
  task. `shown`, `n_shown`, `final_sort`, `final_query` and `final_max_price`
  describe what each participant had narrowed the list to.
- **Repeated task.** `t10` deliberately repeats `t2`; `repeat_of` marks it. Use
  the pair to measure test-retest consistency, and drop one of them before
  training so the same decision is not counted twice.
