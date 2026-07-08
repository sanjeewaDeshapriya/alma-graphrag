# Human Relevance Annotation Protocol — ALMA-GraphRAG

Purpose: replace the rule-based bootstrap gold standard with human judgments so
evaluation results are publishable. This protocol produces the labels consumed
by `evaluation/annotation/aggregate.py` → `evaluation/gold_human.json`, which
`run_eval.py` uses automatically.

## Who

Minimum **3 annotators** judging every item independently. Suitable annotators:
fellow students or staff who have travelled in/around Colombo, or hospitality
staff. Record each annotator's background (one line) for the paper. Annotators
must not discuss judgments with each other before submitting.

## What you judge

Each row of your sheet (`annotator_N.csv`) is one **(query, hotel)** pair with
the hotel's objective attributes (price, rating, stars, travel time, amenities).
Fill the `relevance` column with exactly one of:

| Score | Meaning | Test |
|-------|---------|------|
| **2** | Fully relevant | You would happily present this hotel as an answer to the query. It satisfies the *main* need and has no disqualifying attribute. |
| **1** | Partially relevant | Reasonable but flawed — satisfies the main need only loosely (slightly over budget, slightly slower to reach) or satisfies a secondary aspect while missing part of the main one. |
| **0** | Not relevant | A traveller asking this would consider it a wrong answer. |

Binarization for metrics: **1 and 2 count as relevant** (documented in the
paper; ties across annotators resolve to non-relevant).

## Rules

1. **Judge the query as a traveller would read it**, not as a database filter.
   "cheap" without a number means cheap *relative to the other hotels shown in
   this city's sheet*, not a fixed threshold.
2. **Missing attribute ≠ automatically irrelevant.** If price is blank and the
   query is about price, judge from the rest (star level, name) — use 1 if
   genuinely uncertain, 0 only when evidence points against.
3. **Multi-constraint queries:** all main constraints must hold for a 2; one
   clearly failed constraint caps the score at 1; two failed → 0.
4. **Travel time** refers to drive time from the city reference point under
   traffic (`travel_time_min` column). Under ~5 min is unambiguously "quick"
   for Colombo; over ~10 min is not.
5. Do not look up hotels online — judge only from the sheet, so all annotators
   see identical evidence.
6. Judge every row; leave `relevance` blank only if a row is uninterpretable
   (this is recorded as a missing judgment, not a 0).

## Procedure

1. Maintainer runs `python evaluation/annotation/make_sheets.py`
   (Neo4j must be up). Pools are the union of every system's top-10 plus
   2 random hotels per query; provenance is hidden from you and logged in
   `sheets/pool_meta.json`.
2. Each annotator gets their own CSV (row order differs per annotator by
   design) and fills the `relevance` column. The current set is ~1,280 rows
   (50 queries × ~26 pooled hotels); at a few seconds per row expect **2–4
   hours total — split it across multiple sittings**, the sheet saves fine
   half-filled.
3. Maintainer collects the filled files back into
   `evaluation/annotation/sheets/` (keeping the `annotator_N.csv` names) and
   runs `python evaluation/annotation/aggregate.py`.
4. Agreement gate: **Krippendorff's alpha (interval) ≥ 0.667** required,
   ≥ 0.800 preferred. Below the gate: hold an adjudication session on the
   items with maximal disagreement, clarify rule interpretations, re-judge
   only those items, and re-run aggregation. Report the final alpha and the
   adjudication process in the paper.
5. Re-run `python evaluation/run_eval.py` — it picks up
   `evaluation/gold_human.json` automatically and marks each query's
   `gold_source` in the results file.

## Reporting checklist (for the paper)

- Number of annotators + one-line backgrounds
- Pool construction (union of top-k across N systems + random extras, seed)
- Scale definition and binarization threshold
- Krippendorff's alpha before and after adjudication
- Any queries dropped and why
