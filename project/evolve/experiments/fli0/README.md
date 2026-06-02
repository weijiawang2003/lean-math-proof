# FLI0 — Failure Corpus Extraction for Lemma Invention

FLI0 is a **discovery-preparation** stage, not a production release and not another RC
benchmark. It marks the shift from *proof automation* ("can the system prove this theorem?")
toward *verifier-guided mathematical discovery* ("when the system fails, what intermediate
lemma or proof idea seems to be missing?").

## What FLI0 does

It reads the already-run RC5V2 (complete) and RC5V3 (partial raw) hybrid-search artifacts,
extracts the theorems that the full stack (RC2 → RC4 static → safe dynamic) **failed** to prove,
cleans and enriches them, classifies the *kind* of gap each failure exhibits, and selects a small
set of high-signal **seed cases** for the next stage (FLI1: failure-driven lemma invention).

FLI0 runs **no live Lean** by default — it mines committed/working-tree artifacts. It writes only
under `project/evolve/experiments/fli0/`, `project/evolve/reports/fli/`, and `scripts/fli0_*.py`.

## Core definitions

- **FAILURE_CASE** — a theorem attempted by RC5V2 or RC5V3 but **not solved** by any stage.
- **CLEAN_FAILURE** — a failure with *no* timeout/kill, *no* setup/path/network error, *not*
  caused only by unknown-lemma-name failures, and with a readable error/failed-tactic trace
  (we lack residual goal states, so "readable" = real `proof_failed` attempts exist).
- **LEMMA_INVENTION_SEED** — a clean failure whose **statement + retrieved lemmas + failure
  pattern** suggest a specific missing reusable lemma or bridge.
- **DOWNSTREAM_RESCUE_TARGET** — the original theorem that would likely become provable if that
  intermediate lemma were introduced (i.e. the seed's `theorem`).

## Pipeline (scripts)

1. `fli0_locate_source_artifacts.py` — inventory RC5V2/RC5V3 artifacts (PRESENT/MISSING/PARTIAL).
2. `fli0_extract_failed_cases.py` — per-theorem failure records from both stages.
3. `fli0_enrich_failure_context.py` — add statement, file, difficulty, retrieved lemmas, defs.
4. `fli0_classify_failure_patterns.py` — conservative rule-based pattern taxonomy.
5. `fli0_select_seed_cases.py` — pick 20–40 high-signal seeds for FLI1.
6. `fli0_write_failure_atlas.py` — human-readable atlas.

Outputs land in `cases/` (jsonl case records), `out/` (summaries + atlas md), `data/`
(atlas json). The narrative report is `project/evolve/reports/fli/fli0_failure_corpus_extraction_report.md`.

## What FLI0 is careful about

- It does **not** fabricate an RC5V3 conclusion (RC5V3 analysis layer was never produced —
  see `state_reconciliation.md`). It uses RC5V3 raw per-theorem results only.
- It separates **infra failures** (RC5V3 B5 network outage, timeouts) from **math failures**.
- It does **not** claim a theorem "requires" a new lemma — only that a failure "suggests" /
  "appears to need" a candidate lemma shape. Classification is deliberately conservative.
