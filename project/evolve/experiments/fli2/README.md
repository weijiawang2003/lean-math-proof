# FLI2 — Large-Scale Retrieval-Gap Lemma Deployment

FLI2 scales up FLI1's dominant signal: in many proof-search failures the relevant lemma **already
exists and was even retrieved**, but the system never turned it into the right proof *action*.
FLI1 showed one robust such rescue (`Finset.card_le_one` deployed via `simp [L] <;> aesop`). FLI2
asks, across ~200 failures, whether these **retrieval gaps form reusable lemma-deployment families**.

This is a **discovery experiment**, not a production release and not an RC solved-count benchmark.
The research question:

> Given a theorem RC5 failed and a retrieved lemma `L` that looks semantically close, can we decide
> *when* to deploy `simp [L]` / `simp [L] <;> aesop` / `constructor <;> intro h <;> simp [L] at *` /
> `ext x <;> simp [L]` and thereby rescue the original theorem?

## Definitions

- **RETRIEVAL_GAP** — a relevant existing lemma is available/retrieved but the search system fails
  to deploy it.
- **LEMMA_DEPLOYMENT_ACTION** — a small gated proof action using a specific retrieved `L`.
- **DOWNSTREAM_RESCUE** — the original failed theorem becomes provable *at its theorem position*
  when `L` is deployed (controls fail, non-vacuous, robust).
- **DEPLOYMENT_RULE** — a reusable condition under which a class of lemmas should be deployed
  (e.g. "card-inequality goal + `Finset.card_*` lemma → try `simp [L]`/`gcongr`").

Precedents: RC4B (disjoint_left bridge), RC4C (selected `simp [L]` enablers), FLI1
(`Finset.card_le_one`). FLI2 tries to *discover* such rules from failure analysis at scale.

## Pipeline (scripts)

1. `fli2_build_retrieval_gap_pool.py` — pool from FLI1 retrieval-gaps/exists-close + FLI0 high-signal.
2. `fli2_generate_deployment_actions.py` — gated `simp [L]`-style actions per theorem/lemma.
3. `fli2_build_live_eval_plan.py` — theorem-centric plan + controls (at-position, vacuity-safe).
4. `fli2_run_live_deployment_eval.py` — **live** LeanDojo eval (one Dojo per theorem).
5. `fli2_classify_rescues.py` — TRUE_RETRIEVAL_GAP_RESCUE / PARTIAL / CONTROL_DUP / VACUOUS / …
6. `fli2_mine_deployment_rules.py` — reusable deployment-rule candidates.
7. `fli2_compare_to_rc4bc.py` — overlap/extension vs RC4B/RC4C.
8. `fli2_write_deployment_atlas.py` — researcher-facing atlas.

Outputs in `cases/` (jsonl), `out/` (summaries + atlas md), `data/` (rules + atlas json),
`live_traces/`. Report: `project/evolve/reports/fli/fli2_retrieval_gap_deployment_report.md`.

## Safety

At-position live testing only (no vacuous self-import); controls must fail for a rescue;
candidate wins re-run for robustness; bounded per-tactic/per-theorem/process timeouts; temp/no
Mathlib-source edits; nothing promoted; no wrapper/router/README change; no commit.
