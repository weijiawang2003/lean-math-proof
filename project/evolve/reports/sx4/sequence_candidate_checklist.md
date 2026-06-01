# Sequence-candidate checklist (SX4)

Run this **before** crediting any depth-k sequence candidate `A <;> B [<;> ...]` or proposing it as
an RC component. A candidate is credited **only if every gating item passes**. Default to **no
credit** when evidence is missing.

## Gating checklist

| # | question | pass condition | tool |
|---|---|---|---|
| 1 | Does the **literal production baseline** already solve the theorem? | **No** (`baseline_finished == false`) | `scripts/rc3_run_literal_validation.py` (baseline policy) |
| 2 | Does the **candidate** (production ⊕ sequence) solve the theorem? | **Yes** (`candidate_finished == true`) | `rc3_run_literal_validation.py` (candidate policy) |
| 3 | Do **depth-1 controls** solve? (`A` alone or `B` alone on the initial goal) | **No** | live single-shot controls (e.g. `rc3_minimal_relabel_*`) |
| 4 | Does the **production trace already contain an equivalent intermediate-state continuation** (`A`-advanced state then `B`-close)? | **No** (`equivalent_sequence_observed == false`) | `scripts/sx4_trace_sequence_detector.py` + `sx4_sequence_attribution.py` |
| 5 | Is the win **fresh over literal production** (not a reproduction of an already-solved theorem)? | **Yes** | attribution `classification == TRUE_SEQUENCE_DELTA` |
| 6 | Is the sequence **generic** (a reusable battery tactic), not a copy of the library source proof? | **Yes** (no theorem-specific `rw` bridge / source mirror) | manual + `SOURCE_SPECIFIC` check |
| 7 | Does it pass the **off-gate scan** (0 emissions on theorems outside the gate)? | **Yes** (0 off-gate) | `scripts/rc3_preservation_offgate.py` |
| 8 | Does it **preserve canonical floors** (demo_v1 ≥11/15, nat_defs_medium ≥37/38, nat_defs_large_v5 ≥49/65)? | **Yes**, 0 regressions | `rc3_preservation_offgate.py` |
| 9 | Is it **deterministic** (two runs identical, modulo isolated environment open-flakes)? | **Yes** | `scripts/rc3_determinism_flake_audit.py` |
| 10 | Does it have **fresh holdout support** (≥1 genuine win on theorems not used to derive the candidate)? | **Yes** | held-out theorem set + attribution |

A candidate that fails **#1→#5** has **no literal delta** and must not be credited, regardless of how
the custom/proxy runner scored it. #6→#10 are safety/quality gates that additionally gate promotion.

## ⚠️ Explicit warning

> **Never credit a depth-k sequence based only on depth-(k-1) controls.**
>
> "`A <;> B` succeeds, `A` alone fails, `B` alone fails" is **not** sufficient. A best-first search
> with `max_steps > 1` applies `A` at step *i* (advancing the goal) and `B` at step *i+1* (from the
> advanced state) as two ordinary steps — that **is** `A <;> B`. `B` failing *on the initial goal* is
> irrelevant, because production never applies `B` to the initial goal.
>
> **Always compare against a literal production run with the same `max_steps`/`top_k`, and inspect
> its trace for an equivalent `A`-advanced → `B`-close continuation.** This single check (#1 + #4) is
> what the RC3 / SX3_SET_ITE_AESOP over-credit missed: literal RC2 already solved all 5 "wins" via
> `simp [Set.ite]` → `aesop` across two search steps.

## How the RC3 case scored on this checklist

| # | result for `SX3_SET_ITE_AESOP` |
|---|---|
| 1 | ❌ literal RC2 **already solves** all 5 (deferred 4 + fresh 1) |
| 4 | ❌ production trace shows `simp [Set.ite]` → `aesop` continuation (equivalent_sequence_observed=true, full confidence) |
| 5 | ❌ 0 TRUE_SEQUENCE_DELTA; all 5 → PRODUCTION_SUBSUMED |

→ Fails #1/#4/#5 ⇒ **no credit** (the safety gates #7–#9 incidentally passed, but they are moot
without a literal delta). Decision: `REJECT_NO_LITERAL_DELTA`.
