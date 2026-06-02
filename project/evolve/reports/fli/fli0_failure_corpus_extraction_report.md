# FLI0 — Failure Corpus Extraction for Lemma Invention

**Decision: `FLI0_READY_FOR_FLI1`**

FLI0 turns the negative results of the RC5 hybrid search into a structured, enriched failure
corpus and selects seed cases for failure-driven lemma invention (FLI1). It is a
discovery-preparation stage — no release, no benchmark, no production change, no commit.

---

## 1. Executive summary

- Mined **RC5V2** (complete) and **RC5V3** (`PARTIAL_ARTIFACTS_AVAILABLE`) hybrid-search artifacts.
- Extracted **455 theorem-level failures**; **327 clean** math failures (the rest: 112 infra/
  network from the RC5V3 B5 outage, 15 unknown-name-only, + a few timeout/missing).
- Classified clean failures into a conservative, multi-label pattern taxonomy. Dominant clean
  patterns: **MAP_FILTER_BIND_BRIDGE (101)**, ORDER_STRUCTURE_GAP (42), **SUBSET_BRIDGE (39)**,
  **IFF_SPLIT (36)**, **MEMBERSHIP_BRIDGE**, **INDUCTION_GENERALIZATION**, DISJOINT_BRIDGE,
  SINGLETON_CHARACTERIZATION. **280 high-signal clean** cases.
- Selected **40 seed cases** for FLI1, diversity-capped across 8 patterns and 5 namespaces
  (Finset 14 / List 14 / Multiset 4 / Set 4 / Nat 4).
- Criteria for `FLI0_READY_FOR_FLI1` all met: ≥20 clean seeds (40), ≥3 high-signal patterns (8),
  no production files modified, clear FLI1 action plan.

## 2. Why FLI0 follows RC5V2/RC5V3

The RC5 arc made the hybrid static+dynamic searcher *safe* (RC5S) and showed it yields fresh
out-of-sample wins (RC5V2: RC2 67 → RC4 67 → RC5V2 75). The natural next question is no longer
"can we prove more by searching harder?" but "**when search fails, what reusable lemma is
missing?**" RC5V2/RC5V3 are the freshest, largest pools of *confirmed* failures (every
dynamic-eligible theorem is a CONFIRMED_RC2_FAILURE that RC4 and the safe dynamic stage also
missed) — the ideal substrate for lemma invention.

## 3. Source artifact inventory

`scripts/fli0_locate_source_artifacts.py` → `out/source_artifact_inventory.{json,md}`.
**FLI0 source = BOTH.**

| stage | status | notes |
|---|---|---|
| RC5V2 | **COMPLETE** | all artifacts + final report present; committed attribution (8 fresh deltas, 141 NO_DYNAMIC_WIN) |
| RC5V3 | **PARTIAL_ARTIFACTS_AVAILABLE** | raw B1/B3/B5 + RC2/RC4 baselines + eligibility + retrieval present; attribution / safety / comparison / cost / yield / maintenance / report all **MISSING**; B1/B3/B5 flagged PARTIAL (>25% setup-error records from a B5 network outage) |

Per instruction, the missing RC5V3 conclusion is **not fabricated** — only RC5V3 raw per-theorem
results are consumed. See `experiments/fli0/state_reconciliation.md`.

## 4. Failure extraction method

`scripts/fli0_extract_failed_cases.py`. A theorem is a FAILURE_CASE iff it was dynamic-eligible
(RC2 failed by construction), RC4 static did not solve it, and the dynamic stage did not solve it.
Dynamic successes and RC2/RC4-solved theorems are excluded. RC5V3 B1/B3/B5 are merged
(earliest live success wins; infra-only records flagged). Dynamic results bucket as
`failed / killed / infra_error / unknown_name / missing`.

| | RC5V2 | RC5V3 | total |
|---|---|---|---|
| failures | 141 | 314 | **455** |
| clean | 134 | 193 | **327** |
| dynamic-result split | — | — | failed 328 / infra_error 112 / unknown_name 15 |

**CLEAN_FAILURE** = a live `proof_failed` attempt with a readable trace (no kill, no setup error,
not unknown-name-only). RC5V2 ∩ RC5V3 = 0.

## 5. Enrichment method

`scripts/fli0_enrich_failure_context.py`. Added (no live Lean, no OCR): statement text (95%
coverage), file path, catalog difficulty/num_tactics, name tokens, involved constants/definitions,
top-5 retrieved lemmas with statement text (100% coverage), a similar SOLVED theorem in-namespace
(356/455), last error, failed-tactic trace. **`residual_goal_status = MISSING` for all** —
post-tactic goal states are absent from every artifact (the central limitation handed to FLI1).

## 6. Failure pattern taxonomy

`scripts/fli0_classify_failure_patterns.py`. Conservative, multi-label, rule-based over feature
vector + statement tokens + failure outcomes. **Triggers are goal-driven** (statement/name/
feature), deliberately *not* retrieved-lemma-name text or the noisy `has_singleton` feature, which
over-fired on `{f : α → β}` implicit binders (an early version mislabeled 122 cases as singleton;
tightened to 8 genuine). Labels: MEMBERSHIP_BRIDGE, SINGLETON_CHARACTERIZATION, DISJOINT_BRIDGE,
SUBSET_BRIDGE, MAP_FILTER_BIND_BRIDGE, IFF_SPLIT, EXTENSIONALITY_NEEDED, INDUCTION_GENERALIZATION,
SIMP_LOOP_OR_RECURSION, UNKNOWN_NAME_OR_IMPORT, ORDER_STRUCTURE_GAP, NAT_ARITH_GAP, LOW_SIGNAL,
NEEDS_REVIEW. Each case carries confidence + NL explanation + candidate-lemma-shape (NL) +
recommended FLI1 probe.

## 7. Seed case selection

`scripts/fli0_select_seed_cases.py`. Deterministic score (clean → fresh → readable statement →
high-value pattern → invention-friendly namespace → retrieval-found-but-unclosed → similar solved)
with per-pattern cap 8 and per-namespace cap 14. **40 seeds** from a pool of 217.

| pattern | seeds | | namespace | seeds |
|---|---|---|---|---|
| SUBSET_BRIDGE | 8 | | Finset | 14 |
| MAP_FILTER_BIND_BRIDGE | 8 | | List | 14 |
| MEMBERSHIP_BRIDGE | 7 | | Multiset | 4 |
| INDUCTION_GENERALIZATION | 6 | | Set | 4 |
| SINGLETON_CHARACTERIZATION | 4 | | Nat | 4 |
| IFF_SPLIT | 4 | | | |
| DISJOINT_BRIDGE | 2 | | | |
| EXTENSIONALITY_NEEDED | 1 | | | |

Example seeds + candidate missing-lemma shapes:

- `Finset.biUnion_nonempty` (MEMBERSHIP_BRIDGE): `(s.biUnion t).Nonempty ↔ ∃ x ∈ s, (t x).Nonempty`
  → candidate `x ∈ <transformed container> ↔ <elementwise condition>` rewrite.
- `Finset.card_le_one` (MEMBERSHIP_BRIDGE): `s.card ≤ 1 ↔ ∀ a ∈ s, ∀ b ∈ s, a = b`.
- `Finset.card_le_card` (SUBSET_BRIDGE): `s ⊆ t → s.card ≤ t.card` → subset→card bridge.

## 8. Failure atlas summary

`scripts/fli0_write_failure_atlas.py` → `out/fli0_failure_atlas.md` + `data/fli0_failure_atlas.json`.
Per-pattern intuition, example theorem, why-search-failed, and candidate lemma shape, with hedged
language ("suggests / appears to need").

## 9. Recommended FLI1 task

1. Re-run the 40 seeds live to **capture residual goal states** (the one missing ingredient).
2. For bridge patterns (membership / subset / disjoint / map-filter-bind), **synthesize the
   candidate `↔` lemma**, prove or retrieve it, deploy it as a gated bare `simp [L]` enabling
   action (the RC4B/RC4C deployment pattern), and check whether the downstream theorem closes.
3. Start with the highest-confidence, most-clustered Finset/List membership & subset families —
   one invented lemma may rescue several theorems.
4. Defer ORDER_STRUCTURE_GAP / NAT_ARITH_GAP.

## 10. Limitations

- **No residual goals** in any artifact → labels inferred from statement + retrieval, not a stuck
  goal. Conservative by design; FLI1 must capture goals live.
- **RC5V3 partial**: B5 network outage → 112 infra-only records excluded from clean; some V3
  theorems have only B1 live data.
- **Finset `card_*` cluster**: several seeds are near-variants — treat as a single bet.
- Pattern labels are heuristic and multi-signal; they indicate a *suggested* family, never a
  *required* lemma.

## 11. Protected-file confirmation

`git diff --stat HEAD` over RC1/RC2/RC4-release/RC5S-policy wrappers + NS24 router = **empty**.
No NS9/REL/RC4*/RC5* committed artifact, TR1–TR7 dataset, or production routing modified. FLI0
wrote only under `project/evolve/experiments/fli0/`, `project/evolve/reports/fli/`, and
`scripts/fli0_*.py`. No README update, no production wrapper, nothing promoted, ranker not
retrained, **no commit**.
