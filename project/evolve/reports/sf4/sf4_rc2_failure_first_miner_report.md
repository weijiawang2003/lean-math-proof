# SF4 — RC2 Failure-First Frontier Miner

**Branch:** `sx3-depth2-sequence-search`  ·  **Date:** 2026-05-30  ·  **No commit made.**
Mines candidates **only** from literal RC2 failures, runs them live, and gates every apparent win
through SX4 attribution — so nothing is credited that literal production already does.

---

## 1. Executive summary

| stage | result |
|---|---|
| RC2 failure pool collected | **40** (13 confirmed literal-RC2 failures + 27 frontier-unconfirmed), 17 excluded (RC2-solved), 0 unresolved |
| live RC2 confirmation | **27 CONFIRMED_RC2_FAILURE**, 13 NOW_SOLVED_BY_RC2 (filtered out) |
| failure clusters | **10** (largest: 16 Set `iff` theorems) |
| candidate probes generated | **15** across **8** families (generic, cluster-driven, gated) |
| live probe results (pre-SX4) | 26 `no_win`, 1 `baseline_duplicate`, **0 `candidate_win`** |
| **SX4 attribution: TRUE_DELTA** | **0** (26 FAILED_CANDIDATE, 1 BASELINE_DUPLICATE) |
| promising families | **0** rc_candidate, 0 experimental; 1 training_only signal, 7 reject |
| missing-lemma triage | 2 `POSSIBLE_MISSING_BRIDGE_LEMMA`, 1 `PROOF_SEARCH_DEPTH_GAP`, 7 `NEEDS_MORE_DATA` |
| **DECISION** | **`MISSING_LEMMA_TRIAGE_READY`** (secondary: `NEED_MORE_FRONTIER_DATA`) |

**No RC candidate and no experimental family emerged** — which is the *correct, honest* failure-first
outcome: across 27 confirmed RC2 failures, **no generic cluster-driven tactic or sequence produced a
genuine new win over literal production**. The single live close (`Prop.compl_singleton` via bare
`aesop`) is a routing/depth gap, not a sequence delta, and is excluded by SX4. The residual frontier is
structured-proof / missing-lemma territory, now triaged for a future SF5 lemma pass.

This vindicates the methodology: SF4 did **not** manufacture a candidate the way the SX3 depth-2 runner
did. Mining from literal RC2 failures + SX4 attribution returns 0 false credits.

---

## 2. Methodology

**Failure-first principle.** Candidates are mined **only** from theorems literal RC2 demonstrably
fails (`finished == false`, `top_k=8`, `max_steps=8`, `hybrid_evolved`, `ns24_router`), re-confirmed
live. A theorem RC2 already solves can never enter as a "win."

**SX4 attribution gate.** Every apparent probe win is classified; only `TRUE_DELTA` (confirmed RC2
failure + gated probe solves + bare controls fail + depth-1 sub-controls fail + generic) is credited.
See `project/evolve/experiments/sx4/sx4_methodology.md`.

**Why this avoids the RC3 over-credit.** The RC3/SX3 bug credited `simp [Set.ite] <;> aesop` on
theorems literal RC2 already solved via a 2-step search path. SF4 structurally cannot repeat this: (1)
the input pool is *confirmed* RC2 failures; (2) the live probe runner runs the bare controls
(`simp`/`simp_all`/`aesop`/`classical <;> aesop`) and the sequence's depth-1 sub-tactics alongside
each probe, so a "win" that is really a control or a single sub-tactic is caught; (3) default-to-no-
credit for everything but `TRUE_DELTA`.

---

## 3. RC2 failure pool

**Sources** (`scripts/sf4_collect_rc2_failures.py`): the literal-RC2 results
(`rc3_validation/out/literal_rc2_results.json`, authoritative — 13 confirmed failures with traces) and
the SF1 frontier candidate list (`sf1/out/real/frontier_with_paths.jsonl`, 50 rows, unconfirmed),
enriched with SX3 holdout/cluster baselines.

- **Deduplicated by `full_name`**; excluded any theorem literal RC2 solved (17, incl. the 5 SX3
  production-subsumed). Pool = **40** (13 confirmed + 27 frontier-unconfirmed), all with `file_path`,
  0 unresolved.
- **Live confirmation** (`scripts/sf4_confirm_rc2_failures.py`, literal RC2 on all 40): **27
  CONFIRMED_RC2_FAILURE**, **13 NOW_SOLVED_BY_RC2** (frontier rows RC2 actually solves — e.g.
  `Set.inclusion_*` via `aesop`, `Set.ite_empty*` via `simp [Set.ite]`, `Multiset.disjoint_toFinset`
  via induction — correctly dropped). 0 open-flakes, 0 path-errors.

---

## 4. Failure clusters

`scripts/sf4_cluster_failures.py` → 10 clusters (namespace × goal-shape × name-feature × symptom):

| cluster | size | ns | shape | candidate families |
|---|---|---|---|---|
| `Set__iff__iff` | **16** | Set | iff | set_ext / set_iff_constructor / set_subset_antisymm |
| `Set__ite_if__subset` | 3 | Set | subset | set_ite_simp_aesop / set_ite_ext / set_subset_antisymm |
| `Set__ite_if__equality` | 1 | Set | equality | set_ite_simp_aesop / set_ext |
| `Multiset__iff__iff` | 1 | Multiset | iff | multiset_tofinset_simp_aesop |
| 6 × singletons | 1 | Set/unknown | various | set_ext / generic |

The dominant cluster (16) is Set **iff** theorems — `monotoneOn_iff_monotone`,
`strictMonoOn_iff_strictMono`, `ssubset_iff_insert`, `subset_singleton_iff_eq`, etc. — order/monotone
equivalences and subset/ssubset characterizations.

---

## 5. Candidate probes

`scripts/sf4_generate_candidate_probes.py` → **15 probes / 8 families**, all generic and gated
(namespace + name-feature), `promotion_allowed=false`, no source-specific `rw` bridges. Families:
`set_ite_simp_aesop`, `set_ite_ext`, `set_ext_aesop`, `set_ext_simp`, `set_iff_constructor_aesop`,
`set_subset_antisymm` (high-risk), `multiset_tofinset_simp_aesop`, `generic_aesop_simpall`.

**Live probe run** (`scripts/sf4_run_candidate_probes.py`, driver/worker over the 27 confirmed
failures, gated probes + always-on controls + depth-1 sub-controls):

- **0 `candidate_win`**, **1 `baseline_duplicate`** (`Prop.compl_singleton` — bare `aesop`), 26 `no_win`.
- No gated sequence/tactic closed any confirmed failure where the bare controls did not.

---

## 6. SX4 attribution

`scripts/sf4_apply_sx4_attribution.py`:

| class | count |
|---|---|
| `TRUE_DELTA` (credited) | **0** |
| `BASELINE_DUPLICATE` | 1 (`Prop.compl_singleton`) |
| `FAILED_CANDIDATE` | 26 |

`PRODUCTION_SUBSUMED = 0` (expected on a failure-first pool — confirms the baseline was not stale).
**Credited delta over literal RC2: 0.**

---

## 7. Family analysis

`scripts/sf4_analyze_candidate_families.py` — **0 total TRUE_DELTA, no rc_candidate**:

| family | TRUE_DELTA | reco |
|---|---|---|
| generic_aesop_simpall | 0 | training_only (1 isolated `aesop` close = depth-gap signal) |
| set_ite_simp_aesop, set_ite_ext, set_ext_aesop, set_ext_simp, set_iff_constructor_aesop, set_subset_antisymm, multiset_tofinset_simp_aesop | 0 | reject |

No family meets the rc_candidate bar (≥2 TRUE_DELTA, 0 off-gate, generic, deterministic). Determinism
was not exercised — moot with 0 wins.

---

## 8. Missing-lemma triage

`scripts/sf4_missing_lemma_triage.py` — 2 `POSSIBLE_MISSING_BRIDGE_LEMMA`, 1 `PROOF_SEARCH_DEPTH_GAP`,
7 `NEEDS_MORE_DATA`:

- **`POSSIBLE_MISSING_BRIDGE_LEMMA` — `Set__iff__iff` (16).** Order/monotone equivalences
  (`monotoneOn_iff_monotone`, `strictMonoOn_iff_strictMono`, `not_monotoneOn_not_antitoneOn_iff_…`) and
  subset/ssubset characterizations (`ssubset_iff_insert`, `subset_singleton_iff_eq`,
  `subset_pair_iff_eq`). No generic tactic closes these — they need the **specific defining
  equivalence lemma** then a short finish. *Caveat:* several almost certainly have an **existing
  Mathlib lemma** (these are named like standard library results); the right SF5 first step is an
  **existing-lemma / retrieval** search, not synthesis.
- **`POSSIBLE_MISSING_BRIDGE_LEMMA` — `Set__ite_if__subset` (3).** `Set.ite_eq_of_subset_left/right`,
  `Set.subset_ite`: `ite` with a subset hypothesis; source proofs use `ext` + `by_cases` on the
  condition (depth > 2, per-branch), not expressible as a single generic `<;>` sequence.
- **`PROOF_SEARCH_DEPTH_GAP` — `Prop.compl_singleton` (1).** Bare `aesop` closes it in isolation, but
  literal RC2 did not reach it. RC2's `aesop` priority entry is gated to `Set.Finite./Set.toFinset`
  names, so on a `Prop`-namespace goal the gated `aesop` is suppressed and the generative policy did
  not surface it within `max_steps`. **Cheap potential routing tweak** (broaden where bare `aesop` is
  available) — out of SF4 scope; not a sequence/lemma delta.

No lemmas were invented — directions and rationale only.

---

## 9. Decision

### `MISSING_LEMMA_TRIAGE_READY`  (secondary: `NEED_MORE_FRONTIER_DATA`)

The failure-first miner returned **0 RC candidates and 0 experimental families** — correctly, because
no generic cluster-driven tactic/sequence beats literal RC2 on the confirmed failures. The residual is
triaged into two bridge-lemma clusters (one likely already in Mathlib → retrieval first) and one
proof-search/routing depth gap. Many singleton clusters argue for more frontier data before deeper
investment.

---

## 10. Next steps

- **SF5 (existing-lemma search first, then synthesis):** for `Set__iff__iff`, run a Mathlib
  existing-lemma / retrieval pass — these monotone/subset equivalences likely have direct library
  lemmas the policy isn't retrieving. Only escalate to bridge-lemma synthesis for genuinely missing
  shapes.
- **Routing follow-up (separate, cheap):** investigate the `aesop` gate width — `Prop.compl_singleton`
  shows bare `aesop` solving a confirmed RC2 failure that the narrow gate suppresses. A controlled
  widening would need its own off-gate + floors validation through the SX4/RC process (do not promote
  here).
- **`NEED_MORE_FRONTIER_DATA`:** run a larger literal-RC2 frontier eval to grow the non-singleton
  clusters before committing to lemma synthesis.
- Per task constraints: **no RC4, no promotion** — SF4 yields candidates + evidence only.

---

## 11. Protected-file confirmation

`git diff --stat HEAD` for `rc1_production_wrapper.json`, `rc2_release/rc2_production_wrapper.json`,
`ns24_router.json` → **empty (untouched)**. NS9 genome/checkpoints and REL1/RC1/RC2 release reports
untouched. README not modified. No RC4 created, no candidate promoted. **No commit made** (see
`project/evolve/experiments/sf4/out/protected_files_check.txt`).
