# SF5 — Existing-Lemma Retrieval for Missing-Bridge Targets

**Decision: `MOSTLY_PROOF_DEPTH_GAP`** (with a verified secondary `RETRIEVAL_SIGNAL_FOUND`;
**zero** genuinely-missing lemmas — SF6 synthesis is **not** warranted).

---

## 1. Executive summary

SF4 and TR2 left a residual pool of confirmed literal-RC2 failures whose largest
cluster is the *Set iff-equivalence* family, triaged as
`POSSIBLE_MISSING_BRIDGE_LEMMA` / `MISSING_BRIDGE_LEMMA_CANDIDATE`. Before attempting
any lemma synthesis, SF5 asked the cheaper question: **can these be closed by
retrieving an existing Mathlib lemma, and is the gap merely that RC2's search/routing
never reaches it?**

- **Targets:** 20 (deduped, all confirmed literal-RC2 failures).
- **Lemma index:** 5 994 declarations, **96.5 %** with statement text (5 785 from
  local traced Mathlib source incl. `def`s, 209 from the project catalog).
- **Live retrieval probes:** 800 single-line probes over 20/20 live theorems
  (deterministic; outcomes 633 proof_failed / 160 unknown-name / 2 max-recursion /
  **5 success**).
- **Live retrieval wins over literal RC2: 5 / 20**
  - **4 `EXISTING_LEMMA_GAP`** — definitional unfolds `simp [Pred, PredOn]`
    (`monotoneOn_iff_monotone`, `antitoneOn_iff_antitone`,
    `strictMonoOn_iff_strictMono`, `strictAntiOn_iff_strictAnti`).
  - **1 `RETRIEVAL_ROUTING_GAP`** — `Set.Nonempty.subset_pair_iff_eq` closed by
    `aesop (add simp [Set.subset_pair_iff_eq])`.
- **`TRUE_MISSING_BRIDGE_LEMMA`: 0.** Every one of the 20 targets is itself an
  existing, named Mathlib theorem with a real proof in source — nothing is "missing."
- **`PROOF_DEPTH_GAP`: 15** — the remaining targets have multi-step Mathlib proofs
  (`ext`/`by_cases`/`rw`-chain/`refine`) that RC2's bounded battery does not reach;
  single-lemma retrieval does not close them.

**Recommendation:** capture the 5-win retrieval signal as cheap routing/battery
levers (off-by-default, RC2 untouched); send the 15 depth-gap targets to deeper /
structured search; **do not** open SF6 lemma synthesis — the lemmas already exist.

---

## 2. Motivation

The SF4 failure-first miner found 0 cheap TRUE_DELTA over 27 confirmed RC2 failures
and triaged 2 clusters as `POSSIBLE_MISSING_BRIDGE_LEMMA` with the explicit note
*"verify a Mathlib lemma does not already exist."* TR2's active-probing loop labelled
20 cases `MISSING_BRIDGE_LEMMA_CANDIDATE` but exhausted the fresh frontier (0
net-new rows). The standing conclusion was that the blocker is not model training but
unresolved missing-bridge / retrieval targets. SF5 executes the mandated
verification: **retrieval before synthesis**, so that we never synthesize a lemma
Mathlib already ships.

---

## 3. Target set

- **Sources:** TR2 `MISSING_BRIDGE_LEMMA_CANDIDATE` (20) ∪ SF4
  `POSSIBLE_MISSING_BRIDGE_LEMMA` cluster members (19), intersected with confirmed
  literal-RC2 failures (`rc2_failure_confirmation.json`, 27
  `CONFIRMED_RC2_FAILURE`). Deduplicated by `full_name` → **20 targets**.
- **Clusters:** `Set__iff__iff` (16), `Set__ite_if__subset` (3),
  `Multiset__iff__iff` (1).
- Goal text recovered for all 20 (live error echoes + traced-source statement
  backfill). The existing Mathlib **source proof** of each target was also extracted
  (step count + first tactic) — the decisive evidence separating PROOF_DEPTH from any
  TRUE_MISSING claim.

Artifacts: `project/evolve/experiments/sf5/cases/sf5_missing_bridge_targets.json`,
`sf5_target_manifest.json`, `out/sf5_target_summary.md`.

---

## 4. Lemma index

- **Sources:** local LeanDojo-traced Mathlib source (focused dirs:
  `Data/Set`, `Data/Finset`, `Order/Monotone`, `Order/SetNotation`) +
  `project/discovered_theorems.json`.
- **Size / coverage:** 5 994 declarations, **96.5 %** carry statement text.
  Indexed kinds: `theorem`/`lemma` and (newly) `def`/`abbrev` (260 defs) — adding
  definitions was essential, since the 4 `EXISTING_LEMMA_GAP` wins are *definitional*
  unfolds that a theorem-only index could never surface.
- **Limitations:** the catalog source has no statement text (weak name/token/path
  features only); the source scan is restricted to target-relevant directories;
  signatures are parsed lexically up to `:=`. Predicate `def`s (`Monotone` etc.) live
  in the root namespace and lose namespace/path proximity to the `Set.*` targets, so
  they are surfaced via a dedicated **goal-driven def channel** (defs whose name
  literally appears in the goal), not the lexical top-k.

Artifacts: `out/sf5_lemma_index.jsonl`, `sf5_lemma_index_summary.{json,md}`.

---

## 5. Retrieval method

Per target, top-20 lemma candidates by a deterministic combined score:

1. **Lexical TF-IDF cosine** over {name tokens + statement word tokens} (math symbols
   mapped to words: `↔→iff`, `⊆→subset`, …).
2. **Namespace / file-path proximity** (prefix overlap).
3. **Feature overlap** (Jaccard over Set/iff/monotone/strictmono/subset/compl/
   singleton/insert/ssubset/ite/pair/union/empty flags).

Plus two precision channels: a **goal-driven def channel** (definitions named in the
goal, for `simp [Pred, PredOn]` unfolds) and a **cluster-shared-lemma channel** (for
`simp only [...]` cluster probes). The target's own declaration is always excluded.
Retrieval quality is high — for `Set.Nonempty.subset_pair_iff_eq` the top candidate
is the exact bridge `Set.subset_pair_iff_eq`; for the mono/antitone targets the
goal-driven channel recovers `Monotone`/`MonotoneOn` precisely.

Artifacts: `out/sf5_retrieval_results.{json,md}`,
`out/sf5_retrieval_probe_plan.{json,md}` (800 probes; families: exact / simpa_using /
simp_lemma / rw_lemma / aesop_add_simp / def_unfold_simp / cluster_simp / diagnostic
`exact?`+`apply?`; ≤10 lemmas & ≤40 probes/target, parse-risk recorded).

---

## 6. Live probe results

Run through the standard driver/worker LeanDojo harness (one Dojo per theorem,
per-tactic SIGALRM, OS hard timeout — identical to SF4/TR2). All 20 theorems opened
live; RC2-failure status taken from the prior literal confirmation.

| outcome | count |
|---|---|
| proof_failed | 633 |
| unknown_name (lemma out of scope at target) | 160 |
| max_recursion | 2 |
| **success** | **5** |

**Wins (all over a confirmed literal-RC2 failure):**

| target | tactic | family |
|---|---|---|
| `Set.monotoneOn_iff_monotone` | `simp [Monotone, MonotoneOn]` | def_unfold |
| `Set.antitoneOn_iff_antitone` | `simp [Antitone, AntitoneOn]` | def_unfold |
| `Set.strictMonoOn_iff_strictMono` | `simp [StrictMono, StrictMonoOn]` | def_unfold |
| `Set.strictAntiOn_iff_strictAnti` | `simp [StrictAnti, StrictAntiOn]` | def_unfold |
| `Set.Nonempty.subset_pair_iff_eq` | `aesop (add simp [Set.subset_pair_iff_eq])` | aesop+hint |

**Why RC2 misses these:** the 4 predicate definitions are not `@[simp]`, so RC2's
bare `simp`/`aesop` never unfolds them; the hinted-aesop win needs the existing
`Set.subset_pair_iff_eq` lemma as a simp lemma, which RC2's aesop (gated to
`Set.Finite/toFinset`) never adds. Library search (`exact?`/`apply?`) closed **0/20** —
no single existing *term* closes any goal. 160 `unknown_name` outcomes confirm a real
index limitation: many lexically-retrieved lemmas are declared later / not imported at
the target's source position.

Artifacts: `out/sf5_retrieval_probe_results.{json,md}`, `out/probe_run.log`.

---

## 7. Attribution

Every claimed win is over a confirmed literal-RC2 failure (`every_win_over_literal_rc2
= true`); there is no best-first search to subsume a single-shot probe (contrast the
SX3 sequence over-credit, which lived *inside* RC2's own search).

| class | n | meaning |
|---|---|---|
| `EXISTING_LEMMA_GAP` | 4 | definitional `simp [Pred, PredOn]` closes; RC2 fails only because the defs aren't `@[simp]` |
| `RETRIEVAL_ROUTING_GAP` | 1 | existing lemma `Set.subset_pair_iff_eq` + hinted aesop closes; RC2 routing never adds it |
| `PROOF_DEPTH_GAP` | 15 | target has an existing **multi-step** Mathlib proof; no single retrieved lemma closes it |
| `TRUE_MISSING_BRIDGE_LEMMA` | **0** | — |
| `NO_RETRIEVAL_SIGNAL` | 0 | — |

The central correction over the SF4/TR2 "missing-bridge" framing: **all 20 targets are
existing, named Mathlib theorems with real source proofs** (1–12 steps). They are not
missing lemmas. The 15 unsolved ones fail because their proofs are multi-step
(`refine`/`ext`/`by_cases`/`rw`-chains), several even *reuse* other existing lemmas
(`diff_subset_iff`, `union_subset_iff`) but need a 2-step `rw … ; apply …` that no
single-lemma probe expresses. This is exactly the over-claim the methodology warns
against: had SF5 skipped retrieval, these would have been mislabelled as synthesis
targets.

Artifacts: `out/sf5_retrieval_attribution.{json,md}`.

---

## 8. Cluster analysis

| cluster | size | classes | recommendation |
|---|---|---|---|
| `Set__iff__iff` | 16 | 4 existing-lemma, 1 routing, 11 depth | `deeper_search` |
| `Set__ite_if__subset` | 3 | 3 depth | `deeper_search` |
| `Multiset__iff__iff` | 1 | 1 depth | `deeper_search` |

**Set iff-equivalence cluster (the headline question):** retrieval does **not**
collapse the family onto one or two recurring bridge lemmas. Instead it splits into
(a) a clean definitional sub-family (the 4 `<pred>On_iff_<pred>` lemmas, all closed by
`simp [Pred, PredOn]`), and (b) a long tail of structurally distinct multi-step
proofs (singleton/pair/insert/ssubset characterizations) where each target needs its
*own* existing multi-step proof, not a shared new lemma. No single existing lemma
recurs as a winner across targets. **Routing implication:** the only generalizable
retrieval lever is the definitional-unfold pattern for `…On_iff_…` goals; the rest is
search depth.

Artifacts: `out/sf5_cluster_lemma_analysis.{json,md}`.

---

## 9. Training export

20 additive examples in a TR1-compatible schema (`source_artifact: "sf5_retrieval"`),
**never** overwriting the TR1/TR2 datasets:

| label | n | type |
|---|---|---|
| `EXISTING_LEMMA_GAP` | 4 | positive |
| `RETRIEVAL_ROUTING_GAP` | 1 | positive |
| `PROOF_DEPTH_GAP` | 15 | negative |

**How TR3/TR4 should use them:** merge additively as a retrieval-aware label channel.
The positive labels (5) train a router to fire definitional-unfold / lemma-hint
actions on `…On_iff_…`-shaped and pair/subset goals; the negatives preserve
verified-label discipline (a depth-gap is not a cheap-action target). New features
exported: `has_retrieval_win`, `num_named_lemma_wins`, predicate/structure flags.

Artifacts: `out/sf5_training_examples.jsonl`,
`sf5_training_delta_summary.{json,md}`.

---

## 10. Decision

**`MOSTLY_PROOF_DEPTH_GAP`** — 15/20 targets are existing multi-step Mathlib lemmas
that RC2's bounded battery cannot re-derive; single-lemma retrieval does not close
them.

Secondary, verified: **`RETRIEVAL_SIGNAL_FOUND`** — 5/20 close via existing-lemma
retrieval over literal RC2 (4 definitional unfolds + 1 hinted aesop), deterministic
and reproducible.

Explicitly **not** `TRUE_MISSING_LEMMA_CANDIDATES_FOUND`: every target is an existing
named theorem; **0** genuinely missing lemmas. **SF6 lemma synthesis is not
warranted.**

---

## 11. Next steps

Because the signal is *retrieval + depth*, not missing lemmas:

1. **Capture the retrieval signal cheaply (off-by-default, RC2 untouched).**
   - A `…On_iff_…` definitional-unfold battery action: gated `simp [Pred, PredOn]`
     for goals whose head is a predicate-vs-restricted-predicate iff (mirrors the
     MX2 narrow-gate precedent). Candidate gain: 4 deterministic wins.
   - Extend the existing Set-gated aesop fallback to *add* a retrieved simp lemma for
     pair/subset goals (the `Set.subset_pair_iff_eq` case). Candidate gain: 1.
   - These are routing/battery tweaks, validated through a literal RC2⊕candidate run
     before any promotion — **not** promoted here.
2. **Train a retrieval-aware router (TR3).** Use the 20 SF5 labels additively; the
   positive class is "fire a definitional-unfold / lemma-hint action."
3. **Deeper / structured search for the 15 depth-gap targets** (2-step `rw; apply`
   plans, `ext`+`by_cases`, `refine`-with-holes) — a search-depth experiment, not a
   lemma-synthesis one.
4. **Improve the lemma index** before any future retrieval round: the 160
   `unknown_name` outcomes show many candidates are out of scope at the target's
   source position; a scope-aware (imports-resolved) index would cut that noise.

No SF6. No RC4. No production change.

---

## 12. Protected-file confirmation

- `project/evolve/experiments/rc1/rc1_production_wrapper.json` — **untouched**.
- `project/evolve/experiments/rc2_release/rc2_production_wrapper.json` — **untouched**.
- `project/evolve/routing/ns24_router.json` — **untouched**.
- NS9 genome/checkpoint files, REL1/RC1/RC2 release reports — **untouched**.
- TR1/TR2 training datasets — **untouched** (SF5 wrote a separate additive
  `sf5_training_examples.jsonl`).
- No production routing modified; no RC4 created; no lemma synthesized; README not
  updated. All SF5 artifacts live under `project/evolve/experiments/sf5/` and
  `project/evolve/reports/sf5/`, plus `scripts/sf5_*.py`.

`git diff --stat HEAD` over the three protected wrappers is empty (see §11 verification
in the session log).
