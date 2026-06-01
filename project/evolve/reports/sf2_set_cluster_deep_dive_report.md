# SF2 Set Cluster Deep Dive + SF3 Candidate-Lemma Triage

Task: **SF2 Set Cluster Deep Dive + SF3 Candidate Lemma Triage**
Branch: `rc1-production-stack` · Baseline: REL1 / RC1 (commit `72c1250`)
Run type: live LeanDojo exploratory. **No commit. No production-config change.**

---

## 1. Executive summary

We took the largest high-priority **Set** clusters from the 18 genuine RC1 frontier
failures, extracted each theorem's official Mathlib proof, and ran **live LeanDojo
probe ladders** (Families A–G) against 12 representative failures spanning 6
clusters.

- **12 theorems probed live, 12 opened (100% live), 10 closed by some probe.**
- Gap classification: **5 `tactic_gap`, 5 `search_depth_gap`, 2 unsolved
  (`needs_deeper_search`)** at the theorem level; cluster rollup = 2 `tactic_gap`,
  2 `search_depth_gap`, 2 `mixed`.
- Best-performing probe families: **`E_ite_bycases` (`simp [Set.ite]`)** and
  **`F_source_inspired`** (rw-bridges + ext/by_cases).
- **SF3 candidate-lemma queue size: 0.** Every Set failure is an automation gap
  (tactic / routing / search-depth) over **existing** Mathlib lemmas — *not* a
  missing lemma. This is the same verdict as the Multiset singleton negative result.
- **One generalizing probe family found:** `simp [Set.ite]` closes **2** distinct
  failures (`Set.ite_right`, `Set.ite_empty_right`) where all four baselines
  (`simp`, `simp_all`, `aesop`, `classical<;>aesop`) fail. This is the only
  honest candidate for an off-by-default experimental Set wrapper.

**Recommendation:** Do **not** modify RC1. The defensible next step is an
off-by-default **SET2** experimental probe family `simp [Set.ite]` gated to
`Set.ite`-shaped goals (NS19-style narrow gate), validated against the promotion
gate (positive Δ, 0 regressions, 0 off-gate, NS23 minimal-sufficient attribution,
deterministic reproduction). The remaining wins are theorem-specific simp-lemma
sets or rw-bridges that do **not** generalize and belong to RC1 search-depth, not
a new tactic.

---

## 2. Background

- **SF1 truth layer repaired.** The authoritative per-theorem solved key is
  `finished` (not the nonexistent `proof_finished`/`solved`). RC1 Multiset holdout
  is really **2/3**, not 0/3.
- **Multiset singleton was a negative result.** `Multiset.toFinset_eq_singleton_iff`
  fails because its official proof is count-extensionality over existing lemmas —
  a tactic/search gap, **not** a missing lemma. A 13-probe ladder confirmed it.
- **Why Set clusters now.** Frontier expansion left 18 genuine RC1 failures in 10
  clusters; 8 are high-priority and dominated by the **Set** namespace (iff,
  ite membership/equality, subset/diff/union). These are the largest remaining
  honest target for failure-driven discovery.

---

## 3. Selected Set failures

12 selected (cap ≤12; 3 deferred as duplicates/over-specialised — recorded in
`selected_cases.json`). All from `Mathlib/Data/Set/Basic.lean`, all `all_tactics_errored`.

| theorem | cluster (shape) | goal shape | RC1 failure | selected reason |
|---|---|---|---|---|
| `Set.ssubset_singleton_iff` | future/iff | iff | top-11 err @3 | largest iff cluster rep |
| `Set.antitoneOn_iff_antitone` | future/iff | iff | top-14 err @1 | iff cluster rep |
| `Set.ite_eq_of_subset_left` | broad_aesop/eq | equality | top-12 err @1 | ite equality rep |
| `Set.subset_singleton_iff_eq` | broad_aesop/eq | equality | top-11 err @2 | subset/singleton rep |
| `Set.ite_inter_self` | rc1/membership | equality (∩) | top-11 err @3 | ite-inter rep |
| `Set.ite_inter` | rc1/membership | equality (∩) | top-11 err @3 | ite-inter rep |
| `Set.subset_insert_iff` | rc1/equality | iff | top-11 err @3 | subset/insert rep |
| `Set.diff_singleton_subset_iff` | rc1/equality | iff | top-11 err @5 | diff/subset rep |
| `Set.union_empty_iff` | rc1/equality | iff | top-11 err @4 | union rep |
| `Set.ite_right` | future/membership | equality (∩) | top-11 err @3 | ite rep |
| `Set.ite_empty_right` | future/membership | equality (∩) | top-11 err @3 | ite rep |
| `Set.pair_eq_pair_iff` | future/equality | iff | top-11 err @2 | pair-eq rep |

Deferred: `Set.ite_eq_of_subset_right` (dup of `_left`), `Set.ite_inter_compl_self`
(subsumed by `ite_inter_self`), `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le`
(over-specialised giant simp).

---

## 4. Source-proof analysis

Proof-style histogram over the 12 official proofs:
`{rw_bridge: 5, simp_only: 5, by_cases_ite_split: 1, subset_antisymm: 1}`.

Recurring motifs:
- **`simp [Set.ite]` def-unfold** (`ite_right`, `ite_empty_right`): one-liner; RC1's
  generic `simp`/`aesop` does **not** unfold the irreducible `Set.ite`.
- **rw-bridge over named existing lemmas** (`ite_inter`, `ite_inter_self`,
  `diff_singleton_subset_iff`, `ssubset_singleton_iff`): specific rewrite chains, no
  new lemma, but beyond RC1 search depth.
- **`ext` + `by_cases` on `x ∈ t`** then `simp` with hypotheses (`ite_eq_of_subset_left`).
- **`simp [defn]` unfolding** (`antitoneOn_iff_antitone` → `simp [Antitone, AntitoneOn]`).
- **`subset_antisymm`/`simp [...]; aesop`** (`pair_eq_pair_iff`).
- **obtain/case split with per-branch hypotheses** (`subset_singleton_iff_eq`).

Nearby reusable lemmas confirmed present (so no bridge is missing): `Set.ite`,
`ite_inter_inter`, `ite_same`, `ite_compl`, `diff_subset_iff`, `union_singleton`,
`subset_empty_iff`, `union_subset_iff`, `subset_singleton_iff_eq`, `singleton_ne_empty`.

---

## 5. Probe ladder results

Families tested: A baselines, B ext-equality, C iff-decomp, D subset/diff/union,
E ite/by_cases, F source-inspired (per-theorem), G negative controls (off by default;
not run). Probes ordered cheap-first; first genuine close + minimality battery recorded.

- **Successes: 10/12.** Representative winning probes:
  - `simp [Set.ite]` → `Set.ite_right`, `Set.ite_empty_right` (**generalizes, n=2**)
  - `simp [Antitone, AntitoneOn]` → `Set.antitoneOn_iff_antitone`
  - `simp only [← subset_empty_iff, union_subset_iff]` → `Set.union_empty_iff`
  - `simp [subset_antisymm_iff, insert_subset_iff] <;> aesop` → `Set.pair_eq_pair_iff`
  - `rw [ite_inter_inter, ite_same]` → `Set.ite_inter`
  - `rw [Set.ite, union_inter_distrib_right, diff_inter_self, inter_assoc, inter_self, union_empty]` → `Set.ite_inter_self`
  - `rw [← union_singleton, union_comm] <;> apply diff_subset_iff` → `Set.diff_singleton_subset_iff`
  - `rw [ssubset_iff_subset_ne, …] <;> exact fun h => h ▸ (singleton_ne_empty _).symm` → `Set.ssubset_singleton_iff`
  - `ext x <;> by_cases hx : x ∈ t <;> simp [hx, Set.ite, or_iff_right_of_imp (@h x)]` → `Set.ite_eq_of_subset_left`
- **Minimality:** for every win, all four baseline probes (`simp`, `simp_all`,
  `aesop`, `classical<;>aesop`) **failed** → the win is genuinely non-baseline.
  Each win is `minimality_status: unconfirmed`, `requires_ns23_relabel: true`.
- **Unsolved (2):** `Set.subset_insert_iff`, `Set.subset_singleton_iff_eq`. Both have
  existing-lemma official proofs that branch on `by_cases`/`obtain` and carry a
  **per-branch hypothesis** — the official proofs use `·` bullets, which
  `run_transition` rejects, and the single-line `<;>` form applies the closer to
  *both* branches (e.g. `unknown identifier 'hs'`). This is a search-depth /
  parser-expressibility gap, **not** a missing lemma.

Parse / capability issues encountered (honest):
- **`classical <;> aesop` and `classical <;> simp [...]` parse-error** in
  `run_transition` (`expected '{' or tactic` at the `<;>`). The `classical`
  prefix cannot be `<;>`-chained; that baseline is effectively unavailable. It
  never spuriously solved, so classification is unaffected.
- `simp [Set.ite]` raises `unknown constant 'Set.ite'` on non-ite goals
  (`subset_insert_iff`, `subset_singleton_iff_eq`) — correct no-op noise.
- `ext x` raises `applyExtTheorem only applies to equations` on iff/subset goals
  — Family B correctly gated, errors are diagnostic only.

---

## 6. Cluster-level interpretation

| cluster | label | size | sel | solved | best family | gap type | next action |
|---|---|---|---|---|---|---|---|
| future/iff | future_failure_driven | 4 | 2 | 2 | F_source_inspired | **mixed** | new_probe_family |
| future/membership | future_failure_driven | 2 | 2 | 2 | E_ite_bycases | **tactic_gap** | new_probe_family |
| rc1/equality | rc1_production_stack | 3 | 3 | 2 | F_source_inspired | **mixed** | new_probe_family |
| rc1/membership | rc1_production_stack | 3 | 2 | 2 | F_source_inspired | **search_depth_gap** | new_probe_family |
| broad_aesop/equality | broad_set_aesop_rejected | 3 | 2 | 1 | F_source_inspired | **search_depth_gap** | new_probe_family |
| future/equality | future_failure_driven | 1 | 1 | 1 | F_source_inspired | **tactic_gap** | new_probe_family |

Reading: the **ite-membership** clusters split cleanly — simple def-unfold cases
are a **tactic gap** (`simp [Set.ite]`), compositional ite cases are a
**search-depth gap** (rw-bridges). The iff / subset / union clusters are
**mixed/search-depth**: a couple yield to short `simp [lemmas]`, the rest need
multi-step rw or branch-carrying proofs over existing lemmas.

---

## 7. Candidate-lemma triage

**0 candidate lemmas. 12 rejected as not-missing-lemma.** (`set_candidate_lemmas.json`,
`set_candidate_lemma_queue.jsonl` — empty queue, `set_candidate_lemma_attempts.lean`
— note only, no Lean attempted because nothing qualified.)

- **Not missing lemmas (10):** closed live by an existing tactic/probe.
- **Not missing lemmas (2 unsolved):** official proofs exist over **existing** named
  lemmas (`diff_singleton_subset_iff`, `subset_singleton_iff_eq` building blocks);
  the obstacle is proof-structure depth/parser expressibility, not absent lemmas.

Conservatism rule applied (mirrors Multiset singleton): a failure is a missing-lemma
candidate only if (a) unproven by all probes, **and** (b) its official proof is not
an existing-lemma rw-bridge. Nothing met both.

Potential Set lemma template families considered (ite-membership bridge, set-eq
under ite, subset/diff/union bridge, iff/ext bridge) — **all already exist** in
Mathlib (`Set.mem_ite`, `Set.ite`, `diff_subset_iff`, `Set.ext_iff`, etc.), so no
novel template is proposed.

---

## 8. Relabel recommendations (`relabel_queue_set.jsonl`)

- **NS23 minimal-sufficient relabel candidates (high prio, generalizing):**
  `Set.ite_right`, `Set.ite_empty_right` — both via `simp [Set.ite]`
  (`candidate_family_generalizes: true`, n=2).
- **NS23 relabel candidates (theorem-specific simp/aesop):** `Set.antitoneOn_iff_antitone`,
  `Set.union_empty_iff`, `Set.pair_eq_pair_iff` — short but theorem-specific lemma sets;
  do not generalize.
- **Search-depth only (no new tactic):** `Set.ite_inter`, `Set.ite_inter_self`,
  `Set.diff_singleton_subset_iff`, `Set.ssubset_singleton_iff`,
  `Set.ite_eq_of_subset_left` — rw-bridges over existing lemmas.
- **Deeper search / parser-limited (good traces, queued):** `Set.subset_insert_iff`,
  `Set.subset_singleton_iff_eq`.
- **Rejected as source-copy / theorem-specific:** all rw-bridge winners — reproducing
  the official rewrite chain is not a reusable production tactic.

Every entry: `requires_ns23_relabel: true`, `do_not_promote_yet: true`,
`minimality_status: unconfirmed`.

---

## 9. Production recommendation

**Do not modify RC1.** No item is promotion-ready. Before any Set probe family
enters production it must pass: positive Δ over RC1, **zero regressions**, **zero
off-gate emissions**, **NS23 minimal-sufficient attribution**, **deterministic
reproduction**.

Likely next step — **build an off-by-default SET2 experimental wrapper** around the
single generalizing family `simp [Set.ite]`, narrowly gated to `Set.ite`-shaped
goals (NS19/MX2 narrow-gate pattern), and run it as a controlled A/B against the
RC1 frontier set. Expected upside is small and bounded (the 2–3 def-unfold ite
failures); the compositional ite/iff/subset failures need RC1 best-first search to
go deeper, not a new tactic. If SET2 does not clear the gate with 0 off-gate
emissions, continue SF2 clustering rather than promoting.

This run reinforces the standing lesson: **post-NS9 headroom on strong-base-policy
namespaces (Set) is captured by cheap namespace-gated battery tactics, not by new
lemmas** — and most of it is already inside RC1's reach if search went one step deeper.

---

## 10. Protected-file confirmation

```
$ git diff --stat HEAD -- \
    project/evolve/experiments/rc1/rc1_production_wrapper.json \
    project/evolve/routing/ns24_router.json
(no output — both untouched)
```

`git status --short`: only new `??` SF1/SF2/SF3 artifacts + scripts and the
pre-existing ` M README.md`. **No protected file modified. No NS9/REL1 artifact
touched. No commit made. All changes left in the working tree.**

### Files added this run
- `scripts/sf2_select_set_clusters.py`, `scripts/sf2_extract_set_source_context.py`,
  `scripts/sf2_run_set_probe_ladders.py`, `scripts/sf2_analyze_set_clusters.py`,
  `scripts/sf3_triage_candidate_lemmas.py`
- `project/evolve/experiments/sf2/set_probe_ladders.json`
- `project/evolve/experiments/sf2/out/set_cluster_deep_dive/` (selected_cases,
  source_context, probe_results, cluster_analysis, relabel_queue_set — json/md/jsonl)
- `project/evolve/experiments/sf3/out/set_candidate_lemmas.json`,
  `set_candidate_lemma_queue.jsonl`, `set_candidate_lemma_triage.md`,
  `set_candidate_lemma_attempts.lean`
- `project/evolve/reports/sf2_set_cluster_deep_dive_report.md` (this file)
