# nat_defs_medium — v4.7 evolution sweep on constructor seed

## v4.6 reference

  - Best v4.6 result: **26 / 38** via the `constructor` template variant.
  - First div-family closure: `Nat.div_lt_iff_lt_mul'`.
  - Commit: `5a697c4`.

## Stage 1 — adopt constructor as default

Changes:
  - `evolve/run_evolve.py`: `--template-variant` default flipped
    `v45` → `constructor`. `make_seed_candidate(template_variant=...)`
    keyword default likewise flipped.
  - `evolve/mutator.py`: three new mutation ops covering retrieval
    knobs and family budgets: `retrieval_top_k`, `reorder_retrieval_forms`,
    `family_budget_delta`.

Sanity:
  - Generation-0 reproduction on nat_defs_medium: **26 / 38** ✓
    (run `evolve-20260522-072211-b7f1fc/eval/seed-baseline`).
  - `proved_by_origin = {fallback_tactic: 18, family_tactic: 4, generative_topk: 4}`,
    identical to v4.6's constructor run.
  - 0 crashes, 0 unknown_constant, retrieval: 262 attempts / 7 advances / 0 wins.

## Stage 2 — evolution sweep (generations=2, population-size=4, survivors=2)

| rank | candidate          | gen | mutations                                                            | proved | prog | score   |
|------|--------------------|-----|----------------------------------------------------------------------|--------|------|---------|
| 1    | **g2-i0-tk8-ms8**  | 2   | timeout=10, **family_budget[div]=12**                                | 26/38  | **6** | **2587.7** |
| 2    | seed-baseline      | 0   | (constructor default)                                                | 26/38  | 5    | 2586.7   |
| 3    | g1-i1-tk8-ms8      | 1   | added Nat template                                                   | 26/38  | 5    | 2586.7   |
| 4    | g1-i3-tk8-ms8      | 1   | max_steps=8 (no-op)                                                  | 26/38  | 5    | 2586.7   |
| 5    | g2-i2-tk8-ms8      | 2   | added Nat template, timeout=10                                       | 26/38  | 5    | 2586.7   |
| 6    | g2-i3-tk8-ms8      | 2   | timeout=20, top_k=8                                                  | 26/38  | 5    | 2586.7   |
| 7    | g1-i2-tk8-ms6      | 1   | **family_budget[div]=6**, max_steps=6, timeout=30                    | 26/38  | 3    | 2585.2   |
| 8    | g2-i1-tk8-ms8      | 2   | **reordered fallbacks** (from g1-i1)                                 | 14/38  | 17   | 1384.4   |
| 9    | g1-i0-tk12-ms8     | 1   | reordered AM_GM family, **reordered fallbacks**, **top_k=12**        | 12/38  | 15   | 1180.9   |

## Stage 3 — analysis

### Did any candidate beat 26/38?

**No.** The medium set plateaus at 26/38. Seven of nine candidates
preserve 26/38; the best (g2-i0-tk8-ms8) edges seed by +1.0 in
score via a higher `progress_count` (6 vs 5), not by closing a new
proof.

### Did any candidate preserve 26/38 with better diagnostics?

  - **g2-i0-tk8-ms8** (best). `family_budget[div]=12` (up from 8)
    gives `Nat.dvd_iff_div_mul_eq` enough budget to advance past
    earlier ERROR states into EXH @ max_steps, producing
    `progress_count=6`. Total retrieval attempts climb from 262 →
    357 but no new closure. Useful for diagnostic depth on the
    stuck div theorems; cost is +~30s wall-clock.
  - **g1-i2-tk8-ms6** (cheapest preservation). `family_budget[div]=6`
    and `max_steps=6` cut retrieval attempts from 262 → 132 and
    eval wall-clock from ~5min to ~3min while preserving 26/38.
    Progress drops to 3 (less exploration), score 2585.2. This is
    the fast-preview configuration for future iterations.

### Did any candidate solve Nat.div_pos / Nat.div_pos_iff / Nat.dvd_iff_div_mul_eq?

**No.** All three theorems are ERR (out of candidates at step 3-5)
across every candidate, except for `Nat.dvd_iff_div_mul_eq` under
g2-i0 which moves to EXH @ 8 — more steps consumed but still no
proof. This confirms the v4.6 conclusion that these three theorems
are not template-tractable on the current gen_v5 checkpoint.

### Did any candidate lose `Nat.div_lt_iff_lt_mul'`?

**No.** The closure is *robust* under every mutation in the sweep,
including the two that regressed badly. The retrieval-then-simp_all
path is preserved as long as the div family stays small and doesn't
re-introduce a derailing `rw [Nat.div_eq_of_lt]` template.

### Regressions

Two candidates regressed (`reorder_fallback` is the common cause):

  - **g1-i0-tk12-ms8 (12/38)** — top_k bumped to 12 *and* fallback
    shuffled. The mod-specific simp tactics floated to the head of
    the fallback list; theorems normally closed by `omega` (which
    is now at position 3+) got starved out under the per-state
    budget cap with the larger generative top-k. Pure top-k mutation
    without fallback shuffle would not have caused this.
  - **g2-i1-tk8-ms8 (14/38)** — fallback shuffle alone from g1-i1's
    parent. First three fallbacks become `simp [... add_mod, mod_eq_of_lt]`,
    `simp [add_mod, mod_eq_of_lt] at *`, `rw [Nat.add_comm]` — wrong
    priorities for add-shape theorems.

The lesson: **`reorder_fallback` is destructive when the parent's
fallback order encodes meaningful priority**. A future mutator
should either lock the first few fallback entries or use a smarter
permutation (e.g. only shuffle within compatible buckets).

### Retrieval still useful?

Yes. Across all 26/38 candidates the closure of `Nat.div_lt_iff_lt_mul'`
goes through `rw [Nat.div_lt_iff_lt_mul]` (retrieved_premise) at
step 1, regardless of fallback ordering, retrieval_top_k, or
family budget. Retrieval is now a structural ingredient of the
seed proof set, not just diagnostic.

### Crashes / unknown constants

  - DojoCrashError: **0** across all 9 candidates.
  - unknown_constant: **0** (template verifier holds the line).

## Per-candidate div/dvd status

| theorem                       | seed | g1-i2 (fast) | g2-i0 (best) | g1-i0 / g2-i1 (regressions) |
|-------------------------------|------|---------|---------|---------|
| Nat.div_le_div_right          | EXH  | ERR     | EXH     | EXH     |
| **Nat.div_lt_iff_lt_mul'**    | **PROVED** | **PROVED** | **PROVED** | **PROVED** |
| Nat.div_lt_one_iff            | EXH  | ERR     | EXH     | EXH     |
| Nat.div_pos                   | ERR  | ERR     | ERR     | ERR     |
| Nat.div_pos_iff               | ERR  | ERR     | ERR     | ERR     |
| Nat.dvd_iff_div_mul_eq        | ERR  | ERR     | **EXH** | ERR     |

g2-i0's `family_budget[div]=12` is the only knob that moves
`Nat.dvd_iff_div_mul_eq` past the ERR cliff — but only to EXH, not
to closure.

## Artifacts

  - `project/evolve/runs/evolve-20260522-072211-b7f1fc/`
    - `eval/seed-baseline/`         — 26/38 (constructor default reproduction)
    - `eval/g1-i0-tk12-ms8/`        — 12/38 regression (top_k + shuffle)
    - `eval/g1-i1-tk8-ms8/`         — 26/38 (template added, no impact)
    - `eval/g1-i2-tk8-ms6/`         — 26/38 (faster diagnostic, smaller budgets)
    - `eval/g1-i3-tk8-ms8/`         — 26/38 (no-op mutation)
    - `eval/g2-i0-tk8-ms8/`         — **26/38 best score, family_budget[div]=12**
    - `eval/g2-i1-tk8-ms8/`         — 14/38 regression (shuffle alone)
    - `eval/g2-i2-tk8-ms8/`         — 26/38 (template added, no impact)
    - `eval/g2-i3-tk8-ms8/`         — 26/38 (no-op mutation)
    - `best_candidate.json`, `summary.json`

## Recommendation

The medium-set plateau at 26/38 is stable. Mutator can't beat it
because the three stuck div theorems (`Nat.div_pos`, `Nat.div_pos_iff`,
`Nat.dvd_iff_div_mul_eq`) require proof machinery the generative
checkpoint + tactic search cannot synthesize:

  - `Nat.div_pos` requires a `Nat.lt_of_lt_of_le` chain (`0 < 1 ≤
    a/b`) that needs the right intermediate term.
  - `Nat.div_pos_iff` is an iff whose forward direction needs the
    above chain and whose reverse direction needs case analysis on
    `b = 0`.
  - `Nat.dvd_iff_div_mul_eq` requires either `Nat.div_add_mod`
    arithmetic or term-mode `⟨_, _⟩` construction.

**v4.8 direction: term-mode `exact ⟨_, _⟩` builder for the iff and
dvd shapes.** Specifically, an emitter that on goals of shape `A ↔
B` synthesizes `Iff.intro (fun h => …) (fun h => …)` with a
template-driven inner proof, and on goals of shape `a ∣ b`
synthesizes `⟨quot, proof_of_eq⟩`. This is a proper next research
direction — tactic-search permutations have been exhausted.

The constructor seed and verifier should remain the v4.7 default.
The mutator regression on `reorder_fallback` is a known issue; either
remove that op or restrict it to compatible buckets in v4.8.
