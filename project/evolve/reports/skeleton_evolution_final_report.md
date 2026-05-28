# Skeleton evolution — final report (v3 → NS9)

A consolidated record of the AlphaEvolve-style outer loop that
evolved the `gen_v5` proof-search wrapper, without any retraining,
from 3-theorem coverage to a 17-skeleton compact genome proving
**37/38** on `nat_defs_medium` and **49/65** on
`nat_defs_large_v5`.

## TL;DR

- **No retraining**; the same `gen_v5` t5-small checkpoint throughout.
- **Lean is the only evaluator.** Every reported number is a real
  rollout through `eval_rollout_all.py` against Mathlib via Dojo.
- **Final result**: 37/38 medium · 49/65 large at **17 enabled
  skeletons** (compressed from 48 raw — **65% smaller**).
- Single residual failure (`Nat.AM_GM`) is a *model-capability*
  ceiling; the skeleton bag itself is exhausted.

## Timeline and proof-count progression

`nat_defs_medium` (38 theorems) — the small medium set used as the
primary score:

| stage | enabled skeletons | medium proved | large proved | notes |
|---|---:|---:|---:|---|
| `gen_v5` baseline (no wrapper) | 0 | 3/38 | n/a | model alone |
| v3 hybrid_evolved (fallbacks + templates) | n/a | 18/38 | n/a | first wrapper |
| v3.2 + per-state budget | n/a | 22/38 | n/a | extras cap |
| v3.4 + family tactics | n/a | 24/38 | n/a | mod/div/AM_GM |
| v3.5 library cleanup + medium scale-out | n/a | 25/38 | n/a | first medium baseline |
| v3.6 + per-theorem deny-list | n/a | 25/38 | n/a | crash-handling |
| v4.1 premise retrieval (rw form) | n/a | 28/38 | 33/65 | retriever in wrapper |
| v4.2 retrieval forms + filters | n/a | 30/38 | 37/65 | rw/simp/apply/exact |
| v4.3 bloat-rejector + apply-only skip | n/a | 32/38 | 41/65 | rank-list hygiene |
| v4.4 shape-aware retrieval | n/a | 33/38 | 43/65 | iff/eq/lt/le shape gating |
| v4.5 hypothesis-shape template params | n/a | 34/38 | 45/65 | `{hyp_*}` substitution |
| v4.6 priority_templates discovery | n/a | 36/38 | 47/65 | shape-slot dispatch |
| v4.7 lemma audit jump | n/a | **37/38** | 49/65 | dropping unprovable defs |
| NS1 specificity ordering | n/a | 37/38 | 49/65 | (priority, specificity) sort |
| NS3 lemma audit | n/a | 37/38 | 49/65 | per-theorem failure triage |
| NS3.5 any-as-fallback semantics | n/a | 37/38 | 49/65 | wrapper-side fix |
| NS4 skeleton-bag refactor | 48 | 37/38 | 49/65 | unified representation |
| NS5 archive + evolutionary mutation | 25 | 37/38 | 49/65 | first compaction sweep |
| NS6 assist credit + scoped mutation | 20 | 37/38 | 49/65 | rank-coupling diagnosis |
| NS7 stable IDs + bag-only pre-flight | 20 | 37/38 | 49/65 | first pre-flight detector |
| NS8 cached-ranked-list simulator | 20 | 37/38 | 49/65 | full pre-flight, 0 Lean regressions |
| **NS9 retrieval-gate decoupling** | **17** | **37/38** | **49/65** | **broke the 20-skel floor** |

(`enabled skeletons` is only meaningful from NS4 onward — earlier
configs used flat fallback/template lists.)

## Architectural discoveries

### v3 — `hybrid_evolved` is the right shape

Wrapping `gen_v5` with a deterministic, evolved layer of fallback
tactics and theorem-name-aware family tactics lifted medium from
3/38 to 25/38 with zero retraining. The lesson: a small evolved
program around a fixed model dominates beam-search alone.

### v4.6 — `priority_templates` (shape-slot dispatch)

The biggest single architectural lever before NS9. Templates were
moved out of `tactic_templates` into a shape-slotted dict (`iff`,
`eq`, `lt`, `le`, `dvd`, `and`, `or`, `unknown`, `any`) and emitted
*before* the model's beam-search output. Goal-shape classification
fired the right slot deterministically. The medium jump from 33 →
36 came almost entirely from `pt_iff_8` (`exact ⟨fun h => by omega,
fun h => by omega⟩`) closing 11 iff theorems at step 1.

### v4.7 — lemma audit

Two of the four remaining medium failures (`Nat.AM_GM`,
`Nat.dvd_iff_div_mul_eq`) had skeleton attempts that triggered Dojo
crashes from unknown lemmas in the genome (`Nat.AM_GM` itself,
`Nat.le_div_iff`). Dropping unprovable lemma references from the
genome unblocked `Nat.dvd_iff_div_mul_eq` and surfaced the model
ceiling on `Nat.AM_GM`.

### NS3.5 — `any` as fallback, not exclusive alternative

The wrapper's shape gate was discovered to be *exclusive*: once a
genome configured a slot for shape S, goals of shape S could never
reach `any`. The fix at the wrapper level (NS3.5) restored "shape
first, then `any` as true fallback" semantics, removing the need
for genome authors to manually mirror `any` templates into every
configured shape.

### NS4 — the skeleton bag

The five legacy emit paths (`priority_template`, `family_tactic`,
`term_builder`, `fallback_tactic`, `tactic_template`) were unified
into a single `SkeletonBag` indexed by shape. Each `Skeleton` carries
`(name, shape, template, origin, family, priority, specificity,
enabled)`. The wrapper delegates emit to the bag through
`emit_priority_tactics`, `emit_family_tactics`, etc. NS4.1 added
specificity-sort within family blocks (the NS1 invariant); NS4.2
modeled premise-retrieval emissions as *dynamic* per-state
skeletons synthesized on the fly.

The bag isn't just a data refactor — it's the prerequisite for
treating the strategy as an evolvable genome.

### NS5 — archive + evolutionary mutation

A JSONL ledger at `project/evolve/archive/skeletons.jsonl` records
per-skeleton wins/advances/attempts across runs. Six safe,
archive-guided operators (`disable_dead`, `promote_high_win`,
`clone_to_shape`, `archive_seed`, `budget_trim`, `demote_generic`)
mutate the bag; a no-regression gate accepts only candidates that
still prove ≥ best_proved_medium. The 7.5-hour overnight sweep
compressed the genome from 48 → 25 skeletons preserving 37/49.

The sweep also surfaced two design defects:

  - **Zero-win ≠ useless.** 60+ rejected mutations removed a
    never-winning skeleton that turned out to *advance state* into
    a form a later tactic closes. The archive only tracked `wins`;
    it needed per-step `advances` to prune safely.
  - **Order-changing operators clobbered unrelated bands.** Bag-wide
    resorts shuffled fallback/tactic_template entries that the
    wrapper iterates in *bag order*; that broke
    `Nat.two_mul_ne_two_mul_add_one` and `Nat.add_mod_eq_add_mod_right`.

### NS6 — assist-credit + scoped mutation

`scripts/ns6_assist_credit.py` walks per-step traces and credits
each skeleton with `direct_wins`, `advances`, and `assist_wins_kN`
(skeleton advanced state, and a different tactic closed within K
accepted steps). `disable_dead_skeleton` now requires
`direct_wins=0 AND advances=0 AND assist_wins_k3=0`. Order-changing
operators were scoped by `(origin, shape, family)` — reorders only
within one bucket. Sweep compressed from 25 → 20, zero reorder
regressions.

Two new symptoms surfaced:

  - **Skeleton names drift** across mutations because
    `from_legacy_strategy_config` re-indexes by insertion order.
  - **Rank-coupling regressions exist.** Even with perfect direct/
    assist credit, removing an uncredited skeleton can shift the
    wrapper's top-K cutoff so a correctly-protected skeleton
    drops out. 6/20 NS6 candidates failed this way.

### NS7 — stable IDs + bag-only pre-flight detector

`Skeleton.stable_id` = sha1 of `(origin, shape, family, specificity,
normalized_template)`, invariant across re-indexing.
`evolve/rank_coupling.py::check_rank_coupling` compares the bag's
deterministic skeleton-emit order between baseline and mutated,
flagging any protected skeleton that drops or moves backward.

NS7 caught 3 of the 21 cycles pre-flight, saving ~7 minutes of Lean
time, but missed the deeper rank-coupling effect because it
operated on bag-only ordering — it couldn't see how skeleton
disables affect the wrapper's merged top-K relative to model
outputs.

### NS8 — full ranked-list simulator (the big diagnostic)

`evolve/rank_simulator.py` instantiates the *real* `StrategyWrapperPolicy`
with a `CachedBasePolicy` that returns cached `gen_v5` outputs for
each protected state. The wrapper's existing merge runs unchanged
— same dedup, same cap, same priority/base/extra ordering — so the
simulated ranked list is byte-equivalent to what the live eval
produces. `check_state_coupling` simulates both genomes per
protected state and flags the mutation when the critical tactic
drops out.

NS8 caught **all 10** NS7 Lean-rejected cycles pre-flight (~25 min
saved per sweep) and pinned the 20-skeleton floor to a single
mechanism: disabling `fam_div_14` (the only family_tactic for the
`div` family) made `activated_families` empty, which gated the
wrapper's retrieval block off entirely. The critical
`retrieved:Nat.div_lt_iff_lt_mul:rw` skeleton then disappeared as
a side-effect of pruning a different (zero-credit) skeleton.

### NS9 — retrieval-gate decoupling

Two new wrapper fields: `retrieval_requires_family: bool` and
`retrieval_family_gates: list[str]`. With
`retrieval_requires_family=False`, the retrieval block fires
whenever `full_name` contains a substring in
`retrieval_family_gates`, independent of family_tactic survival.
The change is wrapper-only — the bag, retriever, and simulator
need no architectural updates.

Under the NS9 gate, three previously-rank-coupled prunes became
safe (`fb_19`, `fam_div_14`, `pt_iff_2`). 3 promotions in 20
cycles, 6 pre-flight rejections, **0 Lean rejections**. Best at
cycle 3: **17 enabled skeletons preserving 37/49**.

## Skeleton-count progression

```
baseline (NS3-combined)  48 ████████████████████████
NS5 best                 25 ████████████
NS6 best                 20 ██████████
NS7 best                 20 ██████████
NS8 best                 20 ██████████
NS9 best                 17 ████████          (35% of original)
```

## Operator catalog (final)

Defined in `evolve/skeleton_mutator.py`:

| operator | scope | safety |
|---|---|---|
| `disable_dead_skeleton`        | global | credit-aware: requires zero direct/advance/assist + ≥N attempts |
| `promote_high_win_skeleton`    | per (origin, shape, family) | NS6 scoped — only reorders within scope |
| `demote_generic_skeleton`      | per (origin, shape) | NS6 scoped — re-applies NS1 sort within scope |
| `clone_skeleton_to_shape`      | per shape pair | iff↔any, eq↔iff, lt↔le clone graph |
| `budget_trim`                  | global | reduces only when many dead |
| `archive_seed`                 | global | NS5 wins-only compact-genome (deprecated) |
| `archive_seed_credit`          | global | NS7 credit-aware compact-genome |
| `narrow_family_gate`           | (stub) | awaits per-attempt family-failure logging |
| `expand_family_gate`           | (stub) | awaits per-theorem family-shadow analysis |

Safeguards in order of precedence:

1. **Pre-flight rank-simulator** (NS8) — rejects pre-Lean if any
   protected critical tactic drops from the simulated ranked list.
2. **No-regression gate** — accepts only if
   `proved_medium >= best_proved_medium`.
3. **Strict-compact / strict-improve promotion** — promotes only
   if proved improves OR enabled-count strictly decreases.

## Best genome (NS9, cycle 3)

`project/evolve/best/ns9_best_genome.json` — pointer to the canonical
artifact, plus a per-field summary:

  - 17 enabled skeletons:
      * 12 priority_templates: `pt_iff_{0..7}`, `pt_any_{9,10}`,
        `pt_eq_11`, `pt_le_12`, `pt_lt_8`
      * 3 family_tactic: `fam_mod_{13,14,15}` (no `fam_div_*` —
        pruned)
      * 1 fallback_tactic: `fb_16`
      * 1 retrieval gate (dynamic): `retrieval_family_gates=["div",
        "mod", "pow"]`
  - `retrieval_requires_family=False`
  - `use_skeleton_bag=True`
  - `top_k=8`, `max_steps=8`

Reproducing the best result:

```bash
python eval_rollout_all.py \
  --theorem-set nat_defs_medium \
  --policy-type hybrid_evolved \
  --ckpt-dir project/models/gen_v5 \
  --top-k 8 --max-steps 8 \
  --strategy-config project/evolve/best/ns9_best_genome.json \
  --out-dir /tmp/ns9_repro_medium

python eval_rollout_all.py \
  --theorem-set nat_defs_large_v5 \
  --policy-type hybrid_evolved \
  --ckpt-dir project/models/gen_v5 \
  --top-k 8 --max-steps 8 \
  --strategy-config project/evolve/best/ns9_best_genome.json \
  --out-dir /tmp/ns9_repro_large
```

Expected: 37/38 medium (~2.5 min), 49/65 large (~5 min).

## Remaining limitations

1. **`Nat.AM_GM` is the residual medium failure.** The skeleton bag
   exhausts; the theorem requires a multi-step inductive argument
   the `gen_v5` model + retrieval cannot synthesize. Closing 38/38
   is a model-capability task (NS10: targeted fine-tune).

2. **`nat_defs_large_v5` plateau at 49/65.** 16 unproved theorems
   span structural patterns the current genome doesn't address.
   Same story — likely needs model retraining or new bag operators
   that synthesize multi-step skeletons.

3. **Stable-id-keyed archive.** The archive still indexes by
   skeleton `name`. NS9's stable_id lives only in traces and the
   rank simulator. Migrating the archive to stable_id key would
   make cross-run credit aggregation correct under arbitrary
   mutation order.

4. **`narrow_family_gate` / `expand_family_gate` are stubs.** NS6's
   per-step trace data is rich enough to implement these — the
   operators are designed but not yet wired.

5. **Cross-run credit ledger.** The credit index is rebuilt per
   sweep from baseline traces. A persistent credit JSONL alongside
   `skeletons.jsonl` would let multi-run aggregation see assist
   signal too.

## What works particularly well

  - **Lean as the only evaluator.** No proxy metrics; every accept
    is a real Mathlib rollout. Eliminates an entire class of
    over-fitting risk.
  - **`SkeletonBag` as the genome representation.** Stable across
    refactors; every operator can be expressed as a small bag
    edit; the wrapper consumes the bag through `emit_*` methods so
    bag-side changes don't ripple through the wrapper.
  - **Per-step trace attribution.** Skeleton-level
    direct/advance/assist credit is *the* signal that unlocked NS6
    and NS8. Without per-step traces neither would have been
    possible.
  - **Pre-flight rank simulation** (NS8) — turning Lean rejections
    into pre-flight rejections saved ~25 min per 20-cycle sweep.
    The same technique should generalize to any mutation framework
    where the evaluator is expensive.

## Roadmap

- **Stop skeleton-evolution feature work for now.** The bag is
  exhausted at 17 enabled / 37/38 medium.
- **Next attack vector**: `Nat.AM_GM`-class failures. Either:
  (a) NS10 — gen_v5+1 targeted fine-tune on inductive arguments
      using the existing trace corpus as training signal.
  (b) Multi-step skeleton synthesis (an operator that emits
      `(rw, rw, simp_all)` triples derived from successful retrieval
      chains in the archive).
- **Operational**: stable-id-keyed archive + cross-run credit
  ledger so the next iteration can pick up where NS9 left off.

## Cross-reference index

- `nat_defs_medium_summary.md` — the chronological progression log.
- `ns1_specificity_ordering.md` — NS1 specificity sort invariant.
- `ns3_lemma_audit.md` + `ns3_lemma_audit_results.md` — per-theorem
  failure triage.
- `ns3_5_any_fallback_semantics.md` — wrapper-side `any` fallback fix.
- `ns4_skeleton_bag_design_note.md` + `ns4_1_skeleton_unification.md`
  + `ns4_2_retrieved_dynamic_skeletons.md` + `ns4_skeleton_bag_repro.md`
  — the bag refactor.
- `ns5_skeleton_evolution_plan.md` + `ns5_skeleton_evolution_report.md`
  — first compaction sweep.
- `ns6_assist_credit_analysis.md` + `ns6_credit_aware_mutation.md`
  — credit-aware pruning.
- `ns7_rank_stable_evolution.md` — stable IDs + bag-only detector.
- `ns8_rank_simulation_preflight.md` — cached-model rank simulator.
- `ns9_retrieval_gate_decoupling.md` — gate split that broke the floor.
- `v5_alphaevolve_architecture.md` — AlphaEvolve framing.
- `v5_central_findings.md` — earlier highlights.
- `v5_failure_deep_dive_add_mod_eq_ite.md` — failure-case methodology.
- `v5_priority_templates_insight.md` — priority_templates discovery
  note.

## Commit history

| commit | summary |
|---|---|
| `aed695c` | Decouple retrieved premise activation from family tactic survival (NS9) |
| `04b38bb` | Add cached rank simulation for skeleton mutation preflight (NS8) |
| `709ec70` | Add rank-stable skeleton mutation safeguards (NS7) |
| `2b6044b` | Add assist-credit archive and scoped skeleton mutation (NS6) |
| `9e546f0` | Add skeleton archive and evolutionary mutation runner (NS5) |
| `4a61ea1` | Represent retrieved premises as dynamic skeleton emissions (NS4.2) |
| `28a3b0c` | Route family and fallback emissions through skeleton bag (NS4.1) |
| `88a739b` | Prototype skeleton-bag adapter for proof-search genomes (NS4) |
