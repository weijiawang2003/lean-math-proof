# Direction F — proposed architecture for deeper AlphaEvolve-style evolution

This is the long-arc proposal. It is not what tonight's autonomous loop
implements — it is what tonight's loop is the prototype of. Tonight we
discovered that the v3 → v4 genome (six string lists + integer knobs)
sits on the wrong axis: it mutates tactic order, not proof structure.

The architecture below replaces the flat-string genome with a layered
program representation, two-tier mutation, and an archive that
accumulates good skeletons across runs.

## 1. The genome as a proof-search program

A candidate `Φ` is a small program:

```
program Φ:
  state s, theorem name n
  goal_shape  = classify(s)
  retrieved_p = retrieve_premises(s, n, family_for(n))
  for skeleton in Φ.skeletons[goal_shape]:
      yield from instantiate(skeleton, s, retrieved_p)
  for skeleton in Φ.skeletons["any"]:
      yield from instantiate(skeleton, s, retrieved_p)
  for fallback in Φ.fallbacks:
      yield fallback
```

A *skeleton* is the new first-class object. Examples:

```
SKEL_iff_split = Skeleton(
    name="iff_split",
    shape="iff",
    template="exact ⟨fun h => by {fwd}, fun h => by {bwd}⟩",
    slots={
        "fwd": ["omega", "simp_all", "simp [h]", "subst h; simp_all"],
        "bwd": ["omega", "simp_all", "simp [h]", "subst h; rfl"],
    },
)

SKEL_dvd_witness = Skeleton(
    name="dvd_witness",
    shape="dvd",
    template="exact ⟨{wit}, by {proof}⟩",
    slots={
        "wit":   ["{n}/{d}", "{n}-{d}", "_"],
        "proof": ["simp_all", "rfl", "ring", "omega"],
    },
)

SKEL_split_ifs = Skeleton(
    name="split_ifs",
    shape="any",  # ite goals not classified separately yet
    template="split_ifs <;> {inner}",
    slots={"inner": ["omega", "simp_all", "simp_all <;> omega"]},
)

SKEL_induction = Skeleton(
    name="induction_var",
    shape="any",
    template="induction {var} with | zero => {z} | succ {n} {ih} => {s}",
    slots={
        "var": ...,   # filled per-state from nat_vars
        "z":   ["simp", "omega", "rfl"],
        "s":   ["simp [{ih}]", "omega", "simp_all"],
    },
)
```

Each candidate genome carries:

  - a **skeleton bag**: `dict[shape, list[Skeleton]]` (active skeletons per
    goal shape, ordered);
  - a **slot-vocabulary**: `dict[slot_label, list[str]]` (a small global
    vocabulary the mutator can draw from to fill any skeleton slot of
    that label);
  - the **legacy fields** (fallbacks, family tactics, retrieval knobs)
    kept as a thin compatibility layer for non-skeleton-bearing
    transitions.

Crucially, *all* skeletons go through the same `instantiate` step that
substitutes `{var}`/`{hyp_pos}`/`{hyp_le}` etc. plus slot identifiers
against the slot-vocabulary. So a single skeleton expands into many
candidate tactic strings without explicit enumeration in the genome.

## 2. Two-tier mutation

The mutator has two modes.

**Outer-tier (structural).**
  - Add / remove a skeleton from the bag.
  - Swap a skeleton's shape gate (e.g. promote `SKEL_iff_split` from
    `iff`-only to `any`).
  - Change skeleton ordering within a shape (priority).

**Inner-tier (slot).**
  - Insert / remove an entry from a slot's vocabulary.
  - Reorder a slot's vocabulary (this is the v3 → v4 ordering mutator,
    now scoped to a slot).
  - Substitute one slot vocabulary into another (e.g. let `fwd` borrow
    from `bwd`'s vocabulary).

Outer mutations are *rare* (10% of mutations) and high-impact; inner
mutations are *frequent* and low-impact. This matches the AlphaEvolve
mutation budget pattern.

## 3. Fitness

The current `scoring.py` is a weighted sum of `proved`,
`progress_count`, `total_steps`, `timeout`, `invalid`. Three additions:

  - **Skeleton attribution bonus.** When a skeleton fires and produces
    the win, award the candidate +1 per win attributed to that skeleton,
    weighted by *novelty* (a win on a theorem that the previous
    generation didn't close gets a higher weight than re-closing
    something the seed already proved).
  - **Skeleton breadth bonus.** A small bonus per *distinct goal shape*
    a candidate's skeletons activate on. Discourages
    one-shape-monoculture genomes.
  - **Compute cost.** Subtract proportional to total Lean roundtrips
    (`retrieved_premise_attempt_count + term_builder_attempt_count + ...`).
    Encourages the mutator to find compact, fast genomes.

## 4. Evaluator

Unchanged from v3 → v4 in shape: Lean grades every tactic emitted by
the wrapper. The wrapper is the *only* change: replace the linear
fallback / family / retrieval emit chain with an iterator over the
genome's skeleton bag, then the legacy chain as a tail.

## 5. Archive

A run-spanning JSONL file `project/evolve/archive/skeletons.jsonl` with
one row per (skeleton, theorem-closure) pair:

```json
{
  "skeleton": "iff_split",
  "slots_filled": {"fwd": "simp_all", "bwd": "simp [h]"},
  "theorem": "Nat.mul_eq_left",
  "first_seen_run": "v5-auto-20260522-…",
  "wins": 4,
  "regressions": [],
  "goal_shape": "iff"
}
```

When a new candidate is built, the mutator seeds its skeleton bag with
the top-N archived (skeleton, slots) pairs sorted by `wins`. This is
the cross-run accumulation that AlphaEvolve relies on for non-trivial
program evolution.

The archive is also the surface area where the trace-to-training
pipeline (Direction E) draws its training pairs: every archived
skeleton win that survived ≥ 5 consecutive runs is a candidate for
the next gen_v5+1 dataset.

## 6. Transfer to larger theorem sets

The skeleton vocabulary is *shape-keyed*, not name-keyed. An iff-split
skeleton with `fwd ← simp_all, bwd ← simp [h]` is portable to any iff
goal regardless of file. The first concrete test is the proposed
`nat_defs_large` (60-80 theorems); next is the existing
`curriculum_all`; next is a `Mathlib/Data/Set/Basic.lean`-only set to
see whether arithmetic-iff skeletons leak into set-iff goals.

The expected failure mode: arithmetic skeletons trained on Nat goals
fire on Set goals where `omega` doesn't apply. The
shape-gating prevents misfire only at the syntactic level; the
mutator needs **per-domain skeleton bags** for genuine transfer. This
is a v6 problem, not a v5 one.

## 7. Optional retraining

Out of scope for tonight. The Learn step is described in
`v5_trace_to_training_plan.md` (Direction E). The key constraint that
the architecture imposes on retraining is: the *(state, tactic)* pair
must be reproducible from the trace alone, without retrieval context,
or the trained model fails at inference time. This means
`retrieved_premise` transitions need either:

  - the retrieved lemma name inserted into the prompt at training time
    (premise-augmented dataset), OR
  - exclusion from the training set until the model has its own
    retrieval head.

The v5 corpus naturally fits the second path: train on `fallback`,
`family`, `term_builder` origin wins; keep `retrieved_premise` for a
separate study.

## 8. Why this is "deeper" than v3-v4

| dimension       | v3-v4                                  | v5+ (this proposal)                                 |
|-----------------|----------------------------------------|------------------------------------------------------|
| genome           | flat string lists                     | layered: skeletons + slot vocabularies               |
| mutation         | string-order rewrites                 | two-tier (structural + slot)                         |
| program structure| linear emit chain                     | shape-routed iterator                                |
| state            | per-run population                    | cross-run archive                                    |
| Learn step       | not connected                         | trace-to-training pipeline (Direction E)             |
| transfer         | per-theorem name matching             | per-shape classification                             |
| compute model    | per-state max_extra cap               | per-skeleton fire budget                             |

The single most important change is *not* the skeleton representation
itself — it is that the mutator can operate at two granularities.
v3-v4 climbing-the-hill in slot-vocabulary space (inner-tier) gets us
to the current 26/38. Outer-tier mutation is the only way to add a
new shape gate or a new skeleton, and that is what unlocks the
Bucket II theorems (asymmetric iff splits, dvd witnesses, ite splits).

## 9. Tonight's deliverables vs. the long arc

Tonight implements three small steps toward the architecture above:

  - **term_builder origin** with shape-keyed templates — this is the
    embryo of a `Skeleton[shape=iff, …]` object.
  - **mul/pow/sqrt families** — these encode shape-specific knowledge
    without yet decoupling structure from slot.
  - **autonomous_research_loop.py** — the embryo of the cross-run
    archive (it writes `scoreboard.jsonl` and `best_candidate.json`
    per run).

The next branch (v6) should refactor `strategy_wrapper.py` to drop the
linear emit chain and adopt the iterator-over-skeleton-bag form. After
that, the inner-tier mutator from `evolve/mutator.py` can be reused
verbatim on slot vocabularies, and a new outer-tier mutator added that
adds/removes skeletons. The archive then becomes the bridge to the
Learn step.

## 10. Risks of this design

  - **Combinatorial explosion.** Skeletons × slots × hypotheses can
    produce 50+ candidate tactics per state. Mitigation: per-skeleton
    fire budget, and `template_verifier.filter_templates` extended to
    slot-aware filtering (e.g. drop slot fills that reference an
    `_UNAVAILABLE_LEMMAS` constant).
  - **Mutation invalidates archive.** Adding a slot to a skeleton
    changes the (skeleton, slots_filled) key, breaking archive lookup.
    Mitigation: archive uses *positional* slots (fwd, bwd, etc.) keyed
    by skeleton name + Lean-syntactic-hash of the fully instantiated
    tactic, not by the slot keys themselves.
  - **The wrapper becomes harder to debug.** Per-state emit lists grow.
    Mitigation: per-skeleton trace tagging (already prototyped tonight
    with `term_builder_shape_keys`).

## 11. Open questions

  - Do we *learn* the slot vocabulary or curate it? Tonight's loop
    curates; a v6 approach could mine slot vocabulary from successful
    `retrieved_premise` traces.
  - Should there be a separate skeleton bag per *family* (e.g. div,
    mod), or only per *goal shape*? The Nat.add_mod_eq_ite case
    suggests yes (ite goals want a `split_ifs` skeleton that is
    family-gated, not shape-gated).
  - When does a skeleton get *retired*? Archive could grow without
    bound. Mitigation: skeletons with no win in the last K runs are
    archived to `skeletons.retired.jsonl` and not re-seeded.
