# SX1 — symbolic sequence search prototype

**Branch:** `sx1-symbolic-sequence-search`
**Base:** AX4 GREEN (`8f61f62`, branch `ax4-multiset-symbolic-green`)
**Stage:** search / prototype (no neural retraining)
**Decision:** **Gate B — dataset-generation success** (see §9)

---

## 1. AX4 recap

AX4 trained a learned single-action symbolic selector (TF-IDF char_wb 3–5 +
balanced logistic regression) over the proof state for the two Multiset
induction actions. It is **promotion-eligible but off by default**: CV top-1
0.90, positive recall 0.72, held-out retain 0.538, 0 regressions, 0 effective
non-Multiset FP. The deterministic **WX3 oracle wrapper stays the production
Multiset default** because always emitting the single additive induction action
is free and covers 13/13 held-out. AX4's closing recommendation was to *move to
sequence-level symbolic search, where learned selectivity can actually matter.*
SX1 is that move.

## 2. Motivation

A single symbolic action frequently **advances** the proof without closing it:
`induction s using Multiset.induction_on <;> simp_all` leaves an inductive-step
goal that a follow-up (`aesop`, `simp_all`, base-model top-k) closes. SX1 makes
that two-step shape a first-class, namespace-gated, depth-2 object so the
wrapper can emit it deliberately, and so the resulting `(first symbolic action,
follow-up)` pairs become a learnable label set.

## 3. Multi-step inventory (Stage 1)

Mined **offline** from the existing oracle/symbolic trace corpus (`*_wx3ind_*`
Multiset, `*_ax1sym_*` List). There is no live Lean in this arc; every tactic
the search tried on every state is already recorded with state-hash links and a
`result_kind`.

| metric | value |
|---|---|
| symbolic-action firings scanned | 812 |
| single-shot closes (symbolic ⇒ ProofFinished) | 31 |
| advanced (symbolic ⇒ TacticState) | 217 |
| **unique multistep symbolic-assisted cases** | **5** (AX4 ×3 Multiset, AX2 ×2 List) |

**Decisive structural finding:** the existing NS9/WX3 best-first search *already*
explores **~9 follow-up tactics per advanced symbolic state** (max 22), spanning
generative top-k, templates, the `omega` fallback, and re-applied symbolic
actions. Every one of the 5 multistep cases was therefore **already closed by
that search** — the Multiset ones by base-model `aesop`, the List ones by a
re-applied `cases l <;> simp_all`. A depth-2 fixed-battery sequence mode is, in
effect, **subsumed** by the search that already runs.

Artifacts: `project/data/sx1_multistep_symbolic_cases_meta.json`,
`project/evolve/reports/sx1_multistep_symbolic_cases_inventory.md`.

## 4. Sequence schema (Stage 2)

Added additively to `project/evolve/symbolic_actions.py`:

- `SymbolicActionSequence` — `first_action: SymbolicAction` + `followup_mode`
  (`base_topk` | `fixed_battery` | `simp_all`), `max_depth` (**2 only**),
  `namespace_gate`, `max_followup_tactics`, `priority`, `family_source`,
  `sequence_id`, `stop_condition` (`proof_finished` | `max_depth` |
  `no_progress`). `.actions` returns the leading symbolic action; the follow-up
  is a mode, because observed closers are plain tactics / base top-k, not typed
  symbolic actions. `validate()` enforces depth == 2.
- `battery_for_namespace()` — fixed battery `simp, simp_all, aesop, rfl`, with
  `omega, decide` appended only for arithmetic namespaces (never emitted on a
  Multiset/Option/List goal).
- `load_sequences()` — config → validated sequences.

## 5. Sequence execution (Stage 3)

`project/evolve/symbolic_sequence.py`, **behind a flag**:

- `SequenceSearchConfig.from_config()` parses the `symbolic_sequence_search`
  block. Empty namespace-gate list ⇒ fires nowhere (strictly opt-in).
- `plan_sequences(state, full_name, cfg, base_topk=None)` returns `[]` when the
  flag is disabled or the namespace is not gated — so the raw / single-action
  path is **byte-identical** when sequence mode is off. When on, it instantiates
  each gated symbolic first action on the live state and attaches a capped,
  deduped follow-up list (base top-k ∪ fixed battery). Plans are **additive** to
  the NS9 ranked list; depth 2; capped first actions and follow-ups.

## 6. Configs (Stage 4)

`project/evolve/experiments/sx1/` — all NS9-genome base + the unchanged
single-action `symbolic_actions` block + an experimental
`symbolic_sequence_search` block:

- `sx1_multiset_sequence_safe.json` — Multiset only, induction⇒base/battery.
- `sx1_option_list_sequence_safe.json` — Option/List CASES_SIMP⇒base/battery.
- `sx1_combined_sequence_safe.json` — all three, per-namespace gated.

## 7. Theorem sets (Stage 5)

`project/evolve/routing/sx1_theorem_sets.json` (70 theorems), carved from the
trace corpus, advanced-but-not-closed prioritised:

| set | n |
|---|---|
| sx1_multiset_multistep_candidates | 22 |
| sx1_list_multistep_candidates | 12 |
| sx1_option_multistep_candidates | 0 (corpus has **no** Option symbolic cases — CX3 negative) |
| sx1_mixed_symbolic_sequence_eval | 16 |
| sx1_negative_control | 20 |

## 8. Eval matrix (Stage 6)

Offline replay scoring four wrappers (`project/data/sx1_sequence_probe_meta.json`):

| set | n | A baseline | B single-action oracle | E full wrapper search | D sequence | seq-only vs B | seq-only vs full |
|---|---|---|---|---|---|---|---|
| multiset_multistep | 22 | 0 | 0 | 3 | 3 | **3** | 0 |
| list_multistep | 12 | 0 | 2 | 2 | 2 | 0 | 0 |
| option_multistep | 0 | – | – | – | – | – | – |
| mixed_eval | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| negative_control | 20 | 6 | 0 | 6 | 0 | 0 | 0 |
| **total** | 70 | 6 | 2 | 11 | 5 | **3** | **0** |

- **`sequence_only_beyond_oracle (B) = 3`** — the Multiset `induction … simp_all`
  ⇒ `aesop` cases (`mem_add`, `mem_map`, `mem_sigma`). A deliberate depth-2 plan
  reproduces 3 closes the single-action oracle alone cannot.
- **`sequence_only_beyond_full_wrapper (E) = 0`** — but the *production* WX3
  wrapper, with its open follow-up search, **already wins all of them**. The
  depth-2 plan adds **no net new wins** over what ships today.
- **regressions = 0**; negative control emits nothing (D = 0 there).

## 9. Minimal relabeling (Stage 7)

`project/data/sx1_minimal_sequence_labels.json`,
`project/data/sx1_sequence_family_pools_meta.json`. All 5 multistep cases
classify as **`genuinely_sequence_needed`** (no single battery/symbolic tactic
closes from the initial state in the corpus) — none are raw- or single-action
over-attribution. But the genuine label corpus is tiny: **5 labels, biggest
family pool 3** (`SEQ[Multiset:induction⇒aesop]`), far below the ≥5-per-family /
≥40-total gate that AX3/AX4 needed before a learner was worthwhile.

## 10. Runtime / safety

- Depth 2 only; first actions capped (`max_symbolic_first_actions = 2`);
  follow-ups capped (`max_followup_tactics = 6`); per-theorem timeout 60s.
- Namespace-gated; off-gate emissions provably 0 (Stage 8).
- Disabled by default ⇒ raw and single-action paths byte-identical.
- Plans additive to the NS9 ranked list — never reorder/replace it.

## 11. Preservation (Stage 8)

`project/data/sx1_preservation_matrix.json`,
`project/evolve/reports/sx1_preservation_matrix.md`. Combined config with the
flag **ON**, run through the real planner over each set's initial states:

- **Off-gate emissions = 0** on demo_v1 (15), nat_defs_medium (38),
  nat_defs_large_v5 (64), ns17_set_extra (30), ns17_finset_extra (30).
- Gated surfaces emit additively: wx2_list (38/27/12), wx3_multiset_heldout (34).
- NS9 floors preserved (genome byte-unchanged, additive plans): medium 37/38,
  large 49/65, demo 11/15.

## 12. Decision

**Gate B — dataset-generation success.**

- **Direct gains beyond production = 0.** The existing best-first search already
  performs the depth-2 follow-up; a fixed-battery sequence mode is subsumed by
  it. `seq_only_beyond_full_wrapper = 0`. So: **do not promote SX1 as a
  production search expander** — it would add cost (extra emissions) for no new
  wins.
- **Clean depth-2 traces exist** (217 advanced states richly followed-up; 5
  reproducible `(symbolic first action ⇒ follow-up)` chains, all genuinely
  sequence-needed). This is the durable output.
- Echoing AX4: the lever is **selectivity** — learning *which* depth-2 plan to
  emit — not raw search reach. That is an **AX5 sequence-label learner**, but it
  is **not trainable yet**: 5 genuine labels (biggest pool 3) ≪ the ≥40 / ≥5-
  per-family gate.

**Recommendations**

1. **Keep** the SX1 schema + planner + configs (committed) as the experimental,
   off-by-default depth-2 capability and the clean trace generator.
2. **Keep WX3 oracle wrapper** as the production Multiset default (unchanged);
   **keep AX4 predictor** promotion-eligible but off.
3. **Do not** flip `symbolic_sequence_search.enabled` on in production: 0 net
   wins, added emission cost.
4. **Before AX5**: mine the SX1 candidate sets with **live Lean** under the
   sequence configs to grow the genuine depth-2 label pool to ≥40 (the offline
   corpus only re-derives what the open search already found). If a family
   clears ≥5 unique, train an AX5 sequence-label selector; otherwise return to
   single-action mining / a fresh namespace.
