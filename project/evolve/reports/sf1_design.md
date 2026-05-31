# SF1 — Scalable Frontier Miner (Design)

Status: **DESIGN / SCAFFOLD** (no production change)
Branch: `rc1-production-stack`
Baseline release: RC1 (commit `f3b3100`), release artifacts `72c1250`
Author: evolve pipeline
Date: 2026-05-29

> SF1 is **infrastructure only**. It does **not** modify the RC1 production
> wrapper, the NS9 genome, the NS24 router, or the REL1 release reports. All
> SF1 scripts are safe no-ops under `--dry-run` and never mutate production
> configs.

---

## 1. Motivation (after RC1 / REL1)

RC1 consolidated every deterministic gain the project has proven into a single
production wrapper:

```
RC1 = NS9 base  ⊕  WX3 Multiset induction oracle  ⊕  MX2 narrow Set.Finite aesop
```

It delivers **+15 wins beyond NS9** (WX3 +12 Multiset, MX2 +3 Set.Finite) with
**0 regressions, 0 off-gate emissions**, and preserved floors. The AX4 learned
symbolic selector and the SX1 depth-2 sequence search are wired in but
**OFF by default** because neither beats the deterministic oracle on the surfaces
we have mined so far.

The problem RC1 exposes is **not** quality — it is **method**. Every gain in the
NS/WX/AX/MX/SX line was produced by *hand-guided namespace mining*: a human
picked a namespace (Option, List, Multiset, Finset, Set), guessed an action
family, and ran a bespoke mine. That process has now visibly **saturated**:

- Multiset / List / Option short-token surfaces are exhausted (CX3, WX1–WX3).
- Finset / Set ext/cases surfaces are *base-policy-strong*; the symbolic
  constructor/ext actions close 0 there (MX1), and the only residual headroom is
  cheap namespace-gated battery tactics like `aesop` (MX2).

The lesson captured across NS21–MX2 is structural:

> Post-NS9 headroom on **strong-base-policy** namespaces is captured by cheap
> namespace-gated battery tactics. Symbolic constructor/ext actions only pay on
> **weak-base-policy** structural surfaces (the Multiset quotient).

Hand-guided mining cannot tell us, in advance and at scale, *which* namespaces
are weak-base-policy structural surfaces worth a symbolic action versus
strong-base-policy surfaces worth a cheap battery tweak. SF1 is the machine that
answers that question by sweeping the frontier instead of guessing it.

## 2. Why RC1 is — and stays — the production baseline

- It is **deterministic** and **reproducible**: no learned component in the hot
  path, fixed tactic batteries, syntactic gates.
- Every win is **minimal-sufficiency attributed** (NS23-style relabel): no win is
  credited to a template that plain `omega`/`aesop`/`simp` would have produced.
- It has **0 regressions / 0 off-gate** on the held-out floors.

SF1 treats RC1 as the **control arm**. A candidate family is only interesting if
it adds wins *beyond RC1* under the exact same evaluation harness. SF1 never
proposes replacing RC1 wholesale; it proposes **additive, gated** extensions that
must clear RC1 on every safety axis before promotion.

## 3. Why we stop hand-guided namespace mining

Hand-guided mining has four failure modes that SF1 removes:

1. **Selection bias** — humans mine namespaces they already suspect, so the
   "frontier" is whatever was recently discussed, not what is globally open.
2. **Attribution drift** — bespoke mines re-implement relabeling each time; NS23
   showed how easily a wrapper mislabels a pool (iff-pair stealing `omega` wins).
3. **No consumed-surface bookkeeping** — the same theorems get re-mined across
   experiments, inflating apparent gains (NS21 transfer-ceiling artifact).
4. **Non-comparable evals** — each experiment built its own raw/wrapper compare,
   so cross-experiment deltas were never apples-to-apples.

SF1 replaces this with a single, repeatable **AlphaEvolve-style discovery loop**:

```
Mathlib theorem frontier
  → candidate action family
  → live LeanDojo evaluation
  → minimal-sufficient relabeling
  → family pool update
  → safe promotion / training decision
```

## 4. SF1 pipeline stages

Each stage is a standalone script under `scripts/` that reads/writes JSON(L) and
runs as a deterministic no-op under `--dry-run`. Stages are composable; the
canonical wiring is `extract → filter → classify → make-batches → eval-matrix →
minimal-relabel → promotion-report`.

### a. Mathlib catalog extraction — `sf1_extract_mathlib_catalog.py`
Walk the traced Mathlib cache and emit one record per theorem:
`{name, namespace, file, line, statement, decl_kind}`. Output: `catalog.jsonl`.
(TODO: drive from the 18GB traced LeanDojo cache; today it emits a deterministic
placeholder catalog.)

### b. Consumed-surface exclusion — `sf1_filter_consumed_surfaces.py`
Subtract every theorem already consumed by a prior experiment (NS/WX/AX/MX/SX/RC1
input sets) so we mine genuinely **open** frontier. Output: `frontier.jsonl` plus
an exclusion ledger. This is the bookkeeping hand-guided mining never had.

### c. Syntactic / namespace classification — `sf1_classify_frontier.py`
Tag each frontier theorem with `(namespace_bucket, syntactic_shape)`, where
`syntactic_shape ∈ {eq, iff, forall_eq, membership, cases_like, induction_like,
arith, other}`. This is what lets us tell weak-base-policy structural surfaces
(quotient/induction) from strong-base-policy ones (ext/arith). Output:
`classified.jsonl`.

### d. Balanced batch generation — `sf1_make_batches.py`
Use `sf1_batch_policy.json` to draw a batch that is **balanced by namespace** and
**balanced by syntactic shape**, with explicit **holdout buckets** for Multiset
and Set.Finite (our two known-live surfaces) plus a **general frontier** bucket.
Deterministic seed. Output: `batches.jsonl`.

### e. raw / NS9 / RC1 / experimental evaluation — `sf1_eval_matrix.py`
For every batched theorem, run four arms under one harness:
`raw` (base policy, no wrapper), `ns9` (NS9 base wrapper), `rc1` (production
stack, the control), and one or more `experimental` candidate families from
`sf1_candidate_families.json`. Record pass/fail and the winning tactic per arm.
Output: `eval_matrix.jsonl`. (TODO: live LeanDojo; macOS has no `timeout`, use
`scripts/run_with_timeout.py`. Dojo opens ~6s, ~2s/theorem, hard Set ~80s.)

### f. Minimal-sufficient relabeling — `sf1_minimal_relabel_new_wins.py`
For each theorem an experimental arm wins **beyond RC1**, re-run the minimal
tactic battery (NS23 discipline) to confirm the win is genuinely attributable to
the candidate family and not to plain `omega`/`aesop`/`simp`/`assumption`.
Output: `relabeled.jsonl` with `clean_*` vs `over_attributed` verdicts.

### g. Family-pool update — (consumed by `sf1_promotion_report.py`)
Fold the clean, minimal-sufficient labels back into the candidate family's pool
and update `sf1_candidate_families.json` *status/notes* (proposal only — never
auto-flips a production flag).

### h. Promotion / training recommendation — `sf1_promotion_report.py`
Emit a `promotion_report.{json,md}` that scores each experimental family against
the promotion criteria below and recommends one of:
`PROMOTE` / `KEEP_OFF_BY_DEFAULT` / `REJECT` / `MINE_MORE` (label-limited).

## 5. Promotion criteria

A candidate family is **promotion-eligible into the RC1-class deterministic
stack** only if it satisfies **all** of:

1. **Positive delta over RC1** — `wins(experimental) − wins(rc1) > 0` on the batch,
   measured beyond the control arm, not beyond raw.
2. **Zero regressions** — no theorem won by RC1 is lost by the experimental arm.
3. **Zero off-gate emissions** — the candidate's tactic never fires outside its
   declared syntactic/namespace gate (negative-control namespaces stay at 0).
4. **Strict syntactic gates** — the family declares a narrow, checkable gate
   (e.g. `Set.Finite/toFinset`, not broad `Set.`); broad gates that overfire are
   rejected even at equal wins (MX2 lesson).
5. **Minimal-sufficient attribution** — every credited win survives NS23-style
   relabel as `clean_<family>` (not `over_attributed` / `assumption_closable`).
6. **Deterministic reproducibility** — same seed + same traced cache ⇒ identical
   batch, eval matrix, and labels. No learned component in the promoted hot path.

Families that show signal but miss criterion 1 *only because of label scarcity*
get `MINE_MORE` (the AX3/AX4 path: learner alive but label-limited; mine to ≥40
before live integration). Learned selectors (AX4) and sequence search (SX1) stay
**off by default** until they beat the deterministic oracle, not just raw.

## 6. Relation to prior NS / WX / AX / MX / SX experiments

SF1 generalizes the whole line into one loop:

| Prior work | What it was (hand-guided) | How SF1 subsumes it |
|---|---|---|
| NS9 | base wrapper / genome | the `ns9` eval arm (a candidate family) |
| NS21–NS23 | transfer/complexity ceilings, minimal-tactic relabel | the relabel stage (f) + criteria 5 |
| NS24 | router | untouched; SF1 routes via family registry |
| CX3 | Bool/Option negative result | recorded as consumed + rejected surface |
| WX1–WX3 | state-aware cases / quotient induction | `wx3_multiset_induction` family |
| AX1–AX4 | learned symbolic selector (label-limited) | `ax4_learned_symbolic_selector_off_by_default`, `MINE_MORE` path |
| SX1 | depth-2 sequence search (subsumed by best-first) | `sx1_depth2_sequence_search_off_by_default` |
| MX1 | Set/Finset symbolic mine (aesop over-attributed) | `mx1_set_finset_symbolic_rejected` |
| MX2 | narrow Set.Finite aesop fallback | `mx2_set_finite_tofinset_aesop` (in RC1) |
| RC1 | consolidated production stack | the `rc1` **control arm** |

The structural lesson SF1 operationalizes: **symbolic constructor/ext actions are
reserved for weak-base-policy structural surfaces; strong-base-policy surfaces
get cheap namespace-gated battery tactics.** SF1's classifier (stage c) is exactly
the tool that decides which bucket a fresh frontier namespace falls into.

## 7. RC2 pathway

RC2 is *not* this PR. RC2 is what SF1 enables:

1. Run SF1 end-to-end on a balanced frontier batch with live LeanDojo.
2. Collect candidate families that clear **all six** promotion criteria.
3. For each `PROMOTE` family, add it as an additive, gated layer on top of RC1
   exactly as MX2 was added (deep-copy + gated battery / oracle action, NS19
   pattern), with its own `clean_*` minimal-sufficient labels.
4. Re-freeze as `RC2 = RC1 ⊕ {newly promoted families}`, preserving 0
   regressions / 0 off-gate / floors.
5. `MINE_MORE` families (AX-style learners) graduate only after their live-Lean
   mine reaches the ≥40-label threshold and they beat the deterministic oracle.

Until then RC1 remains the recommended production wrapper
(`project/evolve/experiments/rc1/rc1_production_wrapper.json`) and the recommended
production command is unchanged.
