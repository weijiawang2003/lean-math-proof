# WX3 — Multiset quotient-aware wrapper probe

**Arc type:** mining + state-aware wrapper expansion (no neural training).
**Branch:** `wx3-multiset-quotient-wrapper`. **Router:** `ns24_router`
(Multiset → `default` route = `gen_v5_ns12_balanced`). **Baseline wrapper:**
NS9 genome (unmodified). **Symbolic layer:** AX1, extended additively for a
new `Multiset` var-type and two new action families; AX1/AX2 behavior
unchanged. **No NS9/router/AX1/AX2 artifacts were modified; no checkpoints
written.**

## 0. Goal & AX2 recap

AX2 found the symbolic-action dataset **capped at 27 examples / 4 labels**:
the fresh Option/List/Bool/Sum/Prod surface was exhausted and minimal
relabeling produced **0** new clean single-shot symbolic labels (the +3
fresh List wins were multi-step-assisted). AX2 recommended a **new
namespace** with untapped surface before any AX3 symbolic-action training.

WX3 tests whether a **state-aware Multiset wrapper** unlocks a new
wrapper-ready proof family. Multiset is harder than Option/List: many
proofs need quotient-aware induction (`Multiset.induction_on`) or
count/extensionality reasoning rather than a plain `cases <var>`.

## 1. Multiset catalog audit (Stage 1)

`scripts/wx3_multiset_catalog_audit.py` →
`project/data/wx3_multiset_catalog_audit_meta.json`,
`project/evolve/reports/wx3_multiset_catalog_audit.md`.

| metric | value |
|---|---|
| Multiset discovered (catalog) | 573 |
| Multiset **confirmed-available** | 260 |
| already probed (prior sets, excluded) | 19 |
| **fresh available candidates** | **251** |

By likely proof shape (precedence: quotient > induction > ext > simp):
`induction` 81, `simp` 79, `ext` 74, `quotient` 6, `hard` 11. This is by
far the largest untapped surface of the program (Option had 47, Bool 35,
Prod 5).

## 2. Theorem sets (Stage 2)

`scripts/build_wx3_theorem_sets.py` →
`project/evolve/routing/wx3_theorem_sets.json` (loaded by
`tasks._load_wx3_sets`). Five disjoint sets, **165 candidates** total, all
confirmed-available and fresh:

| set | n | shape target |
|---|---:|---|
| `wx3_multiset_simp_easy` | 40 | simp-only |
| `wx3_multiset_induction_easy` | 40 | `induction_on` |
| `wx3_multiset_ext_medium` | 35 | extensionality / count |
| `wx3_multiset_quotient_medium` | 20 | Quot / `induction_on` (+ hard induction) |
| `wx3_multiset_mixed` | 30 | balanced |

(+ `wx3_multiset_smoke`, 8 thms, for Stage 5.)

## 3. Wrapper action design (Stage 3)

Extended `project/evolve/symbolic_actions.py` and
`project/evolve/state_vars.py` **additively** (AX1 Option/List/Bool
rendering byte-for-byte unchanged; the AX1 config still loads with all 5
actions valid):

- `state_vars`: added `Multiset` coarse type (a `Multiset α` binder now
  classifies as `Multiset`; everything else unchanged).
- `symbolic_actions`: added var-type `Multiset` and action types
  - **`MULTISET_INDUCTION_SIMP`** → `induction {var} using
    Multiset.induction_on <;> {simp_mode}` (quotient-aware induction).
  - **`EXT_SIMP`** → `ext x <;> {simp_mode}` (variable-independent, but
    emission gated on a Multiset variable being present, keeping it
    state-aware).
  - `CASES_SIMP` with `var_type=Multiset` reuses the existing renderer →
    `cases {var} <;> {simp_mode}`.

All actions are namespace-gated to `Multiset` and capped at 1 variable.
Emitted tactics carry `origin = wrapper_symbolic_action` and a
`multiset_*` `family_source`.

## 4. WX3 configs (Stage 4)

`project/evolve/experiments/wx3/` — each config's NS9 base is **byte-identical
to `ns9_best_genome.json`** (verified), differing only by an added
`Multiset.`-gated `symbolic_actions` block:

- `wx3_multiset_induction_safe.json` — `MULTISET_INDUCTION_SIMP` × {simp_all, simp}, max_vars 1.
- `wx3_multiset_ext_safe.json` — `EXT_SIMP` × {simp_all, simp}.
- `wx3_multiset_combined_safe.json` — induction + ext + cases (6 actions).

When the block is disabled / absent, behavior is identical to routed-raw
(`raw`) or the NS9 genome (`ns9`).

## 5. Smoke syntax test (Stage 5)

`project/data/wx3_multiset_smoke_meta.json`. Combined config on the
8-theorem smoke set: **no crashes, no syntax failures**. The Multiset
symbolic tactics are accepted by Lean and correctly tagged; one win
(`Multiset.add_inter_distrib`) came via `multiset_induction_simp_all`,
confirming the new action can close goals. `induction {var} using
Multiset.induction_on <;> simp_all` parses and runs.

## 6. raw vs NS9 vs WX3 matrix (Stage 6)

`scripts/wx3_run_matrix_parallel.sh` (25 runs, ns24_router, top-k 8,
max-steps 8) → `scripts/wx3_extract_probe.py` →
`project/data/wx3_multiset_probe_meta.json`.

| set | raw | NS9 | ind | ext | comb |
|---|---:|---:|---:|---:|---:|
| simp_easy | 4 | 4 | 8 | 5 | 8 |
| induction_easy | 2 | 2 | **16** | 2 | **16** |
| ext_medium | 1 | 1 | 2 | 1 | 2 |
| quotient_medium | 0 | 0 | 2 | 0 | 2 |
| mixed | 1 | 1 | 5 | 1 | 5 |

- **WX3-only wins beyond NS9: +25** (induction-only `ind` ≡ combined `comb`,
  same 25 theorems; `ext`-only adds just 1, subsumed by comb).
- **Regressions vs NS9: 0.**
- Win attribution: **22 via `wrapper_symbolic_action`**
  (`multiset_induction_simp_all`), 3 via base-model `aesop`
  (`generative_topk`) — the latter surfaced by the reordered search and
  resolved in Stage 8.

The workhorse is unambiguous: `induction {var} using Multiset.induction_on
<;> simp_all`.

## 7. Minimal relabeling (Stage 8)

`scripts/wx3_relabel_minimal_multiset.py` (battery: assumption, rfl, decide,
simp, simp_all, aesop, ext×2, cases×2, induction_on×2, wrapper fallback) →
`project/data/wx3_minimal_multiset_labels.json`,
`project/data/wx3_multiset_family_pools_meta.json`.

| metric | value |
|---|---:|
| WX3-only wins relabeled | 25 |
| **clean single-shot symbolic** (wrapper- & SFT-ready) | **20** |
| dominant `MULTISET_INDUCTION_SIMP[Multiset,simp_all]` | **18** |
| `MULTISET_INDUCTION_SIMP[Multiset,simp]` | 2 |
| induction_on family aggregate | **20** |
| dropped (simpler tactic closes single-shot) | 1 (`finite_toSet_toFinset` → `aesop`) |
| multi-step symbolic-assisted | 2 |
| multi-step non-symbolic | 2 (the flaky `aesop` wins) |

All 20 clean labels are solved by **neither raw nor NS9** — genuine new
capability. The single label class `MULTISET_INDUCTION_SIMP[Multiset,
simp_all]` (18 unique) clears the per-label uniqueness gate (≥5).

This is a categorical improvement over AX2 (**0** clean single-shot
symbolic labels). The NS22 "long-structured-tactic" ceiling does **not**
apply: we learn the short, stable *symbolic label* and the wrapper
instantiates `{var}` from the state.

## 8. Preservation (Stage 7)

`scripts/wx3_preservation_extract.py` →
`project/data/wx3_preservation_matrix.json`,
`project/evolve/reports/wx3_preservation_matrix.md`. WX3 base ==
`ns9_best_genome.json` and the Multiset block is `Multiset.`-gated, so on
non-Multiset theorems the ranked list is identical to NS9 — **preservation
by construction**, confirmed empirically:

| set | ns class | NS9 | WX3-comb | Δ | regress | Multiset emit |
|---|---|---:|---:|---:|---:|---:|
| demo_v1 | mixed | 11 | 11 | +0 | 0 | 0 |
| nat_defs_medium | Nat | 37 | 37 | +0 | 0 | 0 |
| ns17_set_extra | Set | 18 | 18 | +0 | 0 | 0 |
| ns17_finset_extra | Finset | 15 | 15 | +0 | 0 | 0 |

`nat_defs_large_v5` (49/65) and `ns14_set_finset_extra` preserved by the
same ranked-list identity (not re-run). NS9 canonical floors preserved:
**medium 37/38, large 49/65, demo 11/15.** Multiset-action emissions
outside Multiset: **0**.

## 9. Gate classification (Stage 9)

`scripts/wx3_gate_decision.py` → `project/data/wx3_gate_decision_meta.json`.

- **A. Wrapper-ready — MET (clean).** +25 wins beyond NS9 (≥5), 0 matrix
  regressions, 0 preservation regressions, 0 leakage; actions state-aware
  and reliable (20 reproducible single-shot symbolic wins).
- **B. Symbolic-learning-ready — BORDERLINE / MET by family aggregate.**
  20 clean single-shot symbolic labels (< 40); strongest single action_id
  18 (< 20 strict); but the `MULTISET_INDUCTION_SIMP` action family
  (both simp modes) = **20 ≥ 20**, and held-out surface remains
  (251 fresh available − 165 evaluated ≈ 86 unused, plus the broader
  induction-shape catalog). So Gate B is met under the family-aggregate
  reading, not under the strict single-action-id reading.
- **C. Multi-step-only — no** (4 of 25 are multi-step; the bulk are
  single-shot).
- **D. Negative — no.**

**Verdict: A (+ borderline B).**

## 10. Recommendation

1. **Promote the WX3 Multiset wrapper.** `wx3_multiset_induction_safe`
   alone captures all 25 wins; `wx3_multiset_combined_safe` adds ext/cases
   for generality at **zero** regression cost. Either is a clean, gated,
   state-aware addition behind an experimental config flag.
2. **AX3 is plausible for the first time.** The induction_on family reaches
   18 (simp_all) / 20 (both modes) clean single-shot symbolic labels —
   at/near the ≥20-in-one-family gate, versus AX2's 0. Recommended path:
   **WX4 / surface expansion first** — mine the ~86 held-out fresh Multiset
   + the full induction-shape catalog under the WX3 induction wrapper to
   push clean labels to **≥40 total / ≥20 in the single `simp_all`
   action_id**, then train **AX3** on the `MULTISET_INDUCTION_SIMP`
   symbolic-action label with a held-out Multiset eval.
3. The 2 multi-step symbolic-assisted wins (`disjoint_union_right`,
   `eq_zero_of_forall_not_mem`) motivate eventual **sequence-level symbolic
   search**, but are a minority and not the priority.

## Artifacts

Scripts: `wx3_multiset_catalog_audit.py`, `build_wx3_theorem_sets.py`,
`wx3_run_eval.sh`, `wx3_run_matrix.sh`, `wx3_run_matrix_parallel.sh`,
`wx3_run_preservation.sh`, `wx3_extract_probe.py`,
`wx3_preservation_extract.py`, `wx3_relabel_minimal_multiset.py`,
`wx3_gate_decision.py`. Module edits: `project/evolve/symbolic_actions.py`,
`project/evolve/state_vars.py`. Configs:
`project/evolve/experiments/wx3/*.json`,
`project/evolve/routing/wx3_theorem_sets.json`. Metadata:
`project/data/wx3_*`. Eval traces/logs/run dirs are git-ignored.
