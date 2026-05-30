# WX2 — state-aware cases-wrapper consolidation & generalization

**Arc type:** wrapper-capability consolidation + generalization probe (no
training). **Branch:** `wx2-state-aware-cases-generalization`. **Router:**
`ns24_router`. **Baseline wrapper:** NS9 best genome (unmodified).
**Configs (experiment files under `project/evolve/experiments/wx2/`):**
`wx2_option_cases_promoted.json` (Option-only, promoted WX1) and
`wx2_option_list_cases_safe.json` (Option+Bool+List generalized).

## 0. WX1 recap

WX1 added a namespace-gated, state-aware `option_cases_skeletons` block
to `StrategyWrapperPolicy` that reads the case variable from the proof
state and emits `cases <var> <;> simp_all`. Result: **+19 Option wins
beyond NS9, zero regressions** — but the family is the variable-dependent
compound `cases <var> <;> simp`, **wrapper-ready, not short-token
SFT-ready**. WX2 consolidates that into a promoted config and tests
whether the same pattern generalizes beyond Option.

## 1. Promoted Option wrapper (Stage 1)

`wx2_option_cases_promoted.json` = NS9 genome + the WX1-validated Option
cases block, trimmed to the two minimal-confirmed tactics
(`cases {var} <;> simp_all`, `cases {var} <;> simp`), Option-namespace
gated. The wrapper itself gained backward-compatible extensions:
per-type namespace gates (`type_namespace_gates`), per-type family
labels (`family_source_by_type`), and notation-aware variable matching
(Sum `⊕` / Prod `×`). With no `option_cases_skeletons` block the wrapper
is byte-identical to NS9.

## 2. Broad preservation matrix (Stage 2)

`scripts/wx2_preservation_extract.py` →
`project/data/wx2_preservation_matrix.json`. NS9 vs **WX2-promoted**:

| set | ns class | NS9 | WX2-prom | Δ | regress | option_cases emit |
|---|---|---:|---:|---:|---:|---:|
| cx3_option_simp_easy | Option | 15 | 26 | +11 | 0 | 19 |
| cx3_option_cases_medium | Option | 7 | 11 | +4 | 0 | 7 |
| cx3_bool_option_mixed | Option/Bool | 18 | 22 | +4 | 0 | 18 |
| cx3_bool_decide_easy | Bool | 2 | 2 | 0 | 0 | 0 |
| demo_v1 | mixed-Nat | 11 | 11 | 0 | 0 | 0 |
| nat_defs_medium | Nat | 37 | 37 | 0 | 0 | 0 |
| ns17_set_extra | Set | 18 | 18 | 0 | 0 | 0 |
| ns17_finset_extra | Finset | 15 | 15 | 0 | 0 | 0 |

- **Option-surface delta +19** — the WX1 gain is **fully retained** by
  the trimmed 2-tactic promoted config.
- **Non-Option regressions: 0. Emissions outside Option: 0.** Nat/Set/
  Finset/demo are identical to NS9 (gate holds empirically; also proven
  by ranked-list identity). `nat_defs_large_v5` / `ns14_set_finset_extra`
  unchanged by the same gate (not re-run).

**The promoted Option wrapper is safe to promote.**

## 3. Generalization-surface audit (Stage 3)

`scripts/wx2_cases_catalog_audit.py` →
`project/data/wx2_cases_catalog_audit_meta.json`. Fresh (unused)
cases-friendly candidates by namespace:

| namespace | available | fresh | cases-friendly | note |
|---|---:|---:|:---:|---|
| List | 260 | **165** | yes | 151 cases + 14 induction |
| Option | 46 | 0 | yes | exhausted by CX3 |
| Bool | 35 | 0 | yes | exhausted by CX3 |
| Prod | 5 | 1 | yes | tiny |
| Sum | 0 | 0 | — | absent from catalog |
| Multiset | 260 | 251 | **no** | quotient type — `cases`/`induction` on a raw var does not apply |

**Verdict: WX2 generalization is fundamentally a List test** — List is
the only large fresh cases-friendly surface.

## 4. Fresh theorem sets (Stage 4)

`scripts/build_wx2_theorem_sets.py` →
`project/evolve/routing/wx2_theorem_sets.json` (loaded by
`tasks._load_wx2_sets`). 95 fresh: `wx2_list_cases_easy` (40),
`wx2_list_cases_medium` (35), `wx2_list_induction` (14),
`wx2_prod_cases` (1), `wx2_bool_cases_control` (5, reuses CX3 fresh-Bool
as a negative control since Bool is exhausted).

## 5. Generalized cases wrapper (Stage 5)

`wx2_option_list_cases_safe.json` enables per-type, per-namespace-gated
skeletons: Option → `cases {var} <;> simp_all|simp`; Bool →
`cases {var} <;> decide|simp_all`; List →
`cases {var} <;> simp_all|simp`, `induction {var} <;> simp_all`. Each
type's tactics fire only on its own namespace (`type_namespace_gates`).

## 6. raw vs NS9 vs WX2-generalized matrix (Stage 6)

`scripts/wx2_generalized_probe_extract.py` →
`project/data/wx2_generalized_cases_probe_meta.json`.

| set | raw | NS9 | **WX2-gen** | gen-only | regress |
|---|---:|---:|---:|---:|---:|
| wx2_list_cases_easy | 4 | 4 | **9** | **+5** | 0 |
| wx2_list_cases_medium | 12 | 12 | **17** | **+5** | 0 |
| wx2_list_induction | 1 | 1 | 1 | 0 | 0 |
| wx2_prod_cases | 0 | 0 | 0 | 0 | 0 |
| wx2_bool_cases_control | 2 | 2 | 2 | 0 | 0 |
| **total** | 19 | 19 | **29** | **+10** | **0** |

- **+10 new List wins beyond NS9, zero regressions.** raw == NS9
  everywhere (NS9 wrapper is a **no-op on List**, same pattern as
  Option/Bool — the base model has no List-cases capability and the NS9
  skeletons don't supply one).
- 8 wins are direct `cases l <;> simp_all` (`wrapper_option_cases`); 2
  are indirect (the cases skeleton advanced the state, then the model
  finished via `simp_all`/`aesop`).
- **`induction` added nothing** (1→1): the fold/length family is not
  closed by `induction <var> <;> simp_all` alone. **Prod** (1 theorem)
  and the **Bool control** added nothing — generalization is
  specifically the `cases` (cons/nil) split on List.

## 7. Minimal-tactic relabeling (Stage 7)

`scripts/wx2_relabel_minimal_cases.py` →
`project/data/wx2_minimal_family_pools_meta.json`. 9/10 resolved
(`List.head_cons_tail` is a multi-step cases win, not closed by any
single battery tactic from the initial state).

| minimal family | unique | short-token? | SFT-ready | wrapper-ready |
|---|---:|:---:|:---:|:---:|
| `list_cases_simp \| List` | **6** | no | ✗ | **✓** |
| `list_cases_simp_all \| List` | 3 | no | ✗ | ✗ |

**Every resolved win's minimal tactic is `cases l <;> simp[_all]`** — a
state-aware compound, not a short token. Plain `simp`/`simp_all`/`aesop`/
`decide` were tried first and fail; the cons/nil split is required.
`any_sft_gate_met: false`; `any_wrapper_gate_met: true`.

## 8. Decision gate (Stage 8): **A — promote wrapper capability**

- **Wrapper-product gain: MET.** +10 new List wins beyond NS9 (≥5), zero
  preservation regressions, state-aware stable `cases` skeletons.
- **SFT-ready: NO.** The winning family is variable-dependent
  (`cases {var} <;> simp`); it is **not** a short stable token. Training
  NS25 on it would repeat NS22's structured-tactic null. Per the WX2
  rule, a variable-dependent tactic is wrapper-ready, not raw-SFT-ready.
- **Not "no gain."** The +10 is clear.

The state-aware cases pattern **generalizes from Option to List** as a
robust post-NS9 **wrapper capability**. Combined with WX1, the wrapper
now adds **+29 wins beyond NS9** (Option +19, List +10) with **zero
regressions**, all wrapper-ready, none SFT-ready.

## 9. Recommendation

- **Promote the generalized cases wrapper** (`wx2_option_list_cases_safe`
  = Option + List; Bool/Prod/induction add nothing, so a lean
  Option+List config is the canonical candidate). Kept as an experiment
  config pending genome/router sign-off (WX2 no-permanent-edit
  constraint); the NS9 genome is untouched.
- **Do NOT train NS25.** No SFT-ready family exists; the capability is
  inherently state-aware and belongs in the wrapper.
- **No tactic abstraction layer needed for SFT** — the variable is read
  cheaply from the state at search time; that is precisely why the
  wrapper is the right home and a 60M imitation model is not.
- **Next:** the cases-friendly catalog surface beyond List is thin
  (Option/Bool exhausted, Sum absent, Prod tiny, Multiset a quotient).
  Further wrapper yield would require either a `Multiset.induction_on`-
  aware skeleton (quotient-specific) or mining a different large
  inductive surface; short-token SFT remains gated on a genuinely
  short-token family (none since Int/omega).

## Artifacts

- `evolve/strategy_wrapper.py` (per-type gates/families + notation matcher),
  `scripts/wx2_cases_catalog_audit.py`,
  `scripts/build_wx2_theorem_sets.py`,
  `scripts/wx2_preservation_extract.py`,
  `scripts/wx2_generalized_probe_extract.py`,
  `scripts/wx2_relabel_minimal_cases.py`
- `project/evolve/experiments/wx2/wx2_option_cases_promoted.json`,
  `…/wx2_option_list_cases_safe.json`
- `project/evolve/routing/wx2_theorem_sets.json`
- `project/data/wx2_cases_catalog_audit_meta.json`,
  `…/wx2_preservation_matrix.json`,
  `…/wx2_generalized_cases_probe_meta.json`,
  `…/wx2_minimal_tactic_labels.json`, `…/wx2_minimal_family_pools_meta.json`
- Eval traces/logs under `project/evolve/eval_runs/wx1_wx2*` (gitignored).
