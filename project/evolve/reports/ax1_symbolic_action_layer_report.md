# AX1 — symbolic action abstraction layer

**Arc type:** design + offline-data + evaluator-integration prototype
(no neural training). **Branch:** `ax1-symbolic-action-layer`. **Router:**
`ns24_router`. **Baseline wrapper:** NS9 genome (unmodified).

## 0. Motivation (WX1/WX2)

The state-aware cases wrapper adds **+29 wins beyond NS9** (Option +19,
List +10), zero regressions — but the winning tactics
(`cases <var> <;> simp_all`) are **not SFT-ready**: the variable is read
from the proof state, so the raw string differs per theorem (`cases o`,
`cases xs`, `cases l`, …). A 60M imitation model cannot reliably emit the
right variable name. AX1 introduces a **symbolic action** layer that
factors the variable out: the learnable label is state-independent
(`CASES_SIMP[List,simp_all]`) and the wrapper instantiates the variable
from the live state at apply time.

## 1. Schema (`project/evolve/symbolic_actions.py`)

`SymbolicAction(action_type, var_type, simp_mode, namespace_gate,
max_vars, priority, family_source)` with:
- `action_type ∈ {CASES_SIMP, INDUCTION_SIMP}`,
  `var_type ∈ {Option, List, Bool}`, `simp_mode ∈ {simp, simp_all, decide}`
- stable `action_id` (e.g. `CASES_SIMP[List,simp_all]`),
  `to_dict`/`from_dict`, `validate`, `gate_allows(full_name)`
- `instantiate_symbolic_action(action, state_pp, full_name) ->
  [(tactic, family_source, action_id)]`

## 2. State extraction (`project/evolve/state_vars.py`)

`extract_state_variables(state_pp) -> [StateVar(name, type_pp,
coarse_type, is_hypothesis)]`, coarse type ∈ {Option, List, Bool, Nat,
Int, unknown}. Conservative: parses only the local context, excludes
function types (`g : α → Option β`), inaccessible (`✝`) names, and
Prop-typed hypotheses; returns `[]` when uncertain. `vars_of_type`
returns goal-preferring, accessible data variables of a coarse type.

## 3. Instantiation (Stage 3)

`CASES_SIMP[Option,simp_all]` + `o : Option α` ⟶ `cases o <;> simp_all`;
`CASES_SIMP[List,simp_all]` + `l : List α` ⟶ `cases l <;> simp_all`;
`INDUCTION_SIMP[List,simp_all]` ⟶ `induction l <;> simp_all`;
`CASES_SIMP[Bool,decide]` ⟶ `cases b <;> decide`. No hardcoded names;
capped at `max_vars`; deduped; namespace-gated.

## 4. Wrapper integration (Stage 4)

`StrategyWrapperPolicy` gained an off-by-default `symbolic_actions`
block. Emitted tactics carry `tactic_origin = wrapper_symbolic_action`,
`tactic_family_source = action.family_source`, and the action id in the
`template_source` slot. With the block absent/`enabled:false` the wrapper
is **byte-identical to NS9** (verified: disabled ⇒ no symbolic entries;
Nat/Set/Finset ⇒ gated out). `eval_rollout_all` reads the block
post-construction, leaving the `load_strategy_config` tuple untouched.

## 5. Symbolic config (Stage 5)

`project/evolve/experiments/ax1/ax1_symbolic_option_list_cases.json` =
NS9 genome + 5 symbolic actions reproducing WX2 Option+List:
`CASES_SIMP[Option,simp_all|simp]` (gated Option),
`CASES_SIMP[List,simp_all|simp]` + `INDUCTION_SIMP[List,simp_all]`
(gated List).

## 6. Equivalence to WX2 (Stage 6)

`scripts/ax1_equivalence_extract.py` →
`project/data/ax1_symbolic_equivalence_meta.json`. WX2 custom cases
wrapper vs AX1 symbolic wrapper, same sets, `ns24_router`:

| set | ns class | WX2 | AX1 | Δ | symbolic emit |
|---|---|---:|---:|---:|---:|
| wx2_list_cases_easy | List | 9 | 9 | +0 | 164 |
| wx2_list_cases_medium | List | 17 | 17 | +0 | 60 |
| cx3_option_simp_easy | Option | 26 | 26 | +0 | 15 |
| cx3_option_cases_medium | Option | 11 | 11 | +0 | 6 |
| cx3_bool_option_mixed | Option | 22 | 22 | +0 | 14 |
| demo_v1 | mixed-Nat | 11 | 11 | +0 | 0 |

**AX1 reproduces WX2 exactly: Δ=0 on every set, zero regressions, zero
symbolic emissions outside the gated Option/List namespaces** (demo
control = 0). The symbolic action layer is a clean, behavior-identical
re-expression of the WX1/WX2 custom cases wrappers.

## 7. Symbolic-label dataset prototype (Stage 7)

`scripts/ax1_build_symbolic_label_dataset.py` →
`project/data/ax1_symbolic_label_dataset_meta.json`. From the WX1+WX2
wins: **27 labelled examples, all variable-dependent raw tactics,
collapsing to just 4 symbolic labels**:

| symbolic label | examples |
|---|---:|
| `CASES_SIMP[Option,simp]` | 17 |
| `CASES_SIMP[List,simp]` | 6 |
| `CASES_SIMP[List,simp_all]` | 3 |
| `CASES_SIMP[Option,simp_all]` | 1 |

(WX1 18, WX2 9.) This is the AX2 training target: a model predicts the
state-independent label `CASES_SIMP[List,simp_all]`, not the
variable-bearing string `cases xs <;> simp_all`. 27 distinct raw strings
become 4 stable, learnable classes.

## 8. Decision

**AX1 reproduced WX2 cleanly (max |Δ| = 0) ⇒ recommend AX2
symbolic-action training.** The abstraction is validated end-to-end:

- The symbolic vocabulary is tiny and stable (4 labels cover all 27
  current wins), so it is a tractable SFT target (classifier or
  constrained generator over action ids), unlike the 27 variable-bearing
  raw strings.
- The wrapper already instantiates symbolic actions from the state with
  zero behavior change, so a trained symbolic predictor plugs into the
  existing apply path with no further engineering.
- This is the **bridge** the project lacked: short stable tactics
  (`omega`/`aesop`) were already SFT-ready (NS15/NS22); variable-dependent
  tactics were wrapper-only (WX1/WX2); a symbolic action makes the
  variable-dependent family SFT-ready *as a label* while keeping
  instantiation in the wrapper.

### Recommendation

1. **AX2: train a symbolic-action predictor.** Target the 4-label
   vocabulary (extensible). Two viable forms: (a) a per-state classifier
   "which symbolic action (if any) applies here", or (b) augment the
   generative model's vocabulary with action-id tokens and teach it to
   emit them; the wrapper instantiates. Use the AX1 dataset schema as the
   label source; expand it by re-mining List/Option/Bool/Sum/Prod cases
   wins under the symbolic wrapper to grow beyond 27 examples (the
   current set is small — grow it before training).
2. **Promote the AX1 symbolic config as the canonical cases wrapper** —
   it is equivalent to WX2 and strictly more general/maintainable (one
   typed schema instead of per-namespace one-off blocks).
3. **No tactic abstraction beyond this is needed yet** — the symbolic
   layer cleanly captures the cases/induction family; broaden the action
   set only when a new state-dependent family appears.

## Artifacts

- `project/evolve/symbolic_actions.py`, `project/evolve/state_vars.py`,
  `evolve/strategy_wrapper.py` (symbolic_actions block),
  `eval_rollout_all.py` (reads the block),
  `scripts/ax1_equivalence_extract.py`,
  `scripts/ax1_build_symbolic_label_dataset.py`
- `project/evolve/experiments/ax1/ax1_symbolic_option_list_cases.json`
- `project/data/ax1_symbolic_equivalence_meta.json`,
  `project/data/ax1_symbolic_label_dataset_meta.json`
- `project/evolve/reports/ax1_symbolic_equivalence_report.md`
- Eval traces/logs under `project/evolve/eval_runs/wx1_ax1_*` (gitignored).
