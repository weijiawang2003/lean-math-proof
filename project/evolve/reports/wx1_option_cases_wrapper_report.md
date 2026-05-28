# WX1 — state-aware Option cases-wrapper expansion

**Arc type:** wrapper-capability probe (no training). **Branch:**
`wx1-option-cases-wrapper`. **Router:** `ns24_router`. **Baseline
wrapper:** NS9 best genome. **Experimental configs:**
`project/evolve/experiments/wx1/wx1_option_cases_safe.json` (Option) and
`…/wx1_bool_option_cases_safe.json` (Bool+Option). The NS9 best genome is
**not** modified.

## 0. CX3 recap — why a wrapper, not training

CX3 found a count-only `cases_simp | Option` headroom gate (13 theorems
the routed model fails) but **no clean short-token family**: the
successful form was the compound, per-theorem-variable
`intros <;> cases <var> <;> simp_all`, which NS22 showed will not
memorize at 60M. The redirect: rather than fine-tune, add a **state-aware
wrapper skeleton** that reads the case variable from the proof state and
emits `cases <var> <;> simp_all` at search time — no checkpoint change.

## 1. Design — state-aware Option cases skeletons

`evolve/strategy_wrapper.py` gains an optional, off-by-default
`option_cases_skeletons` block (origin `wrapper_option_cases`):

- **Namespace-gated** — fires only when `full_name` starts with a
  configured prefix (`Option.`); `require_namespace_match` enforces it.
- **State-aware variable extraction** (`_extract_cases_vars`) — reads
  accessible local-context binders whose type *starts with* a gated
  keyword (`Option`/`Bool`), skipping inaccessible daggered names
  (`x✝`) and function types (`g : α → Option β`), preferring variables
  that occur in the goal, capped at `max_vars_per_state`.
- **Per-type templates** — e.g. Option → `cases {var} <;> simp_all`,
  `cases {var} <;> simp`, `intros <;> cases {var} <;> simp_all`.
- **Placement** — first among the wrapper's extra tactics (after the
  base model's top-k, before generic fallbacks); cap slots reserved so
  it never crowds out family/generic entries.

Disabled (block absent/`enabled:false`) ⇒ wrapper output is
byte-identical to NS9. The block is read post-construction in
`eval_rollout_all.py` (mirroring `retrieval_skip_bloating_apply`), so the
`load_strategy_config` return-tuple is untouched.

## 2. Inventory of the CX3 Option cases pool (Stage 1)

`scripts/wx1_inspect_option_cases.py` →
`project/data/wx1_option_cases_inventory.json`. All **13/13**
`cases_simp|Option` headroom theorems expose an extractable Option
context variable that matches the CX3 minimal tactic's variable (e.g.
`isSome_map`→`o`, `bind_congr'`→`x,y`, `map_bind`→`x`). The extractor
recovers exactly the right name.

## 3. Smoke (Stage 4)

- `demo_v1` under WX1 = **11/15** (= NS9 wrapper baseline), with **0**
  `wrapper_option_cases` emissions (no Option theorems) → namespace gate
  holds; no crashes, no bloat.
- `cx3_option_cases_medium` under WX1 = **11/11** (NS9 was 7/11), 3 wins
  directly via `cases x <;> simp_all`.

## 4. Raw vs NS9 vs WX1 matrix (Stage 5)

`scripts/wx1_extract_probe.py` →
`project/data/wx1_option_cases_probe_meta.json`. Baselines A (raw) and B
(NS9) reuse the CX3 runs; C is the WX1 Option config. All `--top-k 8
--max-steps 8` on `ns24_router`.

| set | raw | NS9 | **WX1** | WX1-only | regress |
|---|---:|---:|---:|---:|---:|
| cx3_option_simp_easy | 15 | 15 | **26** | **+11** | 0 |
| cx3_option_cases_medium | 7 | 7 | **11** | **+4** | 0 |
| cx3_bool_option_mixed | 18 | 18 | **22** | **+4** | 0 |
| cx3_bool_decide_easy (Bool control) | 2 | 2 | 2 | 0 | 0 |
| **total** | 42 | 42 | **61** | **+19** | **0** |

- **+19 new Option wins beyond NS9, zero regressions.** 18/19 are
  directly attributed to `wrapper_option_cases`; 1 (`Option.iget_mem`) is
  a `generative_topk` win the cases-split advanced into.
- Raw == NS9 everywhere (CX3 finding reproduced: the NS9 wrapper is a
  no-op on Bool/Option). WX1 is the first config to move the number.
- **Broader Bool+Option config** (`wx1b`) on the Bool surfaces:
  `cx3_bool_decide_easy` 2/2, `cx3_bool_simp_medium` 1/1 — **no gain**.
  Bool cases adds nothing; the win is entirely Option.

## 5. Minimal-tactic relabeling of WX1-only wins (Stage 6)

`scripts/wx1_relabel_minimal_tactics.py` (battery from the initial state,
short-token tactics first, then state-aware `cases <var> <;> …`) →
`project/data/wx1_minimal_family_pools_meta.json`. 18/19 resolved
(`Option.iget_mem` not closed by the battery from initial state).

| minimal family | unique | short-token? | SFT-ready | wrapper-ready |
|---|---:|:---:|:---:|:---:|
| `option_cases_simp \| Option` | **17** | no | ✗ | **✓** |
| `option_cases_simp_all \| Option` | 1 | no | ✗ | ✗ |

**Every resolved win's minimal tactic is `cases <var> <;> simp`** — a
state-aware compound, *not* a short token. Plain `simp`/`simp_all`/
`aesop`/`decide` were tried first and **fail**; the case split is
required. So:

- **No SFT-ready short-token family exists** (`any_sft_gate_met: false`).
- **A homogeneous wrapper-ready family does** — 17 unique
  `option_cases_simp|Option` (`any_wrapper_gate_met: true`).

This is exactly the CX3 prediction: the Option headroom is real but
inherently state-dependent — a *wrapper* capability, not imitation-SFT
material.

## 6. Preservation (Stage 8)

The feature is namespace-gated, so non-Option behaviour is identical to
NS9 **by construction**. Verified two ways:

- **Ranked-list identity** — on Nat/Set/Finset states the WX1 wrapper
  emits a byte-identical ranked list to the feature-off wrapper (0
  `option_cases` emissions); it differs only on Option.
- **Empirical** — WX1 wrapper on the preservation sets, all with **0**
  `wrapper_option_cases` emissions:

  | set | WX1 wrapper | baseline |
  |---|---:|---:|
  | nat_defs_medium | 37/38 | 37/38 (NS9) |
  | ns17_set_extra | 18/30 | 18/30 |
  | ns17_finset_extra | 15/30 | 15/30 |
  | demo_v1 (smoke) | 11/15 | 11/15 (NS9) |

  `nat_defs_large_v5` / `ns14_set_finset_extra` are unchanged by the same
  gating guarantee (no Option theorems → 0 emissions).

NS9/NS22/NS23/NS24/CX3 artifacts, the NS24 router, and all checkpoints
are untouched.

## 7. Decision

- **Promote WX1 as a wrapper capability** — `wx1_option_cases_safe.json`
  adds **+19 Option wins (42→61) with 0 regressions** and is provably
  safe on every other namespace. It is the first thing to move the
  Bool/Option number since the base model. Recommended to fold the
  `option_cases_skeletons` block into the canonical wrapper genome (kept
  here as an experiment config pending sign-off, per the WX1 no-permanent-
  genome-edit constraint).
- **Use it as a mining wrapper** — it reopens the wrapper-only frontier
  on Option (the 17-theorem `option_cases_simp` family is now
  wrapper-solved where raw and NS9 failed).
- **Do NOT train NS25 on it.** The family requires variable-specific
  generated tactics (`cases {var} <;> simp`); it is **not** short-token
  SFT-ready. Imitation-training it at 60M would repeat NS22's
  structured-tactic null. Keep it as wrapper capability.
- **Bool is exhausted** — the broader Bool config added nothing; no
  further Bool/Option short-tactic path remains.

## 8. Next directions

1. **Generalize the state-aware cases skeleton to List/Multiset** — the
   same `cases/rcases <var> <;> simp` mechanism likely unlocks an
   analogous wrapper frontier on the large unprobed List/Multiset
   surface.
2. **Fold WX1 into the canonical wrapper** (genome + router sign-off),
   then re-baseline the full matrix.
3. Short-token SFT remains gated on finding a genuinely short-token
   family; none has appeared since Int/omega (NS22).

## Artifacts

- `evolve/strategy_wrapper.py` (option_cases_skeletons block + `_extract_cases_vars`),
  `eval_rollout_all.py` (reads the block post-construction)
- `scripts/wx1_inspect_option_cases.py`, `scripts/wx1_extract_probe.py`,
  `scripts/wx1_relabel_minimal_tactics.py`, `scripts/wx1_run_eval.sh`
- `project/evolve/experiments/wx1/wx1_option_cases_safe.json`,
  `…/wx1_bool_option_cases_safe.json`
- `project/data/wx1_option_cases_inventory.json`,
  `…/wx1_option_cases_probe_meta.json`,
  `…/wx1_minimal_tactic_labels.json`,
  `…/wx1_minimal_family_pools_meta.json`
- Eval traces/logs under `project/evolve/eval_runs/wx1_*` (gitignored).
