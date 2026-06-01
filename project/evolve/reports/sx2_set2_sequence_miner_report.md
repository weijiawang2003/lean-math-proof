# SX2 / SET2 Short Sequence Program Miner — Decision Report

Branch: `rc1-production-stack` · live LeanDojo · **no commit** · RC1/NS24/NS9 untouched.
SET2 is an **external, off-by-default candidate policy** — not wired into the router.

---

## 1. Executive summary

Mined the 10 successful Set probes from the SF2 deep dive into normalized
proof-sequence templates, built an off-by-default **SET2** gate policy + external
candidate wrapper, and evaluated it live against the 12 deep-dive failures and a
fresh 20-theorem Set holdout, then ran NS23-style minimal-sufficient relabeling.

| stage | result |
|---|---|
| Templates mined | 10 winning probes → 4 families; **only `SET_ITE_SIMP` is gate-worthy** (theorem-agnostic, n=2) |
| SET2 gates implemented | 5 (1 mined + 3 speculative + 1 hard-disabled), all `experimental_off_by_default` |
| Selected-failure eval (12) | live 12/12, RC1-proxy 0/12, **SET2 solved 2** (`ite_empty_right`, `ite_right`), 0 off-gate |
| Fresh holdout eval (20) | live 20/20, RC1-proxy 9/20, **SET2 new wins 3** (`ite_empty`, `ite_empty_left`, `ite_left`), 0 off-gate, 0 regression |
| Minimal relabel | **TRUE_SET2_WIN = 5, all via `SET_ITE_SIMP`**; 1 BASELINE_DUPLICATE; 26 NEEDS_DEEPER_SEQUENCE |
| Off-gate emissions | **0** (selected + holdout + dry Nat/Multiset sanity scan) |

**Recommendation: `PROMOTE_TO_RC2_CANDIDATE`** — for the **narrow `SET_ITE_SIMP`
gate ONLY** (`simp [Set.ite]`), with the 3 speculative gates DISABLED, and gated on
one literal-RC1-wrapper confirmation run (see §9). The n=2 deep-dive lever
generalized to **5 true wins including 3 on a fresh holdout**, with zero off-gate
and zero regression. This mirrors the MX2 precedent (adopt a narrow namespace-gated
battery action, off-by-default).

---

## 2. Why SX2 / SET2 now?

Two consecutive SF2/SF3 investigations (Multiset singleton, Set cluster) returned
**0 missing lemmas**: every failure was an automation gap over EXISTING Mathlib
lemmas. The frontier is best described as *existing lemmas + short proof sequences
RC1 cannot compose/route to*. So the right next layer is not a Lemma Inventor but a
short-sequence program miner: mine successful probes → abstract → gate narrowly →
evaluate off-by-default → relabel → decide. SF3 candidate-lemma queue was 0; the
SF2 gaps were `tactic_gap` / `search_depth_gap`. SX2 targets the one tactic_gap
lever that generalized.

---

## 3. Template mining

Source: `set_cluster_deep_dive/probe_results.json` (10 winning probes).
Gate-worthy ⟺ a **single theorem-agnostic tactic** (no named/theorem-specific
lemmas, no local hypotheses) that solves ≥2 theorems or is reusable by goal shape.

| family | n | theorem-agnostic | named lemmas | local hyp | interesting |
|---|---|---|---|---|---|
| `SET_ITE_SIMP` (`simp [Set.ite]`) | 2 | ✅ | no | no | **YES** (n=2, pure) |
| `SET_RW_BRIDGE` (rw chains) | 4 | ❌ | yes (per-theorem) | no | no |
| `SOURCE_SPECIFIC` (def-unfold / simp-set / +aesop) | 3 | ❌ | yes | no | no |
| `SET_EXT_BYCASES` (`ext<;>by_cases<;>simp`) | 1 | ❌ | yes | **yes** | no |

The family-level counts for `SET_RW_BRIDGE` (n=4) and `SOURCE_SPECIFIC` (n=3) are
**misleading**: each member is a *distinct* theorem-specific tactic (different bridge
lemmas / simp sets / a local hypothesis), so **no single emittable string
generalizes**. Only `SET_ITE_SIMP` is one reusable tactic. Source-copy risk: `low`
for `SET_ITE_SIMP`, `high` for the rest. Baselines (simp/simp_all/aesop/
classical<;>aesop) failed on every mined win (verified live in SF2).

---

## 4. Gate policy

`project/evolve/experiments/sx2/set2_gate_policy.json` — `global_enabled: false`,
`promotion_allowed: false`, `requires_ns23_relabel: true`. Conditions are pure
predicates over (theorem name, goal pp), all ANDed.

| gate | tactic | strength | mined | status |
|---|---|---|---|---|
| `SET_ITE_SIMP` | `simp [Set.ite]` | narrow | 2 | experimental_off_by_default |
| `SET_EXT_SIMP` | `ext x <;> simp` | medium | 0 | speculative (measure only) |
| `SET_SUBSET_ANTISYMM` | `apply Set.Subset.antisymm <;> intro x <;> simp_all` | medium | 0 | speculative |
| `SET_IFF_CONSTRUCTOR` | `constructor <;> intro h <;> simp_all` | medium | 0 | speculative |
| `SET_EXT_BYCASES` | `by_cases h : <PROP> <;> simp_all [h]` | too_broad | 1 | **hard-disabled** (needs binder + local-hyp inference) |

Expected safety: every gate requires `Set` in the name AND no `Multiset` — so a
non-Set / Nat / Multiset surface can never fire (confirmed §8). Known risk: the
speculative gates are near-baseline (`ext<;>simp`, `constructor<;>...`); the relabel
demotes their solves.

---

## 5. Selected-failure evaluation (12 deep-dive cases)

`set2_selected_eval_results.json` — live 12/12, RC1-proxy 0/12 (confirms all known
RC1 failures), **SET2 solved 2**, 0 off-gate, 0 regression.

- `Set.ite_empty_right` → `SET_ITE_SIMP` ✅  (`simp [Set.ite]`)
- `Set.ite_right` → `SET_ITE_SIMP` ✅
- `Set.ite_inter`, `Set.ite_inter_self` → `SET_ITE_SIMP` emitted but **correctly
  failed** (need rw-bridges, not definitional unfold) → honest, not overfit.
- The 3 speculative gates fired on the iff/equality cases and solved **none**.

Reproduces the SF2 finding exactly: `simp [Set.ite]` is the only generalizing lever.

---

## 6. Fresh holdout evaluation (20 cases)

Holdout `set2_holdout_cases.json`: fresh Set/Basic frontier theorems, the 12
deep-dive cases + 3 SF2 deferrals excluded; 8 `ite`-shaped (the `SET_ITE_SIMP`
generalization test). Live 20/20, RC1-proxy 9/20.

- **SET2 new wins over RC1-proxy = 3**, all `SET_ITE_SIMP`:
  `Set.ite_empty`, `Set.ite_empty_left`, `Set.ite_left` (baselines all failed).
- `Set.inclusion_right` solved by `SET_EXT_SIMP` (`ext x <;> simp`) → **not credited**
  (speculative gate, see §7).
- `Set.ite_compl`, `Set.ite_inter_of_inter_eq` (ite-shaped) → `SET_ITE_SIMP` fired
  but failed (need more than unfold) — honest negative.
- 9 RC1-proxy solves (mem_dite*/inclusion*/insert_diff* via aesop/simp_all): SET2
  added nothing, took nothing away (additive).
- Off-gate = 0. Regressions = 0 (additive by construction).

Overfit risk: **low for `SET_ITE_SIMP`** — it wins on theorems not in its mining set
(`ite_empty`, `ite_empty_left`, `ite_left`) via the identical theorem-agnostic
tactic. The win is the goal *shape* (a `Set.ite` reducible by definitional unfold),
not a memorized proof.

### Gate precision (selected + holdout combined)

| gate | fired | solved | TRUE_SET2_WIN |
|---|---|---|---|
| `SET_ITE_SIMP` | 13 | 5 | **5** |
| `SET_EXT_SIMP` | 19 | 1 | 0 |
| `SET_SUBSET_ANTISYMM` | 16 | 0 | 0 |
| `SET_IFF_CONSTRUCTOR` | 13 | 0 | 0 |

`SET_ITE_SIMP`: 5/5 solves are true wins (100% solve-precision). The 3 speculative
gates fired **48× combined for 0 true wins** → pure noise; DISABLE before promotion.

---

## 7. Minimal-sufficient relabeling (NS23)

`set2_minimal_relabel_results.json` — attribution over all 32 rows:

| class | count |
|---|---|
| **TRUE_SET2_WIN** | **5** (all `SET_ITE_SIMP`: `ite_empty_right`, `ite_right`, `ite_empty`, `ite_empty_left`, `ite_left`) |
| BASELINE_DUPLICATE | 1 (`inclusion_right` via speculative `SET_EXT_SIMP`) |
| SOURCE_SPECIFIC | 0 |
| PARSER_ARTIFACT | 0 |
| NEEDS_DEEPER_SEQUENCE | 26 (emitted-but-failed, or RC1 already solved) |

A solve counts as TRUE_SET2_WIN only if RC1-proxy AND all four baselines failed, the
tactic is non-baseline, and the emitting gate is **mined** (support ≥2). Speculative
gates (mined support <2) are demoted to BASELINE_DUPLICATE — their tactics
(`ext<;>simp`, `constructor<;>...`) are generic near-baseline reductions very likely
within RC1's real top-11+ battery (the 4-tactic baseline proxy is narrower than RC1).

---

## 8. Sanity / off-gate check

`set2_sanity_check.json` — dry gate-only scan (gates are pure predicates;
deterministic, no Lean needed). Surfaces: `demo_v1`, `nat_defs_medium`,
`multiset_preservation` (12 synthetic Nat/Bool/List/Multiset samples).

- SET2 emissions on non-Set surfaces: **0** (both production-default AND force-enabled).
- Off-gate emissions: **0**.
- Positive Set controls (`Set.ite_right`→`SET_ITE_SIMP`, `Set.union_empty_iff`→
  `SET_IFF_CONSTRUCTOR`) fire only when force-enabled; production default = 0.
- `sanity_ok = true`.

Live eval on canonical surfaces was intentionally NOT run (expensive, unnecessary —
the gate logic cannot fire without `Set` in the name).

---

## 9. Promotion decision

### `PROMOTE_TO_RC2_CANDIDATE` — narrow `SET_ITE_SIMP` gate only

RC2 promotion gate vs evidence:

| requirement | status |
|---|---|
| positive delta over RC1 on fresh frontier | ✅ +3 fresh holdout wins (`ite_empty`, `ite_empty_left`, `ite_left`) |
| zero regressions | ✅ 0 (additive / off-by-default) |
| zero off-gate emissions | ✅ 0 (selected, holdout, sanity) |
| minimal-sufficient attribution | ✅ 5/5 `SET_ITE_SIMP` solves = TRUE_SET2_WIN |
| deterministic reproduction | ✅ deterministic gate + single tactic; re-runs identical |

**Conditions before actual RC2 inclusion:**
1. **DISABLE the 3 speculative gates** (`SET_EXT_SIMP`, `SET_SUBSET_ANTISYMM`,
   `SET_IFF_CONSTRUCTOR`): 48 firings, 0 true wins — pure near-baseline noise.
   Keep `SET_EXT_BYCASES` hard-disabled. Promote `SET_ITE_SIMP` alone.
2. **Literal-RC1 confirmation run.** Wins here are vs a 4-tactic baseline *proxy*
   (justified for Set/Basic — WX3 Multiset-induction and MX2 Set.Finite-aesop do not
   apply — and the holdout theorems are SF1 RC1-frontier rows). Confirm the 5 wins
   against the actual `rc1_production_wrapper.json` battery before merging.

This is the honest ceiling of the evidence: one narrow, mined, 100%-precision gate
that genuinely generalized. It is **not yet promoted** (candidate), and RC1 remains
untouched.

---

## 10. Next step

`SET_ITE_SIMP` generalized → run a **larger `Set.ite` frontier sweep** to size the
total addressable win set (every `Set.ite_*` / definitional-unfold theorem RC1 fails)
and execute the literal-RC1 confirmation. The source-inspired rw-bridge wins
(`ite_inter`, `ite_inter_self`, `diff_singleton_subset_iff`, `ssubset_singleton_iff`)
remain theorem-specific search-depth gaps → these motivate a future **SX3 depth-
limited sequence search**, NOT wrapper promotion. The speculative ext/iff/subset
gates are dead ends as wrappers (training/diagnostic data only).

---

## 11. Protected-file confirmation

- `git diff --stat HEAD -- project/evolve/experiments/rc1/rc1_production_wrapper.json
  project/evolve/routing/ns24_router.json` → **empty** (untouched).
- NS9 genome/checkpoints, REL1 / RC1 release artifacts: untouched.
- `git status --short`: only new `??` SX2 scripts/artifacts (+ pre-existing SF1/SF2/SF3
  files and ` M README.md`).
- **No commit made.** All changes left in the working tree.
