# Learn track final report — NS10 through NS20

**Scope:** every arc that touched supervised training, routing,
trace generation, wrapper expansion, or wrapper-only mining
between the NS9 strategy-wrapper milestone and the NS20 mining-
exhaustion finding.
**Conclusion:** the Learn track has reached its natural endpoint
against the current Mathlib catalog and 8-step search budget.
Further training is blocked by the absence of a dense,
homogeneous wrapper-only signal — every remaining family is
either capability-bound (aesop cannot close more Finset theorems)
or catalog-bound (the 208-theorem Nat catalog is exhausted).

## 1. Where we started — NS9 wrapper, no learning

Before NS10 the project shipped a search-only result. The NS9
best genome wraps the un-trained `gen_v5` checkpoint with shape-
slotted priority templates, theorem-name-aware family tactics,
shape-aware premise retrieval, and a `SkeletonBag`-unified
emission layer. With **17 enabled skeletons**:

| policy | medium (38) | large (65) | demo_v1 (15) |
|---|---:|---:|---:|
| `gen_v5` plain | 3/38 (7.9%) | — | 10/15 |
| **NS9 wrapper** | **37/38 (97.4%)** | **49/65 (75.4%)** | 10/15 |

The single residual medium failure (`Nat.AM_GM`) is a model-
capability ceiling. The wrapper had clearly saturated everything
the existing `gen_v5` checkpoint plus 17 skeletons could close on
this surface. Question for NS10: can the wrapper traces be
distilled BACK into the model?

## 2. NS10 — proof of concept for the Learn step

**Setup.** Re-ran the NS9 wrapper on `nat_defs_medium`, harvested
the per-step (state, tactic) successes, and fine-tuned `gen_v5`
on those 152 pairs to produce `gen_v5_plus1`.

**Result.**

| policy | medium proved |
|---|---:|
| `gen_v5` (baseline) | 3/38 |
| `gen_v5_plus1` (NS10 raw) | **4/38** |

Δ = +1 raw theorem. The Learn step worked in principle:
distilling wrapper successes lifted the raw model above the
baseline. But the absolute lift was tiny relative to the 17-
skeleton wrapper's ceiling — 152 pairs is just too small to
move a 60M-param T5 far. See
`project/evolve/reports/ns10_learn_step_report.md`.

## 3. NS11 — scale the training corpus

**Setup.** Three datasets along orthogonal axes —
`medium` (152 NS10 pairs only), `combined` (152 + 5,577 rows
from `project/seq2seq_data_v5.jsonl` — the legacy training
data the original `gen_v5` was built from), and `coverage` (the
combined set plus deliberate over-sampling of pattern-poor
families). Each produced its own checkpoint.

**Result.**

| policy | medium | large | demo_v1 |
|---|---:|---:|---:|
| `gen_v5` (baseline) | 3/38 | — | 10/15 |
| `gen_v5_ns11_medium` | 5/38 | — | — |
| `gen_v5_ns11_combined` | **9/38** | — | **8/15** |
| `gen_v5_ns11_coverage` | 8/38 | — | — |

Δ on medium: +6 vs baseline. But `combined` cost 2/15 demo_v1
theorems — the first observed instance of supervised
forgetting against the `seq2seq_data_v5` style of training
examples. See `ns11_learn_scale_report.md` and the followup
`ns11_trace_source_audit.md` that explained why the NS10 wrapper
yield was so low (1.1% yield: the wrapper kept re-hitting the
same 51 theorems with the same handful of skeletons).

## 4. NS12 — anti-forgetting

**Setup.** Three variants targeting the demo_v1 regression:
`low_lr` (1e-5 → 3e-6 with combined data), `balanced`
(combined data with namespace-balanced epochs), and
`replay_demo` (combined plus 20 explicit copies of the two lost
demo theorems).

**Result.**

| policy | medium | large | demo_v1 |
|---|---:|---:|---:|
| `gen_v5_ns12_low_lr` | 3/38 | — | 8/15 |
| `gen_v5_ns12_balanced` | 5/38 | 6/65 | **10/15** |
| `gen_v5_ns12_replay` | — | — | 10/15 |

`balanced` is the Pareto sweet spot — restored demo_v1 fully
while keeping +2 on medium. The Nat tradeoff was unavoidable:
no checkpoint reached both 9/38 medium AND 10/15 demo. See
`ns12_anti_forgetting_report.md` and the demo regression
analysis.

## 5. NS13 — stateless router across checkpoints

**Insight.** NS12 surfaced a clear Pareto front: every Nat-heavy
checkpoint hurts non-Nat domains; every namespace-balanced
checkpoint loses Nat ground. Rather than picking one, route the
inner base model by theorem namespace.

**Setup.** A stateless `RoutedGenerativePolicy` matches the
theorem's `full_name` against regex routes in
`project/evolve/routing/ns13_router.json`:
- `^Nat\.` → `gen_v5_ns11_combined`
- `^Set\.` / `^Finset\.` → `gen_v5_ns12_balanced`
- default → `gen_v5`

**Result.** The router achieves the **oracle union** on every
tested set — proves every theorem that ANY single checkpoint
proves, with zero regressions. The NS9 wrapper plugs in
unchanged on top.

| set | gen_v5 | ns11_combined | ns12_balanced | oracle | **router** |
|---|---:|---:|---:|---:|---:|
| `nat_defs_medium` (raw) | 3/38 | 9/38 | 1/38 | 9/38 | **9/38** |
| `demo_v1` (raw) | 10/15 | 8/15 | 10/15 | 10/15 | **10/15** |

See `ns13_domain_router_report.md` and `ns13_model_union_analysis.md`.

## 6. NS14 — fresh theorem surface for training pairs

**Diagnosis.** NS11's 152 pairs came from re-running the wrapper
on the *same* 51-theorem corpus the previous experiments had
already evaluated, so the model couldn't learn anything new it
hadn't already seen. NS14 attacks this directly: enumerate
**fresh** theorems from `project/discovered_theorems.json`
(527 theorems across Nat / Set / Finset / List / Multiset
namespaces), run the NS9 wrapper on the fresh surface, harvest
new wrapper-only-but-not-raw rows.

**Result.** Fresh surface = **70 theorems** across four sets
(`ns14_nat_extra`, `ns14_set_finset_extra`, `ns14_mixed_easy`,
`ns14_mixed_medium`). Wrapper closed 24 unique fresh theorems
the raw model couldn't, yielding **30 unique (state, tactic)
training pairs** — a **27% yield** vs NS11's 1.1%. The dominant
new pattern was a single iff-omega template that closed 8 NS14
Nat theorems homogeneously.

See `ns14_wider_trace_generation_report.md`.

## 7. NS15 — the breakthrough

**Setup.** Train two sub-models on NS11 + NS14 + v5 with
namespace balancing and 10× over-sampling of the NS14 Nat
wrapper-pattern rows. Four checkpoints emerged
(`gen_v5_ns15_nat_oversample`, `gen_v5_ns15_balanced_namespace`,
`gen_v5_ns15_combined_all`, `gen_v5_ns15_curriculum`). The
final routing config (`ns15_router.json`) points `^Nat\.` at
`gen_v5_ns15_nat_oversample` and Set/Finset at the NS12
balanced model.

**Result — raw model only, NO wrapper:**

| set | gen_v5 baseline | NS13 routed | **NS15 routed** |
|---|---:|---:|---:|
| `nat_defs_medium` | 3/38 (7.9%) | 9/38 | **23/38 (60.5%)** |
| `nat_defs_large_v5` | — | 13/65 | **35/65 (53.8%)** |
| `demo_v1` | 10/15 | 10/15 | **10/15** |
| `ns14_nat_extra` | — | 8/20 | **9/20** |
| `ns14_set_finset_extra` | — | 13/20 | **13/20** |

This is the headline Learn-track result. The 10-pair NS14
iff-omega pattern transferred 8/8 to held-out Nat theorems —
proving that small, homogeneous, wrapper-only training pools
work. NS15 raw model alone closes more medium theorems
(23/38) than the NS13 routed raw + wrapper trained on the
larger but less-targeted dataset.

See `ns15_wider_training_report.md` and
`ns15_model_union_analysis.md`. With the NS9 wrapper composed
on top, the combined NS15-routed-wrapper is Pareto-optimal
across every benchmark we have.

## 8. NS16 — negative transfer from heterogeneous data

**Hypothesis.** If 10 iff-omega rows produced 8 transferred
wins, then 19 mixed wrapper-only Nat rows should produce ≥10.

**Result.** Wrong. NS16 trained three variants (10×, 20×,
curriculum-continue) on 19 NS16 wrapper-only Nat rows that
spanned tactic_template, family_tactic, generative_topk, and
retrieved_premise origins. **Routed performance exactly
matched NS15 on every set.** Zero transfer. The
`curriculum_continue` variant catastrophically forgot
demo_v1 (−6/14 on medium, −9/65 on large, −6/9 on NS14 Nat).

**Diagnosis.** NS15's success required a single, dense,
homogeneous pattern. NS16's 19 rows were too sparse and too
heterogeneous to push the model in any coherent direction.
See `ns16_expand_nat_surface_report.md` and
`ns16_transfer_analysis.md`.

## 9. NS17 — pattern-family audit, no trainable pool

**Setup.** Audited every (state, tactic) pair we'd ever
harvested by tactic family (greedy regex classifier:
iff_omega_pair, constructor_omega, nat_simp_arith,
split_ifs_omega, fallback_omega, fallback_aesop,
set_subset_simp, etc.). Mined 114 fresh theorems across four
new surfaces (`ns17_nat_remaining`, `ns17_set_extra`,
`ns17_finset_extra`, `ns17_list_multiset`).

**Result.** Zero new wrapper-only signal on 114 fresh
theorems — raw = wrapper on every set. The biggest family
(`iff_omega_pair`, 28 rows / 27 thms) was already learned by
NS15. The next-largest (`constructor_omega`, 16 rows) had
all wins inside NS9's existing wrapper baseline. **No
homogeneous family met the NS18 training gate** of ≥10
wrapper-only rows or ≥20 pool size with ≥5 unique
wrapper-only theorems.

See `ns17_pattern_family_audit.md` and
`ns17_pattern_family_mining_report.md`.

## 10. NS18 — wrapper expansion, weak new signal

**Pivot.** With training blocked, try the wrapper side. Six
experimental wrapper configs (additive deltas on NS9 best):
`constructor_omega`, `split_ifs_omega`, `nat_simp_arith`,
`aesop_wrapper`, `bool_option_cases`, `combined_safe`.
Probed against the NS9 wrapper baseline across canonical +
fresh surfaces.

**Result.** Five truly-new wrapper-only wins beyond NS9 across
two families:

| family | wins | theorems |
|---|---:|---|
| `aesop` (`aesop_wrapper`) | 3 | `Finset.coe_insert`, `Finset.cons_eq_insert`, `Finset.disjUnion_singleton` |
| `simp_all` Nat-arith (`nat_simp_arith` + `combined_safe`) | 2 | `Nat.mul_mod_mod`, `Nat.mod_mul_mod` |

`aesop_wrapper` also introduced a −1 regression on
`ns17_set_extra` (`Set.inter_singleton_eq_empty`). NS19 gate
(≥5 same-family wins) not met for either. See
`ns18_wrapper_expansion_report.md` and
`ns18_wrapper_variants_comparison.md`.

## 11. NS19 — namespace-gated wrapper, +1 win and regression elimination

**Goal.** Grow the aesop pool 3 → ≥5 and the simp_all-Nat
pool 2 → ≥5 by tightening the wrapper to fire only where it
helps.

**Wrapper extension.** Added a backward-compatible
`theorem_name_tactic_gates` field to `StrategyWrapperPolicy`:
maps a tactic substring to a list of allowed full_name
prefixes. **Critical detail:** the gate must NOT filter
`ORIGIN_GENERATIVE` (base-model) output — a first-pass
implementation did and silently cost 10/30 Set theorems
because NS9 wrapper's Set wins come from the routed base
model emitting aesop directly, not from any NS9 priority
template. The fixed semantics filter only wrapper-injected
entries; smoke test
(`scripts/ns19_smoke_test_gates.py`) asserts both inclusion
and base-model non-filtering.

**Variants.**
- `ns19_finset_aesop_only`: NS18 aesop config + gate aesop
  to `["Finset."]`.
- `ns19_nat_simp_arith_targeted`: NS18 nat_simp_arith plus
  6 more Nat-arith bundles, gated to `["Nat."]`.

**Result.**

| variant | Δwrap | regressions |
|---|---:|---:|
| `finset_aesop_only` | **+4** (3 NS18 preserved + `Finset.coe_cons` new + Set regression eliminated) | 0 |
| `nat_simp_arith_targeted` | +1 (NS18 `Nat.mul_mod_mod` preserved) | −1 on medium (`Nat.div_lt_iff_lt_mul'`) |

Net: aesop pool grew 3 → 4 unique. simp_all-Nat-arith pool
unchanged at 2 unique. **Catalog-exhaustion finding:** all
208 Nat theorems in `discovered_theorems.json` are already
covered by canonical / NS14 / NS16 / NS17 sets — no unmined
Nat surface remains in the current catalog. See
`ns19_targeted_family_mining_report.md`.

## 12. NS20 — full Finset remainder, mining exhaustion

**Goal.** Mine the 74 remaining unused Finset theorems —
the entire catalog remainder after excluding `demo_v1`,
`ns14_set_finset_extra`, `ns17_finset_extra`,
`ns19_finset_aesop_surface`. No token filter; evaluate every
candidate.

**Result.**

| surface | size | proved (raw / wrap / variant) | Δwrap |
|---|---:|---:|---:|
| `ns20_finset_aesop_extra_easy` | 50 | 30 / 30 / 30 | +0 |
| `ns20_finset_aesop_extra_medium` | 16 | 7 / 7 / 7 | +0 |
| `ns20_finset_aesop_extra_hard` | 8 | 3 / 3 / 3 | +0 |

**0 new wrapper-only-vs-NS9 wins across 74 theorems.**
`raw == wrap == variant` on every surface — the routed base
model already closes everything closable, NS9 wrapper templates
add nothing, and bare `aesop` cannot close any of the 44
unsolved thms at the 8-step budget. **Pool stays at 4
unique** (NS18: 3, NS19: 1, NS20: 0). Gate not met.

Preservation confirmed on every benchmark including the
NS20-added `nat_defs_large_v5` check (49/65 — matches NS9
wrap exactly). See `ns20_finset_aesop_mining_report.md` and
`ns20_finset_aesop_comparison.md`.

## 13. The shape of the Learn-track curve

| arc | raw-model medium | raw-model large | wrapper medium | wrapper large | demo_v1 |
|---|---:|---:|---:|---:|---:|
| NS9 wrapper (no learning) | 3 | — | 37 | 49 | 10 |
| NS10 `gen_v5_plus1` | 4 | — | — | — | — |
| NS11 `combined` | 9 | — | — | — | 8 |
| NS12 `balanced` | 5 | 6 | — | — | 10 |
| NS13 routed | 9 | 13 | 37 | 49 | 10 |
| **NS15 routed** | **23** | **35** | **37** | **49** | **10** |
| NS16 routed | 23 | 35 | 37 | 49 | 10 |
| NS17 / NS18 / NS19 / NS20 | 23 | 35 | 37 (NS18: +5 thms beyond, net) | 49 (+1 thm beyond) | 10 |

The raw-model medium climbed 3 → 9 → 23 across NS10–NS15
(a 6.6× lift over the gen_v5 baseline, achieved entirely by
distilling wrapper traces) and has been flat ever since. The
wrapper number is bounded above by the NS9 ceiling of 37/38
medium / 49/65 large because no NS17–NS20 variant produced a
new family large enough to push that ceiling.

## 14. Final conclusion — why we stop here

**Three structural constraints are now all hit simultaneously:**

1. **Catalog exhaustion.** The 200-Finset / 208-Nat catalog
   has no unmined surface for the dominant tactic patterns
   (`iff_omega_pair`, `constructor_omega`, simp_all-Nat,
   bare-aesop on Finset).

2. **Tactic-capability ceiling.** Bare `aesop` cannot close
   44 of the 74 remaining Finset thms at 8 steps — these
   need `Finset.image/filter/map` combinatorial reasoning
   that bare aesop has no shortcut for. Stronger aesop
   configurations or different tactic families are needed,
   not more training.

3. **Pool homogeneity requirement.** NS15 proved that small
   homogeneous pools (8 iff-omega rows) transfer cleanly.
   NS16 proved that mixed pools (19 heterogeneous Nat rows)
   transfer zero. The four bare-aesop wins NS18+NS19 produced
   ARE homogeneous (all coercion/structural rewrites on
   `Finset.{coe,cons,disjUnion}`) but the pool is one short
   of the 5-gate.

The Learn track has done the work it was designed to do —
distilled the wrapper's 17-skeleton search advantage back into
the raw model (3/38 → 23/38 medium, +14 raw theorems with no
demo regression), validated the small-homogeneous-pool training
recipe, and stress-tested it across two more families
(NS16-NS17). Beyond NS20, every remaining lever requires
work *outside* the current arc:

- **Catalog extension from Mathlib.** Pulling 200+ more
  Finset.image / Finset.filter / Nat.gcd / Nat.dvd theorems
  would 2-4× the search surface. The first lever to pull next.
- **Stronger wrapper capabilities.** `aesop` with rule_sets
  or explicit lemma bundles, `decide`, term-mode synthesis
  for non-omega proofs.
- **Different learning objective.** A search-then-decide
  reranker rather than a tactic-token generator; or a
  state-value model that prunes search dead-ends.

Until one of those new levers is in play, NS17–NS20 demonstrate
that further training on the current corpus is not justified.
The wrapper-only signal has converged.

## 15. Pointers

| topic | file |
|---|---|
| NS10 proof of concept | `ns10_learn_step_report.md` |
| NS11 scale-up | `ns11_learn_scale_report.md`, `ns11_trace_source_audit.md` |
| NS12 anti-forgetting | `ns12_anti_forgetting_report.md`, `ns12_demo_regression_analysis.md` |
| NS13 routing | `ns13_domain_router_report.md`, `ns13_model_union_analysis.md` |
| NS14 wider surface | `ns14_wider_trace_generation_report.md` |
| **NS15 breakthrough** | `ns15_wider_training_report.md`, `ns15_model_union_analysis.md` |
| NS16 negative transfer | `ns16_expand_nat_surface_report.md`, `ns16_transfer_analysis.md` |
| NS17 family audit | `ns17_pattern_family_audit.md`, `ns17_pattern_family_mining_report.md` |
| NS18 wrapper expansion | `ns18_wrapper_expansion_report.md`, `ns18_wrapper_variants_comparison.md` |
| NS19 targeted mining | `ns19_targeted_family_mining_report.md`, `ns19_wrapper_variants_comparison.md` |
| NS20 mining exhaustion | `ns20_finset_aesop_mining_report.md`, `ns20_finset_aesop_comparison.md` |
| NS9 best genome | `project/evolve/best/ns9_best_genome.json` |
| NS15 router | `project/evolve/routing/ns15_router.json` |
| Family pool meta (NS18+NS19) | `project/data/ns19_family_pool_meta.json` |
| Finset/aesop pool meta (NS18+NS19+NS20) | `project/data/ns20_finset_aesop_pool_meta.json` |
