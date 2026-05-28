# Learn track — executive summary (NS10 → NS20)

The Learn track started with a search-only result and finished
having distilled most of that result back into a smaller, faster
raw model. It is now bottlenecked on Mathlib catalog coverage,
not on our pipeline.

## The arc in one sentence each

1. **Search produced a wrapper.** NS9: 17 skeletons, no
   learning, lifts `gen_v5` from 3/38 → 37/38 medium and 49/65
   large.
2. **The wrapper produced traces.** NS10 + NS11 + NS14 mined
   ~200 successful (state, tactic) pairs from wrapper runs.
3. **Traces improved the raw model.** NS15 fine-tuned on the
   homogeneous 8-pair NS14 iff-omega pattern and lifted the
   raw model 3/38 → **23/38 medium**, **35/65 large**, with
   `demo_v1` preserved at 10/15.
4. **Routing preserved domains.** NS13 + NS15's stateless
   router composes Nat-specialized and Set/Finset-balanced
   sub-models without regression, achieving the oracle union.
5. **Mining exhausted the current surface.** NS16 / NS17 /
   NS18 / NS19 / NS20 confirmed that no homogeneous training
   pool of ≥5 wrapper-only theorems exists outside the
   already-distilled iff-omega family.

## Headline numbers

| layer | medium (38) | large (65) | demo_v1 (15) |
|---|---:|---:|---:|
| `gen_v5` plain (pre-Learn) | 3 (7.9%) | — | 10 |
| **NS15 routed (raw model only)** | **23 (60.5%)** | **35 (53.8%)** | **10** |
| NS9 wrapper composed on top | **37 (97.4%)** | **49 (75.4%)** | **11** |

Δ vs gen_v5 baseline on medium: **+34 wrapper, +20 raw**.

## Why training stops here

Across NS16–NS20 we mined 114 + 80 + 74 = 268 fresh theorems
and harvested **5 truly-new wrapper-only wins**, split across
two families:
- `aesop`-on-Finset: 4 unique (3 NS18 + 1 NS19 + 0 NS20).
- `simp_all` Nat arithmetic: 2 unique.

Neither family meets the NS21 training gate of ≥5 same-family
homogeneous wins. Three structural constraints all hold
simultaneously:
- The 200-Finset / 208-Nat catalog is exhausted relative to
  the dominant wrapper patterns.
- Bare `aesop` has hit a tactic-capability ceiling on the
  remaining Finset surface (combinatorial `image`/`filter`/
  `map` theorems it cannot close in 8 steps).
- NS15 proved that small homogeneous pools transfer; NS16
  proved that mixed pools do not. We have no more homogeneous
  pools to harvest from the current corpus.

## What unlocks the next arc

In rough order of likely yield:

1. **Catalog extension.** Pull more theorems from Mathlib
   (`Finset.image/filter/map`, `Nat.gcd/dvd`, `Nat.mod`
   chains) to 2-4× the search surface. Most promising.
2. **Stronger wrapper capabilities.** `aesop` with rule_sets
   or explicit lemma bundles; `decide`; term-mode synthesis.
   A wrapper-only probe analogous to NS18.
3. **Different learning objective.** Search-then-decide
   reranker, or a state-value pruner. Outside the current
   tactic-token-generator paradigm.

Until one of those is in play, the wrapper-only signal has
converged and further training on the current corpus is not
justified.

## Pointers

- Full report: `learn_track_final_report_ns10_ns20.md`
- NS15 breakthrough: `ns15_wider_training_report.md`
- NS20 exhaustion: `ns20_finset_aesop_mining_report.md`
- NS9 best genome: `project/evolve/best/ns9_best_genome.json`
- NS15 router: `project/evolve/routing/ns15_router.json`
