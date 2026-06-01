# RC5V2 Methodology

## Pipeline

1. **Fresh frontier** — TR6 fresh pool ∪ discovered, minus an exclusion registry covering every
   prior-used theorem (TR1–7 known/eval, TR6 wins, RC4D/RC4R known wins, RC5H true-hybrid wins,
   RC5S benchmark). ≥500 strict-fresh candidates; namespaces {Set,Finset,List,Multiset,Nat}
   preferred; Order/Other only as analysis controls.
2. **Eval batch** — stratified ~240 (Set 45 / Finset 55 / List 45 / Multiset 45 / Nat 35 /
   Order-Other 15) with focused dynamic-tail slices (Finset image/subset/biUnion, Multiset
   bind/add/disjoint, List forall/mem/map/filter, Set subset_pair/disjoint/singleton, Nat light).
3. **RC2 baseline** — literal RC2 (rc2_release, ns24, hybrid_evolved, top-k 8, max-steps 8).
4. **RC4 static stage** — RC4R wrapper, same config; additive ⇒ non-gate-firing theorems forced
   RC4 ≡ RC2 (only gate-firing run live).
5. **Dynamic eligibility** — RC4 failed ∧ non-flake ∧ allowed namespace ∧ retrieval-eligible.
6. **Retrieve** — top-20 lemmas (TR3 ∪ SF5 index) per eligible theorem.
7. **Safe B5 plan** — generate candidate programs, **reject off-policy before scoring** (RC5S
   strict grammar), TR4-rank, keep top-5; final off-policy = 0; no B10/B20 mainline.
8. **Safe B5 live** — `rc5s_timeout_safe_runner` (per-theorem process-group kill, hard wall cap,
   checkpoint + deterministic resume).
9. **Attribution** — controls (simp/simp_all/aesop/classical;aesop/exact L/simpa/simp[L]) per win →
   FRESH_TRUE_RC5V2_DELTA / TRUE_RC5V2_DELTA_KNOWN_CONTROL / STATIC_DUPLICATE / BASELINE_DUPLICATE /
   RC2_ALREADY_SOLVED / SOURCE_SPECIFIC_DYNAMIC_WIN / NO_DYNAMIC_WIN / OPEN_FLAKE / TIMEOUT_BOUNDED.
10. **Compare** RC2 vs RC4 vs RC5V2 (=RC4 + safe B5); **safety audit**; **export** examples.

## Invariants

- Dynamic stage runs only on RC4 static failures ⇒ additive over RC4 (0 regressions by construction).
- Strict grammar + process-group-kill runner ⇒ 0 off-policy, no global stalls, bounded timeouts.
- A win counts as TRUE_RC5V2_DELTA only if RC2 AND RC4 failed and bare controls failed; FRESH only
  if the theorem is not in any prior win set.
- B5-only mainline (RC5S showed B10 adds 0 at cost). No ranker retrain (3 positives hurt PR-AUC).
