# TR1 error analysis

- best model: `sgd`
- within-distribution (LOO) accuracy: **0.877**
- grouped (leave-one-namespace-out) accuracy: **0.386**  → generalization gap **0.491**
- low-support labels: ['PROOF_SEARCH_DEPTH_GAP', 'WX3_MULTISET_INDUCTION']; zero-support: []
- goal-text coverage: 31/57

## Confusions

- MISSING_BRIDGE ↔ NO_CHEAP_ACTION: 1 [{'theorem': 'Multiset.toFinset_eq_singleton_iff', 'true': 'NO_CHEAP_ACTION', 'pred': 'MISSING_BRIDGE_LEMMA_CANDIDATE'}]
- SET_ITE false positives: 0 []

## Name-cue dominance (fraction of label whose name contains the obvious cue)

- `SET_ITE_SIMP`: 1.0
- `MISSING_BRIDGE_LEMMA_CANDIDATE`: 0.842
- `BASELINE_DUPLICATE`: 0.0

## Usable signal

- ['BASELINE_DUPLICATE', 'MISSING_BRIDGE_LEMMA_CANDIDATE', 'NO_CHEAP_ACTION', 'SET_ITE_SIMP', 'SX3_PRODUCTION_SUBSUMED']

## Unreliable labels

- ['PROOF_SEARCH_DEPTH_GAP', 'WX3_MULTISET_INDUCTION']

## Data-collection targets

- WX3_MULTISET_INDUCTION and MX2_TOFINSET_AESOP positives (current support 1 and 0) — mine more verified Multiset-induction / Set.Finite-aesop wins
- PROOF_SEARCH_DEPTH_GAP examples (support 1) — collect more bare-control-closes-but-RC2-missed cases
- non-Set namespaces — grouped generalization is weak; corpus is Set-dominated
- goal-text coverage — only 31/57 have goal text; capture initial goals for all live failures

## Interpretation

- **Leakage/generalization:** High within-distribution OOF accuracy but a large drop under leave-one-namespace-out indicates the model leans on namespace/name-surface cues rather than transferable structure — expected for a 57-example, Set-dominated corpus. Treat as a PILOT signal, not a deployable router.
- **Name vs goal:** SET_ITE / bridge labels are strongly predictable from the theorem NAME alone (see name_cue_dominance_fraction); goal text adds little at this size.