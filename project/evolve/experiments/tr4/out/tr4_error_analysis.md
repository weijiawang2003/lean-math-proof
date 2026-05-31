# TR4 error analysis

- model: hgb | budget decision: RANKER_USEFUL_FOR_PROBE_REDUCTION

## False positives (top-50 ranked failures)
- by outcome: {'proof_failed': 35, 'unknown_name': 2}
- by family: {'d2_simp_aesop': 8, 'd1_simp_lemma': 10, 'd2_rw_aesop': 1, 'd2_simp_simpall': 7, 'def_unfold_simp': 6, 'aesop_add_simp': 1, 'd1_aesop': 1, 'd2_ext_simp': 1, 'd1_tofinset_simp': 2}
- unknown-name in top-50: 2

## False negatives
- successes: 23 | ranked below global-100: 10

## Class imbalance
- positives 23/4737 (0.00486)
- positives by namespace: {'Set': 18, 'Prop': 1, 'Finset': 2, 'List': 2}
- positives by family: {'d2_simp_aesop': 3, 'def_unfold_simp': 14, 'd1_aesop': 1, 'd1_simp_lemma': 4, 'aesop_add_simp': 1}

## Leakage / generalization

Strong by-theorem but weak by-namespace PR-AUC => the ranker relies on namespace/family-surface cues that do NOT transfer to a held-out namespace (same gap as TR1). Within-distribution probe reduction is real; cross-namespace transfer is not established.
- by_theorem PR-AUC 0.5216, by_namespace 0.0079, by_cluster 0.3357

## Recommendations

- Probe reduction is usable WITHIN seen namespaces (Set/Finset/List); do NOT assume transfer to an unseen namespace — collect positives there first.
- Collect more positives via RC4B (Set.disjoint_left) and RC4C (d2_simp_aesop) validation before retraining — positives (23) and namespaces with positives (4) are the binding constraint, not model capacity.
- A better/scope-aware retrieval index (cut the unknown-name failures that dominate false positives: 2/50 top-ranked failures) likely helps more than further model tuning.
- Nat has 0 positives despite many failures — retrieval-aware programs do not address Nat arithmetic depth gaps; route those to a depth/search experiment.
