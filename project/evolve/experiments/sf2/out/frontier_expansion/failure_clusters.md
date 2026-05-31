# SF2 Failure Clusters

- failures: 20 (genuine 18, junk/unresolved 2) | clusters: 10
- eval results used: ['project/evolve/experiments/sf1/out/real/eval_matrix_results.json']

| priority | cluster_id | size | trace | capability | next |
|---|---|---|---|---|---|
| high | `Set|future_failure_driven_lemma_candidate|iff|all_tactics_errored` | 4 | True | iff decomposition before simp/ext | probe |
| high | `Set|broad_set_aesop_rejected|equality|all_tactics_errored` | 3 | True | needs source-proof inspection | probe |
| high | `Set|rc1_production_stack|membership|all_tactics_errored` | 3 | True | needs source-proof inspection | probe |
| high | `Set|rc1_production_stack|equality|all_tactics_errored` | 3 | True | needs source-proof inspection | probe |
| high | `Set|future_failure_driven_lemma_candidate|membership|all_tactics_errored` | 2 | True | needs source-proof inspection | probe |
| high | `Multiset|wx3_multiset_induction|equality|all_tactics_errored` | 1 | True | Multiset induction routing (avoid on membership/iff goals) | probe |
| high | `Function|future_failure_driven_lemma_candidate|equality|all_tactics_errored` | 1 | True | needs source-proof inspection | probe |
| high | `Set|future_failure_driven_lemma_candidate|equality|all_tactics_errored` | 1 | True | needs source-proof inspection | probe |
| low | `Prop|future_failure_driven_lemma_candidate|equality|unresolved_or_junk` | 1 | True | needs source-proof inspection | ignore |
| low | `Eq|future_failure_driven_lemma_candidate|equality|unresolved_or_junk` | 1 | True | needs source-proof inspection | ignore |

## Representative theorems per cluster
- **high** `Set|future_failure_driven_lemma_candidate|iff|all_tactics_errored` (n=4): Set.ssubset_singleton_iff, Set.ssubset_singleton_iff, Set.antitoneOn_iff_antitone, Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le
- **high** `Set|broad_set_aesop_rejected|equality|all_tactics_errored` (n=3): Set.ite_eq_of_subset_left, Set.subset_singleton_iff_eq, Set.ite_eq_of_subset_right
- **high** `Set|rc1_production_stack|membership|all_tactics_errored` (n=3): Set.ite_inter_self, Set.ite_inter_compl_self, Set.ite_inter
- **high** `Set|rc1_production_stack|equality|all_tactics_errored` (n=3): Set.subset_insert_iff, Set.diff_singleton_subset_iff, Set.union_empty_iff
- **high** `Set|future_failure_driven_lemma_candidate|membership|all_tactics_errored` (n=2): Set.ite_right, Set.ite_empty_right
- **high** `Multiset|wx3_multiset_induction|equality|all_tactics_errored` (n=1): Multiset.toFinset_eq_singleton_iff
- **high** `Function|future_failure_driven_lemma_candidate|equality|all_tactics_errored` (n=1): Function.Injective.nonempty_apply_iff
- **high** `Set|future_failure_driven_lemma_candidate|equality|all_tactics_errored` (n=1): Set.pair_eq_pair_iff
- **low** `Prop|future_failure_driven_lemma_candidate|equality|unresolved_or_junk` (n=1): Prop.compl_singleton
- **low** `Eq|future_failure_driven_lemma_candidate|equality|unresolved_or_junk` (n=1): Eq.subset