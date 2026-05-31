# SF2 Set Cluster — Cluster-Level Analysis

- clusters: 6 | total probe successes: 10
- gap-type histogram: `{'mixed': 2, 'tactic_gap': 2, 'search_depth_gap': 2}`

| cluster | label | size | sel | solved | best family | gap | action | prio |
|---|---|---|---|---|---|---|---|---|
| `future_failure_dri/iff` | future_failure_driven_lemma_candidate | 4 | 2 | 2 | F_source_inspired | **mixed** | new_probe_family | high |
| `future_failure_dri/membership` | future_failure_driven_lemma_candidate | 2 | 2 | 2 | E_ite_bycases | **tactic_gap** | new_probe_family | high |
| `rc1_production_sta/equality` | rc1_production_stack | 3 | 3 | 2 | F_source_inspired | **mixed** | new_probe_family | high |
| `rc1_production_sta/membership` | rc1_production_stack | 3 | 2 | 2 | F_source_inspired | **search_depth_gap** | new_probe_family | high |
| `broad_set_aesop_re/equality` | broad_set_aesop_rejected | 3 | 2 | 1 | F_source_inspired | **search_depth_gap** | new_probe_family | high |
| `future_failure_dri/equality` | future_failure_driven_lemma_candidate | 1 | 1 | 1 | F_source_inspired | **tactic_gap** | new_probe_family | high |

## Set|future_failure_driven_lemma_candidate|iff|all_tactics_errored
- gap: **mixed** | action: new_probe_family | styles: {'simp_only': 1, 'rw_bridge': 1}
  - `Set.antitoneOn_iff_antitone` solved=True gap=tactic_gap src=simp_only win=`simp [Antitone, AntitoneOn]`
  - `Set.ssubset_singleton_iff` solved=True gap=search_depth_gap src=rw_bridge win=`rw [ssubset_iff_subset_ne, subset_singleton_iff_eq, or_and_right, and_not_self_iff, or_false_iff, and_iff_left_iff_imp] <;> exact fun h => h ▸ (singleton_ne_empty _).symm`

## Set|future_failure_driven_lemma_candidate|membership|all_tactics_errored
- gap: **tactic_gap** | action: new_probe_family | styles: {'simp_only': 2}
  - `Set.ite_empty_right` solved=True gap=tactic_gap src=simp_only win=`simp [Set.ite]`
  - `Set.ite_right` solved=True gap=tactic_gap src=simp_only win=`simp [Set.ite]`

## Set|rc1_production_stack|equality|all_tactics_errored
- gap: **mixed** | action: new_probe_family | styles: {'rw_bridge': 2, 'simp_only': 1}
  - `Set.diff_singleton_subset_iff` solved=True gap=search_depth_gap src=rw_bridge win=`rw [← union_singleton, union_comm] <;> apply diff_subset_iff`
  - `Set.subset_insert_iff` solved=False gap=needs_deeper_search src=rw_bridge win=`None`
  - `Set.union_empty_iff` solved=True gap=tactic_gap src=simp_only win=`simp only [← subset_empty_iff, union_subset_iff]`

## Set|rc1_production_stack|membership|all_tactics_errored
- gap: **search_depth_gap** | action: new_probe_family | styles: {'rw_bridge': 2}
  - `Set.ite_inter` solved=True gap=search_depth_gap src=rw_bridge win=`rw [ite_inter_inter, ite_same]`
  - `Set.ite_inter_self` solved=True gap=search_depth_gap src=rw_bridge win=`rw [Set.ite, union_inter_distrib_right, diff_inter_self, inter_assoc, inter_self, union_empty]`

## Set|broad_set_aesop_rejected|equality|all_tactics_errored
- gap: **search_depth_gap** | action: new_probe_family | styles: {'by_cases_ite_split': 1, 'simp_only': 1}
  - `Set.ite_eq_of_subset_left` solved=True gap=search_depth_gap src=by_cases_ite_split win=`ext x <;> by_cases hx : x ∈ t <;> simp [hx, Set.ite, or_iff_right_of_imp (@h x)]`
  - `Set.subset_singleton_iff_eq` solved=False gap=needs_deeper_search src=simp_only win=`None`

## Set|future_failure_driven_lemma_candidate|equality|all_tactics_errored
- gap: **tactic_gap** | action: new_probe_family | styles: {'subset_antisymm': 1}
  - `Set.pair_eq_pair_iff` solved=True gap=tactic_gap src=subset_antisymm win=`simp [subset_antisymm_iff, insert_subset_iff] <;> aesop`
