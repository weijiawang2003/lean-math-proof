# TR3 retrieval results

- confirmed-failure targets: 92 | index 10790 | top-20

## Per-target top-5
### Multiset.toFinset_eq_singleton_iff
- `Multiset.toFinset_card_eq_card_iff_nodup` (1.3954) — lex=0.46; ns=1.00; path=0.75; feat=0.67; name=0.44
- `Multiset.singleton_eq_cons_iff` (1.3852) — lex=0.36; ns=1.00; path=0.50; feat=1.00; name=0.50
- `Multiset.add_singleton_eq_iff` (1.3538) — lex=0.33; ns=1.00; path=0.50; feat=1.00; name=0.50
- `Multiset.map_eq_singleton` (1.3439) — lex=0.36; ns=1.00; path=0.50; feat=1.00; name=0.38
- `Multiset.toFinset_singleton` (1.3365) — lex=0.40; ns=1.00; path=0.75; feat=0.67; name=0.43

### Set.diff_singleton_subset_iff
- `Set.subset_insert_diff_singleton` (1.7222) — lex=0.57; ns=1.00; path=1.00; feat=1.00; name=0.50
- `Set.subset_diff_singleton` (1.5557) — lex=0.46; ns=1.00; path=1.00; feat=0.80; name=0.57
- `Set.insert_diff_singleton` (1.5176) — lex=0.49; ns=1.00; path=1.00; feat=0.80; name=0.38
- `Set.insert_diff_eq_singleton` (1.4714) — lex=0.45; ns=1.00; path=1.00; feat=0.80; name=0.33
- `Set.diff_singleton_sSubset` (1.4633) — lex=0.48; ns=1.00; path=1.00; feat=0.67; name=0.38

### Set.ite_eq_of_subset_left
- `Set.ite_eq_of_subset_right` (1.5918) — lex=0.51; ns=1.00; path=1.00; feat=0.80; name=0.56
- `Set.ite_subset_union` (1.4468) — lex=0.43; ns=1.00; path=1.00; feat=0.80; name=0.33
- `Set.ite_left` (1.4198) — lex=0.47; ns=1.00; path=1.00; feat=0.60; name=0.38
- `Set.union_eq_self_of_subset_left` (1.3517) — lex=0.36; ns=1.00; path=1.00; feat=0.60; name=0.50
- `Set.inter_subset_ite` (1.3223) — lex=0.38; ns=1.00; path=1.00; feat=0.60; name=0.33

### Set.ite_eq_of_subset_right
- `Set.ite_eq_of_subset_left` (1.587) — lex=0.50; ns=1.00; path=1.00; feat=0.80; name=0.56
- `Set.ite_subset_union` (1.4498) — lex=0.43; ns=1.00; path=1.00; feat=0.80; name=0.33
- `Set.ite_inter_of_inter_eq` (1.3785) — lex=0.49; ns=1.00; path=1.00; feat=0.40; name=0.44
- `Set.inter_subset_ite` (1.3747) — lex=0.43; ns=1.00; path=1.00; feat=0.60; name=0.33
- `Set.union_eq_self_of_subset_right` (1.3559) — lex=0.37; ns=1.00; path=1.00; feat=0.60; name=0.50

### Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` (1.7995) — lex=0.62; ns=1.00; path=1.00; feat=1.00; name=0.60
- `Set.monotoneOn_iff_monotone` (1.3414) — lex=0.25; ns=1.00; path=1.00; feat=1.00; name=0.30
- `Set.not_subset_iff_exists_mem_not_mem` (1.2668) — lex=0.36; ns=1.00; path=1.00; feat=0.50; name=0.36
- `Set.not_disjoint_iff` (1.265) — lex=0.31; ns=1.00; path=1.00; feat=0.67; name=0.30
- `Set.ne_univ_iff_exists_not_mem` (1.2318) — lex=0.27; ns=1.00; path=1.00; feat=0.67; name=0.33

### Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` (1.7966) — lex=0.62; ns=1.00; path=1.00; feat=1.00; name=0.60
- `Set.monotoneOn_iff_monotone` (1.3392) — lex=0.25; ns=1.00; path=1.00; feat=1.00; name=0.30
- `Set.not_subset_iff_exists_mem_not_mem` (1.2637) — lex=0.35; ns=1.00; path=1.00; feat=0.50; name=0.36
- `Set.not_disjoint_iff` (1.2624) — lex=0.31; ns=1.00; path=1.00; feat=0.67; name=0.30
- `Set.ne_univ_iff_exists_not_mem` (1.2296) — lex=0.26; ns=1.00; path=1.00; feat=0.67; name=0.33

### Set.pair_eq_pair_iff
- `Set.subset_pair_iff_eq` (1.4894) — lex=0.52; ns=1.00; path=1.00; feat=0.50; name=0.57
- `Set.pair_eq_singleton` (1.474) — lex=0.45; ns=1.00; path=1.00; feat=0.75; name=0.43
- `Set.pair_subset_iff` (1.4736) — lex=0.51; ns=1.00; path=1.00; feat=0.60; name=0.43
- `Set.subset_pair_iff` (1.4513) — lex=0.48; ns=1.00; path=1.00; feat=0.60; name=0.43
- `Set.pairwise_pair` (1.3718) — lex=0.49; ns=1.00; path=0.60; feat=0.75; name=0.29

### Set.ssubset_iff_insert
- `Set.ssubset_insert` (1.5698) — lex=0.55; ns=1.00; path=1.00; feat=0.67; name=0.50
- `Set.ssubset_iff_of_subset` (1.4524) — lex=0.47; ns=1.00; path=1.00; feat=0.67; name=0.38
- `Set.ssubset_iff_sdiff_singleton` (1.4131) — lex=0.37; ns=1.00; path=1.00; feat=0.83; name=0.38
- `Set.ssubset_singleton_iff` (1.3893) — lex=0.38; ns=1.00; path=1.00; feat=0.71; name=0.43
- `Set.lt_iff_ssubset` (1.3876) — lex=0.39; ns=1.00; path=1.00; feat=0.67; name=0.43

### Set.ssubset_iff_sdiff_singleton
- `Set.diff_singleton_sSubset` (1.6578) — lex=0.55; ns=1.00; path=1.00; feat=1.00; name=0.38
- `Set.ssubset_singleton_iff` (1.5518) — lex=0.45; ns=1.00; path=1.00; feat=0.83; name=0.57
- `Set.ssubset_iff_of_subset` (1.4058) — lex=0.39; ns=1.00; path=1.00; feat=0.80; name=0.33
- `Set.lt_iff_ssubset` (1.3861) — lex=0.35; ns=1.00; path=1.00; feat=0.80; name=0.38
- `Set.empty_ssubset_singleton` (1.3766) — lex=0.40; ns=1.00; path=1.00; feat=0.67; name=0.38

### Set.ssubset_singleton_iff
- `Set.ssubset_iff_sdiff_singleton` (1.6388) — lex=0.47; ns=1.00; path=1.00; feat=1.00; name=0.57
- `Finset.ssubset_singleton_iff` (1.5637) — lex=0.91; path=0.50; feat=0.83; name=0.67
- `Set.diff_singleton_sSubset` (1.5533) — lex=0.42; ns=1.00; path=1.00; feat=1.00; name=0.43
- `Set.empty_ssubset_singleton` (1.5456) — lex=0.55; ns=1.00; path=1.00; feat=0.67; name=0.43
- `Multiset.ssubset_singleton_iff` (1.5233) — lex=0.80; path=0.50; feat=1.00; name=0.67

### Set.subset_insert_iff
- `Set.insert_subset_iff` (1.6208) — lex=0.50; ns=1.00; path=1.00; feat=0.80; name=0.67
- `Set.diff_singleton_subset_iff` (1.5429) — lex=0.43; ns=1.00; path=1.00; feat=1.00; name=0.38
- `Set.subset_insert_diff_singleton` (1.4977) — lex=0.39; ns=1.00; path=1.00; feat=1.00; name=0.38
- `Set.insert_subset_insert_iff` (1.4847) — lex=0.36; ns=1.00; path=1.00; feat=0.80; name=0.67
- `Set.subset_insert_iff_of_not_mem` (1.4271) — lex=0.37; ns=1.00; path=1.00; feat=0.80; name=0.44

### Set.subset_ite
- `Set.inter_subset_ite` (1.6303) — lex=0.64; ns=1.00; path=1.00; feat=0.60; name=0.50
- `Set.ite` (1.5697) — lex=0.66; ns=1.00; path=1.00; feat=0.40; name=0.50
- `Set.ite_subset_union` (1.5499) — lex=0.60; ns=1.00; path=1.00; feat=0.50; name=0.50
- `Set.ite_diff_self` (1.4328) — lex=0.51; ns=1.00; path=1.00; feat=0.60; name=0.29
- `Set.ite_inter_self` (1.3725) — lex=0.53; ns=1.00; path=1.00; feat=0.40; name=0.29

### Set.subset_pair_iff_eq
- `Set.Nonempty.subset_pair_iff_eq` (1.8047) — lex=0.86; ns=0.50; path=1.00; feat=0.67; name=0.86
- `Set.subset_pair_iff` (1.6537) — lex=0.56; ns=1.00; path=1.00; feat=0.80; name=0.57
- `Set.pair_eq_pair_iff` (1.5325) — lex=0.52; ns=1.00; path=1.00; feat=0.60; name=0.57
- `Set.pair_subset_iff` (1.496) — lex=0.40; ns=1.00; path=1.00; feat=0.80; name=0.57
- `Set.subset_singleton_iff_eq` (1.4582) — lex=0.44; ns=1.00; path=1.00; feat=0.67; name=0.50

### Set.subset_singleton_iff_eq
- `Set.singleton_subset_iff` (1.5556) — lex=0.38; ns=1.00; path=1.00; feat=1.00; name=0.57
- `Set.subset_singleton_iff` (1.5408) — lex=0.37; ns=1.00; path=1.00; feat=1.00; name=0.57
- `Set.singleton_subset_singleton` (1.523) — lex=0.39; ns=1.00; path=1.00; feat=1.00; name=0.43
- `Set.singleton_eq_singleton_iff` (1.4115) — lex=0.34; ns=1.00; path=1.00; feat=0.75; name=0.57
- `Set.subset_compl_singleton_iff` (1.3998) — lex=0.33; ns=1.00; path=1.00; feat=0.80; name=0.50

### Set.union_empty_iff
- `Set.empty_union` (1.4079) — lex=0.42; ns=1.00; path=1.00; feat=0.60; name=0.50
- `Set.union_empty` (1.4079) — lex=0.42; ns=1.00; path=1.00; feat=0.60; name=0.50
- `Set.iUnion_eq_empty` (1.3145) — lex=0.38; ns=1.00; path=0.75; feat=0.80; name=0.25
- `Set.mem_diff_singleton_empty` (1.2944) — lex=0.31; ns=1.00; path=1.00; feat=0.80; name=0.22
- `Set.sUnion_eq_empty` (1.2885) — lex=0.36; ns=1.00; path=0.75; feat=0.80; name=0.25

### Set.Nonempty.subset_pair_iff_eq
- `Set.subset_pair_iff_eq` (2.0564) — lex=0.93; ns=1.00; path=1.00; feat=0.67; name=0.86
- `Set.subset_pair_iff` (1.6218) — lex=0.55; ns=1.00; path=1.00; feat=0.80; name=0.50
- `Set.pair_eq_pair_iff` (1.5013) — lex=0.51; ns=1.00; path=1.00; feat=0.60; name=0.50
- `Set.pair_subset_iff` (1.467) — lex=0.40; ns=1.00; path=1.00; feat=0.80; name=0.50
- `Set.subset_singleton_iff_eq` (1.4333) — lex=0.43; ns=1.00; path=1.00; feat=0.67; name=0.44

### Set.antitoneOn_iff_antitone
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` (1.2583) — lex=0.43; ns=1.00; path=1.00; feat=0.33; name=0.30
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` (1.2566) — lex=0.43; ns=1.00; path=1.00; feat=0.33; name=0.30
- `Set.antitoneOn_singleton` (1.1546) — lex=0.53; ns=1.00; path=0.75; name=0.29
- `Set.EqOn.congr_antitoneOn` (1.1461) — lex=0.51; ns=0.50; path=0.75; feat=0.50; name=0.25
- `Set.Subsingleton.antitoneOn` (1.0905) — lex=0.63; ns=0.50; path=0.75; name=0.33

### Set.monotoneOn_iff_monotone
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` (1.3693) — lex=0.41; ns=1.00; path=1.00; feat=0.67; name=0.30
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` (1.3677) — lex=0.41; ns=1.00; path=1.00; feat=0.67; name=0.30
- `Set.monotoneOn_singleton` (1.2325) — lex=0.51; ns=1.00; path=0.75; feat=0.25; name=0.29
- `Set.EqOn.congr_monotoneOn` (1.1933) — lex=0.49; ns=0.50; path=0.75; feat=0.67; name=0.25
- `Set.Subsingleton.monotoneOn` (1.1686) — lex=0.61; ns=0.50; path=0.75; feat=0.25; name=0.33

### Set.strictAntiOn_iff_strictAnti
- `Set.antitoneOn_iff_antitone` (0.9698) — lex=0.09; ns=1.00; path=1.00; feat=0.50; name=0.25
- `Set.ext_iff` (0.9629) — lex=0.08; ns=1.00; path=1.00; feat=0.50; name=0.29
- `Set.mem_inter_iff` (0.9479) — lex=0.07; ns=1.00; path=1.00; feat=0.50; name=0.25
- `Set.mem_sep_iff` (0.9403) — lex=0.07; ns=1.00; path=1.00; feat=0.50; name=0.25
- `Set.not_disjoint_iff` (0.9372) — lex=0.06; ns=1.00; path=1.00; feat=0.50; name=0.25

### Set.strictMonoOn_iff_strictMono
- `Set._root_.StrictMonoOn.strictMono` (1.0313) — lex=0.47; ns=0.33; path=0.75; feat=0.33; name=0.43
- `Set.strictMonoOn_singleton` (1.0129) — lex=0.29; ns=1.00; path=0.75; feat=0.25; name=0.29
- `Set.EqOn.congr_strictMonoOn` (0.9999) — lex=0.30; ns=0.50; path=0.75; feat=0.67; name=0.25
- `strictMono_restrict` (0.9695) — lex=0.47; path=0.75; feat=0.67; name=0.14
- `Set.ncard_strictMono` (0.9574) — lex=0.23; ns=1.00; path=0.75; feat=0.25; name=0.29

### Eq.subset
- `Set.eq_of_subset_of_subset` (1.1618) — lex=0.39; path=1.00; feat=1.00; name=0.40
- `Set.le_eq_subset` (1.1431) — lex=0.37; path=1.00; feat=1.00; name=0.40
- `Set.Subset.rfl` (1.1077) — lex=0.38; path=1.00; feat=1.00; name=0.25
- `Set.eq_univ_of_subset` (1.0982) — lex=0.35; path=1.00; feat=1.00; name=0.33
- `Set.sep_eq_of_subset` (1.0826) — lex=0.33; path=1.00; feat=1.00; name=0.33

### Function.Injective.nonempty_apply_iff
- `Function.Surjective.nonempty_preimage` (1.0096) — lex=0.31; ns=0.50; path=0.75; feat=0.67; name=0.22
- `Function.invFunOn_apply_mem` (0.9677) — lex=0.36; ns=1.00; path=0.75; name=0.22
- `Function.invFunOn_apply_eq` (0.9621) — lex=0.36; ns=1.00; path=0.75; name=0.22
- `Function.Injective.subsingleton_image_iff` (0.932) — lex=0.37; ns=0.50; path=0.75; feat=0.25; name=0.33
- `Function.Injective.mem_range_iff_exists_unique` (0.9124) — lex=0.33; ns=0.50; path=0.75; feat=0.33; name=0.27

### Prop.compl_singleton
- `Set.compl_singleton_eq` (1.062) — lex=0.46; path=1.00; feat=0.67; name=0.29
- `Set.compl_ne_eq_singleton` (1.0217) — lex=0.43; path=1.00; feat=0.67; name=0.25
- `Set.mem_compl_singleton_iff` (0.9776) — lex=0.45; path=1.00; feat=0.50; name=0.25
- `Set.subset_compl_singleton_iff` (0.9056) — lex=0.42; path=1.00; feat=0.40; name=0.25
- `compls_singleton` (0.83) — lex=0.39; path=0.50; feat=0.67; name=0.17

### Set.eq_of_inclusion_surjective
- `Set.inclusion` (1.4704) — lex=0.50; ns=1.00; path=1.00; feat=0.67; name=0.33
- `Set.inclusion_eq_id` (1.3517) — lex=0.37; ns=1.00; path=1.00; feat=0.67; name=0.38
- `Set.coe_inclusion` (1.2942) — lex=0.35; ns=1.00; path=1.00; feat=0.67; name=0.25
- `Set.inclusion_right` (1.2924) — lex=0.35; ns=1.00; path=1.00; feat=0.67; name=0.25
- `Set.inclusion_self` (1.2695) — lex=0.33; ns=1.00; path=1.00; feat=0.67; name=0.25

### Set.ite_inter_of_inter_eq
- `Set.ite_inter` (1.5033) — lex=0.51; ns=1.00; path=1.00; feat=0.67; name=0.43
- `Set.ite_inter_inter` (1.5033) — lex=0.51; ns=1.00; path=1.00; feat=0.67; name=0.43
- `Set.ite_right` (1.4254) — lex=0.48; ns=1.00; path=1.00; feat=0.67; name=0.25
- `Set.ite_eq_of_subset_right` (1.4199) — lex=0.53; ns=1.00; path=1.00; feat=0.40; name=0.44
- `Set.ite_inter_self` (1.4075) — lex=0.43; ns=1.00; path=1.00; feat=0.67; name=0.38

### Set.pairwiseDisjoint_filter
- `Set.PairwiseDisjoint` (1.4654) — lex=0.47; ns=1.00; path=0.40; feat=1.00; name=0.50
- `Set.PairwiseDisjoint.mono` (1.3632) — lex=0.57; ns=0.50; path=0.40; feat=1.00; name=0.40
- `Set.PairwiseDisjoint.attach` (1.2892) — lex=0.41; ns=0.50; path=0.75; feat=1.00; name=0.40
- `Set.PairwiseDisjoint.prod` (1.273) — lex=0.48; ns=0.50; path=0.40; feat=1.00; name=0.40
- `Set.pairwiseDisjoint_fiber` (1.2685) — lex=0.32; ns=1.00; path=0.40; feat=1.00; name=0.33

### Set.powerset_singleton
- `Set.mem_singleton` (1.3366) — lex=0.24; ns=1.00; path=1.00; feat=1.00; name=0.33
- `Set.singleton_injective` (1.3114) — lex=0.21; ns=1.00; path=1.00; feat=1.00; name=0.33
- `Set.eq_of_mem_singleton` (1.2836) — lex=0.21; ns=1.00; path=1.00; feat=1.00; name=0.25
- `Set.mem_singleton_of_eq` (1.2778) — lex=0.20; ns=1.00; path=1.00; feat=1.00; name=0.25
- `Set.setOf_eq_eq_singleton` (1.2715) — lex=0.19; ns=1.00; path=1.00; feat=1.00; name=0.29

### Finset.Nonempty.cons_induction
- `Finset.cons_induction` (1.96) — lex=0.92; ns=1.00; path=0.75; feat=0.67; name=0.80
- `Finset.cons_induction_on` (1.5261) — lex=0.59; ns=1.00; path=0.75; feat=0.67; name=0.43
- `Finset.induction` (1.3012) — lex=0.44; ns=1.00; path=0.75; feat=0.50; name=0.40
- `Finset.induction_on` (1.2091) — lex=0.39; ns=1.00; path=0.75; feat=0.50; name=0.29
- `Finset.nonempty_cons` (1.1822) — lex=0.23; ns=1.00; path=0.75; feat=0.67; name=0.50

### Finset.Nontrivial.exists_cons_eq
- `Finset.Nontrivial` (1.4281) — lex=0.39; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Finset.cons` (1.3529) — lex=0.32; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Finset.range_nontrivial` (1.3351) — lex=0.32; ns=1.00; path=0.75; feat=1.00; name=0.25
- `Finset.exists_list_nodup_eq` (1.3083) — lex=0.27; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Finset.exists_mem_eq_inf'` (1.3074) — lex=0.27; ns=1.00; path=0.75; feat=1.00; name=0.33

### Finset.Nontrivial.sdiff_singleton_nonempty
- `Finset.nontrivial_iff_ne_singleton` (1.3615) — lex=0.42; ns=1.00; path=0.75; feat=0.75; name=0.33
- `Finset.singleton_nonempty` (1.3569) — lex=0.39; ns=1.00; path=0.75; feat=0.75; name=0.43
- `Finset.not_nontrivial_singleton` (1.3223) — lex=0.47; ns=1.00; path=0.75; feat=0.50; name=0.38
- `Finset.erase_nonempty` (1.3027) — lex=0.39; ns=1.00; path=0.75; feat=0.75; name=0.25
- `Finset.nontrivial_prod_iff` (1.2996) — lex=0.40; ns=1.00; path=0.75; feat=0.75; name=0.22

### Finset.cons_induction
- `Finset.cons_induction_on` (1.5496) — lex=0.60; ns=1.00; path=0.75; feat=0.67; name=0.50
- `Finset.Nonempty.cons_induction` (1.4988) — lex=0.50; ns=0.50; path=0.75; feat=1.00; name=0.80
- `Finset.induction` (1.3328) — lex=0.45; ns=1.00; path=0.75; feat=0.50; name=0.50
- `Finset.induction_on` (1.2247) — lex=0.39; ns=1.00; path=0.75; feat=0.50; name=0.33
- `Finset.induction_on'` (1.1361) — lex=0.34; ns=1.00; path=0.75; feat=0.40; name=0.33

### Finset.disjoint_filter_filter'
- `Finset.map_filter'` (1.5674) — lex=0.54; ns=1.00; path=0.75; feat=1.00; name=0.29
- `Finset.filter` (1.4471) — lex=0.39; ns=1.00; path=0.75; feat=1.00; name=0.40
- `Finset.disjoint_filter_filter` (1.4167) — lex=0.33; ns=1.00; path=0.75; feat=1.00; name=0.50
- `Finset.disjoint_filter_filter_neg` (1.3305) — lex=0.26; ns=1.00; path=0.75; feat=1.00; name=0.43
- `Finset.filter_filter` (1.2566) — lex=0.22; ns=1.00; path=0.75; feat=1.00; name=0.33

### Finset.eq_singleton_iff_nonempty_unique_mem
- `Set.eq_singleton_iff_nonempty_unique_mem` (1.6874) — lex=0.93; path=0.50; feat=1.00; name=0.78
- `Finset.eq_singleton_iff_unique_mem` (1.6478) — lex=0.61; ns=1.00; path=0.75; feat=0.75; name=0.67
- `Finset.nonempty_iff_eq_singleton_default` (1.645) — lex=0.56; ns=1.00; path=0.75; feat=1.00; name=0.50
- `Finset.singleton_iff_unique_mem` (1.5388) — lex=0.53; ns=1.00; path=0.75; feat=0.75; name=0.56
- `Finset.singleton_nonempty` (1.2745) — lex=0.34; ns=1.00; path=0.75; feat=0.75; name=0.33

### Finset.erase_inj
- `Finset.erase` (1.3606) — lex=0.54; ns=1.00; path=0.75; feat=0.33; name=0.50
- `Finset.singleton_inj` (1.3233) — lex=0.29; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Finset.sdiff_singleton_eq_erase` (1.2967) — lex=0.28; ns=1.00; path=0.75; feat=1.00; name=0.25
- `Finset.mem_erase` (1.2706) — lex=0.37; ns=1.00; path=0.75; feat=0.67; name=0.33
- `Finset.erase_ne_self` (1.2185) — lex=0.33; ns=1.00; path=0.75; feat=0.67; name=0.29

### Finset.erase_nonempty
- `Finset.Nontrivial.erase_nonempty` (1.6673) — lex=0.80; ns=0.50; path=0.75; feat=0.67; name=0.80
- `Finset.nontrivial_prod_iff` (1.2965) — lex=0.32; ns=1.00; path=0.75; feat=1.00; name=0.12
- `Finset.image_nonempty` (1.2964) — lex=0.26; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Finset.pi_nonempty` (1.2926) — lex=0.26; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Finset.erase` (1.2777) — lex=0.46; ns=1.00; path=0.75; feat=0.33; name=0.50

### Finset.induction_on_union
- `Finset.cons_induction_on` (1.2483) — lex=0.38; ns=1.00; path=0.75; feat=0.50; name=0.43
- `Finset.induction_on` (1.2097) — lex=0.36; ns=1.00; path=0.75; feat=0.40; name=0.50
- `Finset.fold_union_empty_singleton` (1.189) — lex=0.18; ns=1.00; path=0.75; feat=1.00; name=0.22
- `Finset.induction` (1.1789) — lex=0.36; ns=1.00; path=0.75; feat=0.40; name=0.40
- `Finset.empty_union` (1.169) — lex=0.25; ns=1.00; path=0.75; feat=0.75; name=0.29

### Finset.insert_erase
- `Finset.erase_insert` (1.5626) — lex=0.45; ns=1.00; path=0.75; feat=1.00; name=0.60
- `Finset.erase_insert_eq_erase` (1.5485) — lex=0.46; ns=1.00; path=0.75; feat=1.00; name=0.50
- `Finset.erase_insert_of_ne` (1.4983) — lex=0.43; ns=1.00; path=0.75; feat=1.00; name=0.43
- `Finset.insert_erase_invOn` (1.4338) — lex=0.35; ns=1.00; path=0.75; feat=1.00; name=0.50
- `Finset.erase_insert_subset` (1.4111) — lex=0.46; ns=1.00; path=0.75; feat=0.67; name=0.50

### Finset.inter_subset_inter
- `Finset.subset_inter` (1.3991) — lex=0.41; ns=1.00; path=0.75; feat=0.67; name=0.60
- `Set.inter_subset_inter` (1.3605) — lex=0.79; path=0.50; feat=0.67; name=0.60
- `Finset.inter_subset_left` (1.3526) — lex=0.40; ns=1.00; path=0.75; feat=0.67; name=0.50
- `Finset.inter_subset_right` (1.3521) — lex=0.40; ns=1.00; path=0.75; feat=0.67; name=0.50
- `Finset.inter_subset_inter_left` (1.3448) — lex=0.39; ns=1.00; path=0.75; feat=0.67; name=0.50

### Finset.mem_disjUnion
- `Finset.disjUnion` (1.372) — lex=0.48; ns=1.00; path=0.75; feat=0.50; name=0.50
- `Finset.card_disjUnion` (1.2358) — lex=0.40; ns=1.00; path=0.75; feat=0.50; name=0.33
- `Finset.singleton_disjUnion` (1.2314) — lex=0.29; ns=1.00; path=0.75; feat=0.75; name=0.33
- `Finset.disjUnion_eq_union` (1.2301) — lex=0.41; ns=1.00; path=0.75; feat=0.50; name=0.29
- `Finset.disjUnion_singleton` (1.2299) — lex=0.29; ns=1.00; path=0.75; feat=0.75; name=0.33

### Finset.pairwise_cons'
- `Finset.pairwise_disjoint_powersetCard` (1.253) — lex=0.23; ns=1.00; path=0.75; feat=1.00; name=0.29
- `Finset.noncommProd` (1.2241) — lex=0.23; ns=1.00; path=0.75; feat=1.00; name=0.20
- `Finset.pairwiseDisjoint_slice` (1.1971) — lex=0.22; ns=1.00; path=0.75; feat=1.00; name=0.14
- `Finset.pairwiseDisjoint_fibers` (1.1685) — lex=0.19; ns=1.00; path=0.75; feat=1.00; name=0.14
- `Finset.pairwiseDisjoint_map_sigmaMk` (1.1603) — lex=0.19; ns=1.00; path=0.75; feat=1.00; name=0.12

### Finset.range_filter_eq
- `Finset.filter_singleton` (1.2381) — lex=0.51; ns=1.00; path=0.75; feat=0.25; name=0.29
- `Finset.filter_eq` (1.0869) — lex=0.30; ns=1.00; path=0.75; feat=0.25; name=0.50
- `Finset.filter_insert` (1.0843) — lex=0.46; ns=1.00; path=0.75; name=0.29
- `Finset.filter_const` (1.0641) — lex=0.44; ns=1.00; path=0.75; name=0.29
- `Finset.card_erase_eq_ite` (1.0617) — lex=0.32; ns=1.00; path=0.75; feat=0.33; name=0.22

### Finset.sizeOf_lt_sizeOf_of_mem
- `List.sizeOf_lt_sizeOf_of_mem` (1.3143) — lex=0.77; path=0.50; feat=0.50; name=0.71
- `Multiset.sizeOf_lt_sizeOf_of_mem` (1.3081) — lex=0.77; path=0.50; feat=0.50; name=0.71
- `Finset.eq_of_mem_singleton` (1.1539) — lex=0.12; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Finset.inter_singleton_of_mem` (1.1485) — lex=0.11; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Finset.singleton_inter_of_mem` (1.1485) — lex=0.11; ns=1.00; path=0.75; feat=1.00; name=0.33

### Finset.ssubset_iff_exists_cons_subset
- `Finset.ssubset_iff_exists_subset_erase` (1.5989) — lex=0.49; ns=1.00; path=0.75; feat=1.00; name=0.56
- `Finset.ssubset_iff_of_subset` (1.5363) — lex=0.47; ns=1.00; path=0.75; feat=1.00; name=0.44
- `Finset.ssubset_iff_subset_ne` (1.4678) — lex=0.40; ns=1.00; path=0.75; feat=1.00; name=0.44
- `Finset.ssubset_cons` (1.426) — lex=0.48; ns=1.00; path=0.75; feat=0.75; name=0.38
- `Finset.ssubset_iff` (1.4208) — lex=0.45; ns=1.00; path=0.75; feat=0.80; name=0.38

### Finset.ssubset_iff_exists_subset_erase
- `Finset.ssubset_iff_exists_cons_subset` (1.5091) — lex=0.48; ns=1.00; path=0.75; feat=0.80; name=0.56
- `Finset.ssubset_iff_of_subset` (1.4108) — lex=0.42; ns=1.00; path=0.75; feat=0.80; name=0.44
- `Finset.ssubset_iff_subset_ne` (1.387) — lex=0.40; ns=1.00; path=0.75; feat=0.80; name=0.44
- `Finset.erase_ssubset` (1.3785) — lex=0.49; ns=1.00; path=0.75; feat=0.60; name=0.38
- `Finset.lt_iff_ssubset` (1.3091) — lex=0.35; ns=1.00; path=0.75; feat=0.80; name=0.33

### Finset.subset_union_elim
- `Finset.union_subset_iff` (1.3039) — lex=0.40; ns=1.00; path=0.75; feat=0.60; name=0.43
- `Finset.subset_union_left` (1.2846) — lex=0.32; ns=1.00; path=0.75; feat=0.75; name=0.43
- `Finset.union_subset` (1.2828) — lex=0.30; ns=1.00; path=0.75; feat=0.75; name=0.50
- `Finset.subset_union_right` (1.2752) — lex=0.31; ns=1.00; path=0.75; feat=0.75; name=0.43
- `Finset.union_subset_union_left` (1.2694) — lex=0.30; ns=1.00; path=0.75; feat=0.75; name=0.43

### List.perm_of_nodup_nodup_toFinset_eq
- `List.toFinset_eq_of_perm` (1.6217) — lex=0.50; ns=1.00; path=0.75; feat=1.00; name=0.62
- `List.toFinset_card_of_nodup` (1.4113) — lex=0.34; ns=1.00; path=0.75; feat=1.00; name=0.44
- `List.toFinset_eq` (1.3974) — lex=0.35; ns=1.00; path=0.75; feat=1.00; name=0.38
- `List.toFinset_inter` (1.3527) — lex=0.35; ns=1.00; path=0.75; feat=1.00; name=0.22
- `List.toFinset_eq_iff_perm_dedup` (1.2926) — lex=0.44; ns=1.00; path=0.75; feat=0.50; name=0.40

### List.toFinset.ext_iff
- `List.toFinset.ext` (1.5054) — lex=0.70; ns=0.50; path=0.75; feat=0.67; name=0.60
- `Finset.ext_iff` (1.503) — lex=0.90; path=0.75; feat=0.67; name=0.50
- `Set.ext_iff` (1.413) — lex=0.87; path=0.50; feat=0.67; name=0.50
- `Finset.ext` (1.2716) — lex=0.77; path=0.75; feat=0.67; name=0.17
- `List.ext_get_iff` (1.2698) — lex=0.40; ns=1.00; path=0.50; feat=0.67; name=0.43

### List.toFinset_eq
- `Multiset.toFinset_eq` (1.4612) — lex=0.89; path=0.75; feat=0.50; name=0.60
- `List.toFinset` (1.3442) — lex=0.46; ns=1.00; path=0.75; feat=0.50; name=0.50
- `List.toFinset_coe` (1.1796) — lex=0.34; ns=1.00; path=0.75; feat=0.50; name=0.33
- `List.toFinset_eq_of_perm` (1.1795) — lex=0.31; ns=1.00; path=0.75; feat=0.50; name=0.43
- `List.toFinset_card_of_nodup` (1.177) — lex=0.36; ns=1.00; path=0.75; feat=0.50; name=0.25

### List.toFinset_eq_empty_iff
- `List.toFinset_nonempty_iff` (1.3668) — lex=0.32; ns=1.00; path=0.75; feat=1.00; name=0.38
- `List.toFinset_nil` (1.3464) — lex=0.47; ns=1.00; path=0.75; feat=0.67; name=0.25
- `List.isEmpty_iff_eq_nil` (1.3073) — lex=0.33; ns=1.00; path=0.50; feat=1.00; name=0.33
- `List.toFinset_eq_iff_perm_dedup` (1.2371) — lex=0.30; ns=1.00; path=0.75; feat=0.67; name=0.44
- `List.mem_toFinset` (1.2322) — lex=0.35; ns=1.00; path=0.75; feat=0.67; name=0.25

### List.toFinset_eq_iff_perm_dedup
- `List.dedup_eq_cons` (1.3818) — lex=0.41; ns=1.00; path=0.50; feat=1.00; name=0.33
- `List.disjoint_toFinset_iff_disjoint` (1.3639) — lex=0.33; ns=1.00; path=0.75; feat=1.00; name=0.33
- `List.toFinset_eq_of_perm` (1.3618) — lex=0.49; ns=1.00; path=0.75; feat=0.50; name=0.44
- `List.dedup_eq_self` (1.3199) — lex=0.34; ns=1.00; path=0.50; feat=1.00; name=0.33
- `List.toFinset.ext` (1.3088) — lex=0.47; ns=0.50; path=0.75; feat=1.00; name=0.25

### List.toFinset_filter
- `Multiset.toFinset_filter` (1.6251) — lex=0.86; path=0.75; feat=1.00; name=0.60
- `List.toFinset` (1.4655) — lex=0.38; ns=1.00; path=0.75; feat=1.00; name=0.50
- `List.toFinset_coe` (1.3472) — lex=0.31; ns=1.00; path=0.75; feat=1.00; name=0.33
- `List.toFinset_val` (1.3197) — lex=0.28; ns=1.00; path=0.75; feat=1.00; name=0.33
- `List.coe_toFinset` (1.2924) — lex=0.25; ns=1.00; path=0.75; feat=1.00; name=0.33

### List.toFinset_nonempty_iff
- `List.toFinset_eq_empty_iff` (1.3637) — lex=0.31; ns=1.00; path=0.75; feat=1.00; name=0.38
- `List.mem_toFinset` (1.2698) — lex=0.38; ns=1.00; path=0.75; feat=0.67; name=0.29
- `List.toFinset` (1.2197) — lex=0.43; ns=1.00; path=0.75; feat=0.33; name=0.40
- `List.disjoint_toFinset_iff_disjoint` (1.2056) — lex=0.27; ns=1.00; path=0.75; feat=0.67; name=0.43
- `List.toFinset_eq_iff_perm_dedup` (1.1772) — lex=0.27; ns=1.00; path=0.75; feat=0.67; name=0.33

### List.toFinset_surj_on
- `List.toFinset` (1.2045) — lex=0.35; ns=1.00; path=0.75; feat=0.50; name=0.40
- `List.toFinset_eq` (1.1063) — lex=0.28; ns=1.00; path=0.75; feat=0.50; name=0.29
- `List.coe_toFinset` (1.0845) — lex=0.26; ns=1.00; path=0.75; feat=0.50; name=0.29
- `List.toFinset_card_of_nodup` (1.0827) — lex=0.28; ns=1.00; path=0.75; feat=0.50; name=0.22
- `List.toFinset_coe` (1.0465) — lex=0.22; ns=1.00; path=0.75; feat=0.50; name=0.29

### Multiset.Nodup.toFinset_inj
- `Multiset.toFinset` (1.5095) — lex=0.45; ns=1.00; path=0.75; feat=1.00; name=0.40
- `Multiset.toFinset_card_of_nodup` (1.5067) — lex=0.46; ns=1.00; path=0.75; feat=1.00; name=0.38
- `Multiset.toFinset_eq` (1.4673) — lex=0.44; ns=1.00; path=0.75; feat=1.00; name=0.29
- `Multiset.Nodup` (1.4476) — lex=0.45; ns=1.00; path=0.50; feat=1.00; name=0.40
- `Multiset.inj_on_of_nodup_map` (1.4417) — lex=0.47; ns=1.00; path=0.50; feat=1.00; name=0.33

### Multiset.toFinset_ssubset
- `Set.Finite.toFinset_ssubset` (1.3473) — lex=0.75; path=0.50; feat=0.80; name=0.50
- `Multiset.zero_ssubset` (1.3226) — lex=0.35; ns=1.00; path=0.50; feat=1.00; name=0.33
- `Multiset.ssubset_singleton_iff` (1.2408) — lex=0.36; ns=1.00; path=0.50; feat=0.80; name=0.29
- `Multiset.toFinset_subset` (1.2342) — lex=0.30; ns=1.00; path=0.75; feat=0.75; name=0.33
- `Multiset.ssubset_cons` (1.2019) — lex=0.33; ns=1.00; path=0.50; feat=0.75; name=0.33

### Multiset.toFinset_subset
- `Set.Finite.toFinset_subset` (1.3004) — lex=0.73; path=0.50; feat=0.75; name=0.50
- `Multiset.toFinset` (1.2885) — lex=0.47; ns=1.00; path=0.75; feat=0.33; name=0.50
- `Multiset.mem_toFinset` (1.2829) — lex=0.38; ns=1.00; path=0.75; feat=0.67; name=0.33
- `Multiset.toFinset_ssubset` (1.2684) — lex=0.33; ns=1.00; path=0.75; feat=0.75; name=0.33
- `Multiset.subset_iff` (1.2083) — lex=0.23; ns=1.00; path=0.50; feat=1.00; name=0.33

### Nat.add_sub_one_le_mul
- `Nat.lt_mul_self_iff` (1.187) — lex=0.59; ns=1.00; path=0.75; name=0.18
- `Nat.succ_add_sub_one` (1.1328) — lex=0.46; ns=1.00; path=0.75; name=0.44
- `Nat.sub_one_add_self` (1.1234) — lex=0.45; ns=1.00; path=0.75; name=0.44
- `Nat.add_succ_sub_one` (1.1141) — lex=0.44; ns=1.00; path=0.75; name=0.44
- `Nat.one_add_le_iff` (1.0513) — lex=0.38; ns=1.00; path=0.75; name=0.44

### Nat.diag_induction
- `Nat.mul_eq_right` (1.1628) — lex=0.19; ns=1.00; path=0.75; feat=1.00; name=0.12
- `Nat.div_le_iff_le_mul_add_pred` (1.1514) — lex=0.19; ns=1.00; path=0.75; feat=1.00
- `Nat.div_lt_one_iff` (1.1501) — lex=0.18; ns=1.00; path=0.75; feat=1.00; name=0.11
- `Nat.add_pos_iff_pos_or_pos` (1.1402) — lex=0.17; ns=1.00; path=0.75; feat=1.00; name=0.11
- `Nat.div_eq_iff_eq_of_dvd_dvd` (1.1393) — lex=0.17; ns=1.00; path=0.75; feat=1.00

### Nat.div_div_div_eq_div
- `Nat.dvd_iff_div_mul_eq` (1.0806) — lex=0.43; ns=1.00; path=0.75; name=0.38
- `Nat.mul_div_eq_iff_dvd` (1.0806) — lex=0.43; ns=1.00; path=0.75; name=0.38
- `Nat.eq_zero_of_dvd_of_div_eq_zero` (1.0377) — lex=0.39; ns=1.00; path=0.75; name=0.38
- `Nat.eq_of_dvd_of_div_eq_one` (1.0369) — lex=0.39; ns=1.00; path=0.75; name=0.38
- `Nat.div_eq_self` (1.033) — lex=0.35; ns=1.00; path=0.75; name=0.50

### Nat.div_eq_iff_eq_of_dvd_dvd
- `Nat.dvd_iff_div_mul_eq` (1.6025) — lex=0.50; ns=1.00; path=0.75; feat=1.00; name=0.56
- `Nat.div_dvd_iff_dvd_mul` (1.5483) — lex=0.48; ns=1.00; path=0.75; feat=1.00; name=0.44
- `Nat.dvd_iff_le_div_mul` (1.4839) — lex=0.43; ns=1.00; path=0.75; feat=1.00; name=0.40
- `Nat.div_ne_zero_iff_of_dvd` (1.4734) — lex=0.39; ns=1.00; path=0.75; feat=1.00; name=0.50
- `Nat.dvd_left_iff_eq` (1.462) — lex=0.39; ns=1.00; path=0.75; feat=1.00; name=0.44

### Nat.div_eq_self
- `Nat.dvd_iff_div_mul_eq` (1.3746) — lex=0.34; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Nat.pred_eq_self_iff` (1.3665) — lex=0.32; ns=1.00; path=0.75; feat=1.00; name=0.38
- `Nat.mul_right_eq_self_iff` (1.3477) — lex=0.31; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Nat.mul_left_eq_self_iff` (1.3444) — lex=0.31; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Nat.div_eq_iff_eq_of_dvd_dvd` (1.3268) — lex=0.29; ns=1.00; path=0.75; feat=1.00; name=0.33

### Nat.div_eq_sub_mod_div
- `Nat.div_mod_eq_mod_mul_div` (1.1751) — lex=0.49; ns=1.00; path=0.75; name=0.50
- `Nat.sub_mod_eq_zero_of_mod_eq` (1.1082) — lex=0.44; ns=1.00; path=0.75; name=0.44
- `Nat.dvd_sub_mod` (1.0547) — lex=0.40; ns=1.00; path=0.75; name=0.38
- `Nat.add_div_eq_of_add_mod_lt` (1.0507) — lex=0.39; ns=1.00; path=0.75; name=0.40
- `Nat.modEq_sub` (1.0378) — lex=0.43; ns=1.00; path=0.75; name=0.25

### Nat.div_le_of_le_mul'
- `Nat.div_lt_iff_lt_mul'` (1.1554) — lex=0.52; ns=1.00; path=0.75; name=0.33
- `Nat.mul_le_of_le_div` (1.0569) — lex=0.37; ns=1.00; path=0.75; name=0.50
- `Nat.not_prime_mul'` (1.0335) — lex=0.43; ns=1.00; path=0.75; name=0.22
- `Nat.eq_zero_of_le_div` (1.0222) — lex=0.35; ns=1.00; path=0.75; name=0.44
- `Nat.add_div_le_add_div` (0.9996) — lex=0.35; ns=1.00; path=0.75; name=0.38

### Nat.div_le_self'
- `Nat.div_lt_self'` (1.297) — lex=0.63; ns=1.00; path=0.75; name=0.43
- `Nat.exists_mul_self'` (1.062) — lex=0.45; ns=1.00; path=0.75; name=0.25
- `Nat.add_div_le_add_div` (0.986) — lex=0.32; ns=1.00; path=0.75; name=0.43
- `Nat.div_mul_div_le_div` (0.9797) — lex=0.31; ns=1.00; path=0.75; name=0.43
- `Nat.div_le_div_right` (0.965) — lex=0.30; ns=1.00; path=0.75; name=0.43

### Nat.div_mul_div_comm
- `Nat.mul_div_mul_comm` (1.1336) — lex=0.40; ns=1.00; path=0.75; name=0.67
- `Nat.div_mul_div_le_div` (1.0789) — lex=0.41; ns=1.00; path=0.75; name=0.43
- `Nat.div_mul_div_le` (1.0592) — lex=0.39; ns=1.00; path=0.75; name=0.43
- `Nat.dvd_div_of_mul_dvd` (1.0382) — lex=0.39; ns=1.00; path=0.75; name=0.38
- `Nat.mul_le_of_le_div` (1.0342) — lex=0.38; ns=1.00; path=0.75; name=0.38

### Nat.div_mul_div_le
- `Nat.mul_le_of_le_div` (1.2595) — lex=0.55; ns=1.00; path=0.75; name=0.57
- `Nat.div_mul_div_le_div` (1.2588) — lex=0.52; ns=1.00; path=0.75; name=0.67
- `Nat.mul_div_le_mul_div_assoc` (1.1732) — lex=0.46; ns=1.00; path=0.75; name=0.57
- `Nat.dvd_iff_le_div_mul` (1.1637) — lex=0.48; ns=1.00; path=0.75; name=0.50
- `Nat.div_le_div_right` (1.1228) — lex=0.46; ns=1.00; path=0.75; name=0.43

### Nat.div_mul_div_le_div
- `Nat.div_mul_div_le` (1.2488) — lex=0.51; ns=1.00; path=0.75; name=0.67
- `Nat.mul_le_of_le_div` (1.2086) — lex=0.50; ns=1.00; path=0.75; name=0.57
- `Nat.mul_div_le_mul_div_assoc` (1.1871) — lex=0.48; ns=1.00; path=0.75; name=0.57
- `Nat.dvd_iff_le_div_mul` (1.178) — lex=0.49; ns=1.00; path=0.75; name=0.50
- `Nat.le_div_iff_mul_le'` (1.113) — lex=0.43; ns=1.00; path=0.75; name=0.50

### Nat.div_pow
- `Nat.maxPowDiv` (1.0249) — lex=0.43; ns=1.00; path=0.75; name=0.20
- `Nat.self_div_pow_eq_ofDigits_drop` (0.9852) — lex=0.35; ns=1.00; path=0.75; name=0.33
- `Nat.div_le_div_right` (0.9669) — lex=0.34; ns=1.00; path=0.75; name=0.29
- `Nat.div_mul_div_le` (0.9666) — lex=0.34; ns=1.00; path=0.75; name=0.29
- `Nat.ofDigits_div_pow_eq_ofDigits_drop` (0.9639) — lex=0.31; ns=1.00; path=0.75; name=0.38

### Nat.dvd_sub'
- `Nat.dvd_iff_dvd_dvd` (0.8799) — lex=0.24; ns=1.00; path=0.75; name=0.33
- `Nat.div_dvd_of_dvd` (0.8498) — lex=0.23; ns=1.00; path=0.75; name=0.29
- `Nat.totient_dvd_of_dvd` (0.8416) — lex=0.22; ns=1.00; path=0.75; name=0.29
- `Nat.dvd_left_iff_eq` (0.8407) — lex=0.23; ns=1.00; path=0.75; name=0.25
- `Nat.dvd_right_iff_eq` (0.8405) — lex=0.23; ns=1.00; path=0.75; name=0.25

### Nat.eq_of_dvd_of_lt_two_mul
- `Nat.dvd_of_forall_prime_mul_dvd` (1.0818) — lex=0.44; ns=1.00; path=0.75; name=0.36
- `Nat.eq_zero_of_dvd_of_lt` (1.0516) — lex=0.36; ns=1.00; path=0.75; name=0.50
- `Nat.lt_of_pow_dvd_right` (0.9801) — lex=0.33; ns=1.00; path=0.75; name=0.36
- `Nat.dvd_div_of_mul_dvd` (0.9793) — lex=0.32; ns=1.00; path=0.75; name=0.40
- `Nat.le_of_lt_add_of_dvd` (0.9584) — lex=0.31; ns=1.00; path=0.75; name=0.36

### Nat.findGreatest_eq_iff
- `Nat.findGreatest_eq_zero_iff` (1.6068) — lex=0.50; ns=1.00; path=0.75; feat=1.00; name=0.57
- `Nat.findGreatest_pos` (1.4677) — lex=0.44; ns=1.00; path=0.75; feat=1.00; name=0.29
- `Nat.find_eq_iff` (1.3693) — lex=0.30; ns=1.00; path=0.75; feat=1.00; name=0.43
- `Nat.add_eq_min_iff` (1.3026) — lex=0.25; ns=1.00; path=0.75; feat=1.00; name=0.38
- `Nat.add_eq_one_iff` (1.3003) — lex=0.25; ns=1.00; path=0.75; feat=1.00; name=0.38

### Nat.findGreatest_mono_left
- `Nat.findGreatest_mono` (1.2482) — lex=0.56; ns=1.00; path=0.75; name=0.50
- `Nat.findGreatest_mono_right` (1.1177) — lex=0.45; ns=1.00; path=0.75; name=0.43
- `Nat.count_mono_left` (1.0601) — lex=0.39; ns=1.00; path=0.75; name=0.43
- `Nat.findGreatest_le` (0.9919) — lex=0.37; ns=1.00; path=0.75; name=0.29
- `Nat.findGreatest_zero` (0.9833) — lex=0.36; ns=1.00; path=0.75; name=0.29

### Nat.findGreatest_mono_right
- `Nat.findGreatest_mono` (1.2267) — lex=0.54; ns=1.00; path=0.75; name=0.50
- `Nat.findGreatest_mono_left` (1.1191) — lex=0.45; ns=1.00; path=0.75; name=0.43
- `Nat.findGreatest_le` (0.9897) — lex=0.37; ns=1.00; path=0.75; name=0.29
- `Nat.findGreatest` (0.9846) — lex=0.33; ns=1.00; path=0.75; name=0.40
- `Nat.findGreatest_zero` (0.9812) — lex=0.36; ns=1.00; path=0.75; name=0.29

### Nat.findGreatest_spec
- `Nat.le_findGreatest` (1.1602) — lex=0.52; ns=1.00; path=0.75; name=0.33
- `Nat.findGreatest_le` (0.9662) — lex=0.33; ns=1.00; path=0.75; name=0.33
- `Nat.findGreatest_zero` (0.9585) — lex=0.32; ns=1.00; path=0.75; name=0.33
- `Nat.findGreatest_of_not` (0.9481) — lex=0.32; ns=1.00; path=0.75; name=0.29
- `Nat.findGreatest_pos` (0.9371) — lex=0.30; ns=1.00; path=0.75; name=0.33

### Nat.find_add
- `Nat.find_le` (1.1664) — lex=0.53; ns=1.00; path=0.75; name=0.33
- `Nat.find_le_iff` (1.0607) — lex=0.44; ns=1.00; path=0.75; name=0.29
- `Nat.find_lt_iff` (1.0573) — lex=0.43; ns=1.00; path=0.75; name=0.29
- `Nat.find_pos` (1.0547) — lex=0.42; ns=1.00; path=0.75; name=0.33
- `Nat.lt_find_iff` (1.0458) — lex=0.42; ns=1.00; path=0.75; name=0.29

### Nat.find_eq_iff
- `Nat.lt_find_iff` (1.5662) — lex=0.50; ns=1.00; path=0.75; feat=1.00; name=0.43
- `Nat.findGreatest_eq_iff` (1.5637) — lex=0.50; ns=1.00; path=0.75; feat=1.00; name=0.43
- `Nat.le_find_iff` (1.5604) — lex=0.49; ns=1.00; path=0.75; feat=1.00; name=0.43
- `Nat.find_eq_zero` (1.5545) — lex=0.49; ns=1.00; path=0.75; feat=1.00; name=0.43
- `Nat.find_pos` (1.517) — lex=0.49; ns=1.00; path=0.75; feat=1.00; name=0.29

### Nat.leRecOn_injective
- `Nat.leRecOn_surjective` (1.2894) — lex=0.65; ns=1.00; path=0.75; name=0.33
- `Nat.leRecOn_trans` (1.1376) — lex=0.50; ns=1.00; path=0.75; name=0.33
- `Nat.leRecOn_succ'` (1.0765) — lex=0.44; ns=1.00; path=0.75; name=0.33
- `Nat.leRecOn_succ` (1.0571) — lex=0.42; ns=1.00; path=0.75; name=0.33
- `Nat.leRecOn_self` (1.0491) — lex=0.41; ns=1.00; path=0.75; name=0.33

### Nat.leRecOn_surjective
- `Nat.leRecOn_injective` (1.2735) — lex=0.64; ns=1.00; path=0.75; name=0.33
- `Nat.leRecOn_trans` (1.1181) — lex=0.48; ns=1.00; path=0.75; name=0.33
- `Nat.leRecOn_succ'` (1.0595) — lex=0.42; ns=1.00; path=0.75; name=0.33
- `Nat.leRecOn_succ` (1.0408) — lex=0.40; ns=1.00; path=0.75; name=0.33
- `Nat.leRecOn_self` (1.0331) — lex=0.40; ns=1.00; path=0.75; name=0.33

### Nat.le_induction
- `Nat.decreasingInduction_trans` (0.9332) — lex=0.35; ns=1.00; path=0.75; name=0.14
- `Nat.squarefree_mul` (0.8845) — lex=0.30; ns=1.00; path=0.75; name=0.14
- `Nat.findGreatest_mono_right` (0.8802) — lex=0.31; ns=1.00; path=0.75; name=0.12
- `Nat.primeFactors_mono` (0.8726) — lex=0.29; ns=1.00; path=0.75; name=0.14
- `Nat.count_strict_mono` (0.8716) — lex=0.30; ns=1.00; path=0.75; name=0.12

### Nat.not_dvd_of_pos_of_lt
- `Nat.not_prime_of_dvd_of_lt` (1.2215) — lex=0.52; ns=1.00; path=0.75; name=0.56
- `Nat.not_prime_of_dvd_of_ne` (1.1096) — lex=0.45; ns=1.00; path=0.75; name=0.40
- `Nat.not_dvd_of_between_consec_multiples` (1.0424) — lex=0.40; ns=1.00; path=0.75; name=0.36
- `Nat.mul_div_lt_iff_not_dvd` (1.0173) — lex=0.37; ns=1.00; path=0.75; name=0.36
- `Nat.not_pos_pow_dvd` (1.016) — lex=0.35; ns=1.00; path=0.75; name=0.44

### Nat.not_two_dvd_bit1
- `Nat.bit1_mod_two` (1.1063) — lex=0.46; ns=1.00; path=0.75; name=0.38
- `Nat.bit1_le` (0.9998) — lex=0.39; ns=1.00; path=0.75; name=0.25
- `Nat.bit1_eq_bit1` (0.9553) — lex=0.34; ns=1.00; path=0.75; name=0.25
- `Nat.bit1_val` (0.9364) — lex=0.32; ns=1.00; path=0.75; name=0.25
- `Nat.bit1_eq_one` (0.9288) — lex=0.32; ns=1.00; path=0.75; name=0.22

### Nat.one_lt_mul_iff
- `Nat.lt_mul_iff_one_lt_left` (1.5311) — lex=0.41; ns=1.00; path=0.75; feat=1.00; name=0.62
- `Nat.lt_mul_iff_one_lt_right` (1.5283) — lex=0.40; ns=1.00; path=0.75; feat=1.00; name=0.62
- `Nat.lt_one_iff` (1.479) — lex=0.37; ns=1.00; path=0.75; feat=1.00; name=0.57
- `Nat.add_eq_one_iff` (1.4655) — lex=0.43; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Nat.lt_one_add_iff` (1.4551) — lex=0.37; ns=1.00; path=0.75; feat=1.00; name=0.50

### Nat.sqrt.iter_sq_le
- `Nat.sqrt.lt_iter_succ_sq` (1.525) — lex=0.63; ns=0.50; path=0.75; feat=1.00; name=0.44
- `Nat.iter_fp_bound` (1.2369) — lex=0.23; ns=1.00; path=0.75; feat=1.00; name=0.22
- `Nat.sqrt_le` (1.0259) — lex=0.36; ns=1.00; path=0.75; name=0.43
- `Nat.sqrt_le_sqrt` (1.0259) — lex=0.36; ns=1.00; path=0.75; name=0.43
- `Nat.le_sqrt` (1.0189) — lex=0.35; ns=1.00; path=0.75; name=0.43

### Nat.sqrt.lt_iter_succ_sq
- `Nat.sqrt.iter_sq_le` (1.5353) — lex=0.64; ns=0.50; path=0.75; feat=1.00; name=0.44
- `Nat.iter_fp_bound` (1.2175) — lex=0.22; ns=1.00; path=0.75; feat=1.00; name=0.20
- `Nat.lt_succ_sqrt` (1.1193) — lex=0.43; ns=1.00; path=0.75; name=0.50
- `Nat.sqrt_mul_sqrt_lt_succ` (1.0856) — lex=0.41; ns=1.00; path=0.75; name=0.44
- `Nat.succ_iterate` (1.0664) — lex=0.06; ns=1.00; path=0.75; feat=1.00; name=0.22

### Set.Nonempty.eq_univ
- `Set.empty_ne_univ` (1.531) — lex=0.46; ns=1.00; path=1.00; feat=1.00; name=0.25
- `Set.nonempty_iff_univ_nonempty` (1.4996) — lex=0.50; ns=1.00; path=1.00; feat=0.67; name=0.43
- `Set.univ_nonempty` (1.4718) — lex=0.32; ns=1.00; path=1.00; feat=1.00; name=0.50
- `Set.mul_univ` (1.4495) — lex=0.46; ns=1.00; path=0.60; feat=1.00; name=0.29
- `Set.smul_univ` (1.4478) — lex=0.46; ns=1.00; path=0.60; feat=1.00; name=0.29

### Set.compl_union_self
- `Set.union_compl_self` (1.7875) — lex=0.59; ns=1.00; path=1.00; feat=1.00; name=0.67
- `Set.compl_union` (1.5386) — lex=0.39; ns=1.00; path=1.00; feat=1.00; name=0.50
- `Set.inter_union_compl` (1.5171) — lex=0.39; ns=1.00; path=1.00; feat=1.00; name=0.43
- `Set.inter_eq_compl_compl_union_compl` (1.4903) — lex=0.38; ns=1.00; path=1.00; feat=1.00; name=0.38
- `Set.union_eq_compl_compl_inter_compl` (1.4903) — lex=0.38; ns=1.00; path=1.00; feat=1.00; name=0.38

### Set.diff_singleton_sSubset
- `Set.ssubset_iff_sdiff_singleton` (1.6055) — lex=0.49; ns=1.00; path=1.00; feat=1.00; name=0.38
- `Set.ssubset_singleton_iff` (1.5137) — lex=0.45; ns=1.00; path=1.00; feat=0.83; name=0.43
- `Set.empty_ssubset_singleton` (1.4236) — lex=0.43; ns=1.00; path=1.00; feat=0.67; name=0.43
- `Set.subset_diff_singleton` (1.3951) — lex=0.35; ns=1.00; path=1.00; feat=0.80; name=0.43
- `Set.mem_diff_singleton` (1.3909) — lex=0.42; ns=1.00; path=1.00; feat=0.60; name=0.43

### Set.disjoint_iff_forall_ne
- `Set.not_disjoint_iff` (1.447) — lex=0.33; ns=1.00; path=1.00; feat=1.00; name=0.38
- `Set.disjoint_left` (1.4196) — lex=0.34; ns=1.00; path=1.00; feat=1.00; name=0.25
- `Set.disjoint_right` (1.4116) — lex=0.34; ns=1.00; path=1.00; feat=1.00; name=0.25
- `Set.eq_univ_iff_forall` (1.3794) — lex=0.28; ns=1.00; path=1.00; feat=1.00; name=0.33
- `Set.forall_sups_iff` (1.3507) — lex=0.30; ns=1.00; path=0.75; feat=1.00; name=0.38

### Set.disjoint_right
- `Finset.disjoint_right` (1.6263) — lex=0.92; path=0.50; feat=1.00; name=0.60
- `Multiset.disjoint_right` (1.6124) — lex=0.91; path=0.50; feat=1.00; name=0.60
- `Set.disjoint_left` (1.5516) — lex=0.45; ns=1.00; path=1.00; feat=1.00; name=0.33
- `Set.disjoint_sdiff_right` (1.5194) — lex=0.37; ns=1.00; path=1.00; feat=1.00; name=0.50
- `Set.disjoint_singleton_right` (1.499) — lex=0.48; ns=1.00; path=1.00; feat=0.67; name=0.50

### Set.disjoint_singleton_left
- `Set.disjoint_singleton_right` (1.5006) — lex=0.51; ns=1.00; path=1.00; feat=0.67; name=0.43
- `Finset.disjoint_singleton_left` (1.4973) — lex=0.91; path=0.50; feat=0.67; name=0.67
- `Set.disjoint_singleton` (1.4863) — lex=0.47; ns=1.00; path=1.00; feat=0.67; name=0.50
- `Set.disjoint_left` (1.3593) — lex=0.48; ns=1.00; path=1.00; feat=0.33; name=0.50
- `Set.disjoint_sdiff_left` (1.2467) — lex=0.38; ns=1.00; path=1.00; feat=0.33; name=0.43

### Set.insert_subset_insert_iff
- `Set.subset_insert_iff_of_not_mem` (1.6418) — lex=0.51; ns=1.00; path=1.00; feat=1.00; name=0.44
- `Set.insert_subset_iff` (1.6184) — lex=0.42; ns=1.00; path=1.00; feat=1.00; name=0.67
- `Finset.insert_subset_insert_iff` (1.5805) — lex=0.86; path=0.50; feat=1.00; name=0.67
- `Set.subset_insert_iff` (1.5681) — lex=0.37; ns=1.00; path=1.00; feat=1.00; name=0.67
- `Set.ssubset_iff_insert` (1.4928) — lex=0.44; ns=1.00; path=1.00; feat=0.80; name=0.43

### Set.nonempty_compl_of_nontrivial
- `Set.nonempty_compl` (1.3026) — lex=0.33; ns=1.00; path=1.00; feat=0.60; name=0.43
- `Set.Nontrivial.nonempty` (1.2583) — lex=0.55; ns=0.50; path=0.75; feat=0.50; name=0.50
- `Set.compl_ne_univ` (1.2215) — lex=0.31; ns=1.00; path=1.00; feat=0.60; name=0.22
- `Set.inter_compl_nonempty_iff` (1.2152) — lex=0.32; ns=1.00; path=1.00; feat=0.50; name=0.33
- `Set.compl_singleton_eq` (1.1985) — lex=0.23; ns=1.00; path=1.00; feat=0.75; name=0.22

