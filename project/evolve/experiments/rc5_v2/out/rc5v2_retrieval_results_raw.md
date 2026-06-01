# TR6 retrieval results

- targets: 149 | index 10790 | top-20 | coverage 149/149 | avg best score 1.44

## Per-target top-3 (first 25 targets)
### Set.InjOn.image_diff_subset
- `Set.subset_image_diff` (1.4883) — lex=0.48; ns=1.00; path=0.75; feat=0.75; name=0.57
- `Set.subset_diff_singleton` (1.3875) — lex=0.34; ns=1.00; path=0.75; feat=1.00; name=0.38
- `Set.diff_subset_diff` (1.3806) — lex=0.41; ns=1.00; path=0.75; feat=0.75; name=0.43
- goal_defs: ['Set.InjOn']

### Set.InjOn.mem_of_mem_image
- `Set.exists_image_eq_injOn_of_subset_range` (1.3043) — lex=0.33; ns=1.00; path=1.00; feat=0.67; name=0.36
- `Set.InjOn.mem_image_iff` (1.2929) — lex=0.52; ns=0.50; path=1.00; feat=0.50; name=0.50
- `Set.InjOn.image_inter` (1.2644) — lex=0.46; ns=0.50; path=1.00; feat=0.67; name=0.38
- goal_defs: ['Set.InjOn']

### Set.SurjOn.image_invFunOn_image_of_subset
- `Set.SurjOn.image_invFunOn_image` (1.3301) — lex=0.56; ns=0.50; path=1.00; feat=0.50; name=0.50
- `Set.SurjOn.bijOn_subset` (1.3056) — lex=0.48; ns=0.50; path=1.00; feat=0.75; name=0.33
- `Set.surjOn_of_subsingleton'` (1.2272) — lex=0.23; ns=1.00; path=1.00; feat=0.75; name=0.33
- goal_defs: ['Set.SurjOn', 'Finset.Nonempty', 'Function.invFunOn']

### Set.biInter_subset_of_mem
- `Set.biInter_subset_biInter_left` (1.3919) — lex=0.41; ns=1.00; path=1.00; feat=0.67; name=0.38
- `Set.biInter_singleton` (1.3153) — lex=0.37; ns=1.00; path=1.00; feat=0.67; name=0.25
- `Set.biInter_mono` (1.3075) — lex=0.37; ns=1.00; path=1.00; feat=0.67; name=0.25

### Set.compl_range_subset_kernImage
- `Set.kernImage_preimage_eq_iff` (1.475) — lex=0.58; ns=1.00; path=1.00; feat=0.60; name=0.20
- `Set.kernImage_compl` (1.4149) — lex=0.49; ns=1.00; path=1.00; feat=0.50; name=0.43
- `Set.kernImage_eq_compl` (1.3911) — lex=0.48; ns=1.00; path=1.00; feat=0.50; name=0.38
- goal_defs: ['Set.kernImage', 'Set.range', 'Finset.range', 'Multiset.range']

### Set.diff_singleton_sSubset
- `Set.ssubset_singleton_iff` (1.5481) — lex=0.49; ns=1.00; path=1.00; feat=0.83; name=0.43
- `Set.ssubset_iff_sdiff_singleton` (1.5433) — lex=0.43; ns=1.00; path=1.00; feat=1.00; name=0.38
- `Set.empty_ssubset_singleton` (1.4836) — lex=0.49; ns=1.00; path=1.00; feat=0.67; name=0.43

### Set.diff_singleton_subset_iff
- `Set.subset_diff_singleton` (1.6647) — lex=0.49; ns=1.00; path=1.00; feat=1.00; name=0.57
- `Set.subset_insert_diff_singleton` (1.6012) — lex=0.53; ns=1.00; path=1.00; feat=0.80; name=0.50
- `Set.singleton_subset_iff` (1.5944) — lex=0.42; ns=1.00; path=1.00; feat=1.00; name=0.57

### Set.exists_image_eq_injOn_of_subset_range
- `Set.image_preimage_eq_of_subset` (1.4381) — lex=0.36; ns=1.00; path=0.75; feat=1.00; name=0.45
- `Set.subset_range_iff_exists_image_eq` (1.4254) — lex=0.46; ns=1.00; path=0.75; feat=0.67; name=0.55
- `Set.InjOn.exists_subset_injOn_subset_range_eq` (1.4197) — lex=0.41; ns=0.50; path=1.00; feat=1.00; name=0.60
- goal_defs: ['Set.InjOn', 'Set.range', 'Finset.range', 'Multiset.range']

### Set.image_iInter
- `Set.iInter` (1.7185) — lex=0.57; ns=1.00; path=1.00; feat=1.00; name=0.50
- `Set.image_iInter_subset` (1.6385) — lex=0.49; ns=1.00; path=1.00; feat=1.00; name=0.50
- `Set.iInter_subset` (1.4977) — lex=0.40; ns=1.00; path=1.00; feat=1.00; name=0.33

### Set.preimage_invFun_of_mem
- `Set.preimage_invFun_of_not_mem` (1.6376) — lex=0.69; ns=1.00; path=1.00; feat=0.40; name=0.62
- `Set.range_extend` (1.1002) — lex=0.23; ns=1.00; path=1.00; feat=0.60; name=0.11
- `Set.nonempty_of_nonempty_preimage` (1.07) — lex=0.26; ns=1.00; path=0.75; feat=0.40; name=0.38
- goal_defs: ['Set.range', 'Finset.Nonempty', 'Finset.range', 'Multiset.range']

### Set.preimage_invFun_of_not_mem
- `Set.preimage_invFun_of_mem` (1.6187) — lex=0.67; ns=1.00; path=1.00; feat=0.40; name=0.62
- `Set.preimage_const_of_not_mem` (1.2442) — lex=0.27; ns=1.00; path=0.75; feat=0.67; name=0.56
- `Set.exists_eq_const_of_preimage_singleton` (1.1985) — lex=0.19; ns=1.00; path=0.75; feat=1.00; name=0.25
- goal_defs: ['Finset.Nonempty']

### Set.ssubset_iff_sdiff_singleton
- `Set.diff_singleton_sSubset` (1.6915) — lex=0.58; ns=1.00; path=1.00; feat=1.00; name=0.38
- `Set.ssubset_singleton_iff` (1.6452) — lex=0.54; ns=1.00; path=1.00; feat=0.83; name=0.57
- `Set.empty_ssubset_singleton` (1.4653) — lex=0.49; ns=1.00; path=1.00; feat=0.67; name=0.38

### Set.ssubset_singleton_iff
- `Set.ssubset_iff_sdiff_singleton` (1.727) — lex=0.56; ns=1.00; path=1.00; feat=1.00; name=0.57
- `Set.diff_singleton_sSubset` (1.6277) — lex=0.50; ns=1.00; path=1.00; feat=1.00; name=0.43
- `Set.empty_ssubset_singleton` (1.5592) — lex=0.56; ns=1.00; path=1.00; feat=0.67; name=0.43

### Set.subset_biUnion_of_mem
- `Finset.subset_biUnion_of_mem` (1.4407) — lex=0.80; path=0.50; feat=0.75; name=0.71
- `Set.biUnion_subset_biUnion_left` (1.3855) — lex=0.37; ns=1.00; path=1.00; feat=0.75; name=0.38
- `Set.biUnion_of_singleton` (1.3681) — lex=0.36; ns=1.00; path=1.00; feat=0.75; name=0.38

### Set.subset_pair_iff_eq
- `Set.pair_subset_iff` (1.7254) — lex=0.55; ns=1.00; path=1.00; feat=1.00; name=0.57
- `Set.subset_pair_iff` (1.7151) — lex=0.54; ns=1.00; path=1.00; feat=1.00; name=0.57
- `Set.nontrivial_iff_pair_subset` (1.601) — lex=0.51; ns=1.00; path=0.75; feat=1.00; name=0.50

### Set.subset_singleton_iff_eq
- `Set.singleton_subset_iff` (1.6699) — lex=0.50; ns=1.00; path=1.00; feat=1.00; name=0.57
- `Set.subset_singleton_iff` (1.6508) — lex=0.48; ns=1.00; path=1.00; feat=1.00; name=0.57
- `Set.singleton_subset_singleton` (1.6404) — lex=0.51; ns=1.00; path=1.00; feat=1.00; name=0.43

### Set.BijOn.exists_extend_of_subset
- `Set.BijOn.exists_extend` (1.6149) — lex=0.77; ns=0.50; path=1.00; feat=0.67; name=0.50
- `Set.exists_subset_bijOn` (1.4144) — lex=0.40; ns=1.00; path=1.00; feat=0.67; name=0.50
- `Set.surjOn_iff_exists_bijOn_subset` (1.3225) — lex=0.40; ns=1.00; path=1.00; feat=0.50; name=0.40
- goal_defs: ['Set.BijOn', 'Set.SurjOn']

### Set.BijOn.image_eq
- `Set.EqOn.image_eq` (1.1274) — lex=0.50; ns=0.50; path=1.00; name=0.67
- `Set.BijOn` (1.0533) — lex=0.40; ns=1.00; path=0.75; name=0.40
- `Set.BijOn.union` (1.0436) — lex=0.52; ns=0.50; path=1.00; name=0.33
- goal_defs: ['Set.BijOn']

### Set.BijOn.iterate
- `Set.bijOn_of_subsingleton'` (1.6218) — lex=0.60; ns=1.00; path=1.00; feat=0.80; name=0.33
- `Set.bijOn_id` (1.3747) — lex=0.57; ns=1.00; path=1.00; feat=0.20; name=0.40
- `Set.bijOn_of_subsingleton` (1.3569) — lex=0.50; ns=1.00; path=1.00; feat=0.40; name=0.33
- goal_defs: ['Set.BijOn', 'Set.Subsingleton', 'Cycle.Subsingleton', 'Finset.Nonempty']

### Set.BijOn.subset_left
- `Set.exists_subset_bijOn` (1.3199) — lex=0.32; ns=1.00; path=1.00; feat=0.67; name=0.43
- `Set.BijOn.subset_range` (1.2557) — lex=0.44; ns=0.50; path=1.00; feat=0.67; name=0.43
- `Set.bijOn_of_subsingleton` (1.2167) — lex=0.28; ns=1.00; path=1.00; feat=0.67; name=0.25
- goal_defs: ['Set.BijOn']

### Set.BijOn.subset_range
- `Set.SurjOn.subset_range` (1.6445) — lex=0.62; ns=0.50; path=1.00; feat=1.00; name=0.67
- `Set.exists_subset_bijOn` (1.4564) — lex=0.33; ns=1.00; path=1.00; feat=1.00; name=0.43
- `Set.image_subset_range` (1.3568) — lex=0.29; ns=1.00; path=0.75; feat=1.00; name=0.43
- goal_defs: ['Set.BijOn', 'Set.range', 'Finset.range', 'Multiset.range']

### Set.BijOn.subset_right
- `Set.exists_subset_bijOn` (1.3128) — lex=0.32; ns=1.00; path=1.00; feat=0.67; name=0.43
- `Set.BijOn.subset_range` (1.2462) — lex=0.43; ns=0.50; path=1.00; feat=0.67; name=0.43
- `Set.inter_subset_right` (1.2171) — lex=0.28; ns=1.00; path=0.75; feat=0.67; name=0.43
- goal_defs: ['Set.BijOn']

### Set.EqOn.image_eq
- `Set.BijOn.image_eq` (1.1638) — lex=0.54; ns=0.50; path=1.00; name=0.67
- `Set.EqOn` (1.1605) — lex=0.50; ns=1.00; path=0.75; name=0.40
- `Set.EqOn.union` (1.1194) — lex=0.59; ns=0.50; path=1.00; name=0.33
- goal_defs: ['Set.EqOn']

### Set.EqOn.image_eq_self
- `Set.EqOn` (1.3419) — lex=0.50; ns=1.00; path=0.75; feat=0.50; name=0.33
- `Set.eqOn_singleton` (1.3094) — lex=0.37; ns=1.00; path=1.00; feat=0.67; name=0.25
- `Set.piecewise_eqOn` (1.2425) — lex=0.37; ns=1.00; path=1.00; feat=0.50; name=0.25
- goal_defs: ['Set.EqOn']

### Set.EqOn.inter_preimage_eq
- `Set.EqOn` (1.4931) — lex=0.46; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Set.EqOn.image_eq` (1.4356) — lex=0.50; ns=0.50; path=1.00; feat=1.00; name=0.38
- `Set.piecewise_eqOn` (1.4072) — lex=0.33; ns=1.00; path=1.00; feat=1.00; name=0.25
- goal_defs: ['Set.EqOn']

