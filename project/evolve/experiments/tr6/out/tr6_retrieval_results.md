# TR6 retrieval results

- targets: 137 | index 10790 | top-20 | coverage 137/137 | avg best score 1.376

## Per-target top-3 (first 25 targets)
### Set.disjoint_sUnion_left
- `Set.disjoint_sUnion_right` (1.4873) — lex=0.46; ns=1.00; path=1.00; feat=0.75; name=0.43
- `Set.disjoint_iUnion_left` (1.3387) — lex=0.31; ns=1.00; path=1.00; feat=0.75; name=0.43
- `Set.disjoint_union_left` (1.3291) — lex=0.36; ns=1.00; path=0.75; feat=0.75; name=0.43
- goal_defs: ['AList.Disjoint', 'Multiset.Disjoint']

### Set.disjoint_sUnion_right
- `Set.disjoint_sUnion_left` (1.4871) — lex=0.46; ns=1.00; path=1.00; feat=0.75; name=0.43
- `Set.disjoint_iUnion_right` (1.3399) — lex=0.31; ns=1.00; path=1.00; feat=0.75; name=0.43
- `Set.disjoint_union_right` (1.3304) — lex=0.36; ns=1.00; path=0.75; feat=0.75; name=0.43
- goal_defs: ['AList.Disjoint', 'Multiset.Disjoint']

### Set.injOn_union
- `Set.BijOn.union` (1.359) — lex=0.55; ns=0.50; path=1.00; feat=0.67; name=0.40
- `Set.disjoint_union_left` (1.3424) — lex=0.32; ns=1.00; path=0.75; feat=1.00; name=0.29
- `Set.disjoint_union_right` (1.3421) — lex=0.32; ns=1.00; path=0.75; feat=1.00; name=0.29
- goal_defs: ['Set.InjOn', 'AList.Disjoint', 'Multiset.Disjoint']

### Set.disjoint_iUnion_left
- `Set.disjoint_iUnion` (1.549) — lex=0.50; ns=1.00; path=1.00; feat=0.75; name=0.50
- `Set.disjoint_iUnion_right` (1.5033) — lex=0.47; ns=1.00; path=1.00; feat=0.75; name=0.43
- `Set.disjoint_sUnion_left` (1.3389) — lex=0.31; ns=1.00; path=1.00; feat=0.75; name=0.43
- goal_defs: ['AList.Disjoint', 'Multiset.Disjoint']

### Set.disjoint_iUnion_right
- `Set.disjoint_iUnion_left` (1.5031) — lex=0.47; ns=1.00; path=1.00; feat=0.75; name=0.43
- `Set.disjoint_iUnion` (1.4603) — lex=0.41; ns=1.00; path=1.00; feat=0.75; name=0.50
- `Set.disjoint_sUnion_right` (1.3402) — lex=0.31; ns=1.00; path=1.00; feat=0.75; name=0.43
- goal_defs: ['AList.Disjoint', 'Multiset.Disjoint']

### Set.InjOn.mem_image_iff
- `Set.InjOn.image_subset_image_iff` (1.3228) — lex=0.45; ns=0.50; path=1.00; feat=0.75; name=0.50
- `Set.InjOn.image_eq_image_iff` (1.3102) — lex=0.44; ns=0.50; path=1.00; feat=0.75; name=0.50
- `Set.InjOn.mem_of_mem_image` (1.3053) — lex=0.53; ns=0.50; path=1.00; feat=0.50; name=0.50
- goal_defs: ['Set.InjOn']

### Set.mapsTo_sInter
- `Set.mapsTo_singleton` (1.3562) — lex=0.32; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Set.sInter_diff_singleton_univ` (1.3185) — lex=0.24; ns=1.00; path=1.00; feat=1.00; name=0.25
- `Set.sInter_eq_univ` (1.2598) — lex=0.31; ns=1.00; path=1.00; feat=0.67; name=0.29
- goal_defs: ['Set.MapsTo']

### Set.mapsTo_sUnion
- `Set.mapsTo_iUnion` (1.2938) — lex=0.29; ns=1.00; path=1.00; feat=0.75; name=0.33
- `Set.MapsTo.union` (1.2714) — lex=0.59; ns=0.50; path=0.75; feat=0.50; name=0.40
- `Set.disjoint_sUnion_left` (1.2698) — lex=0.28; ns=1.00; path=1.00; feat=0.75; name=0.29
- goal_defs: ['Set.MapsTo']

### Set.mapsTo'
- `Set.mapsTo_inter` (1.0197) — lex=0.23; ns=1.00; path=1.00; feat=0.33; name=0.20
- `Set.mapsTo_image_iff` (1.0148) — lex=0.23; ns=1.00; path=1.00; feat=0.33; name=0.17
- `Set.maps_univ_to` (1.0071) — lex=0.22; ns=1.00; path=1.00; feat=0.33; name=0.17
- goal_defs: ['Set.MapsTo']

### Set.kernImage_preimage_eq_iff
- `Set.compl_range_subset_kernImage` (1.3977) — lex=0.50; ns=1.00; path=1.00; feat=0.60; name=0.20
- `Set.kernImage_eq_compl` (1.3491) — lex=0.48; ns=1.00; path=1.00; feat=0.40; name=0.38
- `Set.subset_kernImage_iff` (1.3354) — lex=0.45; ns=1.00; path=0.75; feat=0.60; name=0.38
- goal_defs: ['Set.kernImage', 'Set.range', 'Finset.range', 'Multiset.range']

### Set.InjOn.image_eq_image_iff
- `Set.InjOn.image_subset_image_iff` (1.3772) — lex=0.54; ns=0.50; path=1.00; feat=0.67; name=0.50
- `Set.image_preimage_eq_iff` (1.2877) — lex=0.33; ns=1.00; path=0.75; feat=0.67; name=0.50
- `Set.InjOn.mem_image_iff` (1.2835) — lex=0.44; ns=0.50; path=1.00; feat=0.67; name=0.50
- goal_defs: ['Set.InjOn']

### Set.InjOn.image_subset_image_iff
- `Set.image_subset_image_iff` (1.814) — lex=0.63; ns=1.00; path=0.75; feat=1.00; name=0.83
- `Set.InjOn.image_eq_image_iff` (1.5198) — lex=0.54; ns=0.50; path=1.00; feat=1.00; name=0.50
- `Set.image_subset_iff` (1.4795) — lex=0.37; ns=1.00; path=0.75; feat=1.00; name=0.57
- goal_defs: ['Set.InjOn']

### Set.InjOn.image_ssubset_image_iff
- `Set.ssubset_iff_of_subset` (1.4153) — lex=0.38; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Set.lt_iff_ssubset` (1.3988) — lex=0.35; ns=1.00; path=0.75; feat=1.00; name=0.38
- `Set.ssubset_iff_subset_ne` (1.3975) — lex=0.36; ns=1.00; path=0.75; feat=1.00; name=0.33
- goal_defs: ['Set.InjOn']

### Set.surjOn_iff_exists_map_subtype
- `Set.surjOn_iff_exists_bijOn_subset` (1.4061) — lex=0.29; ns=1.00; path=1.00; feat=1.00; name=0.40
- `Set.mapsTo_iff_exists_map_subtype` (1.3517) — lex=0.32; ns=1.00; path=1.00; feat=0.67; name=0.56
- `Set.exists_subset_range_and_iff` (1.2855) — lex=0.27; ns=1.00; path=0.75; feat=1.00; name=0.27
- goal_defs: ['Set.SurjOn']

### Set.biUnion_compl_eq_of_pairwise_disjoint_of_iUnion_eq_univ
- `Set.iInter_eq_compl_iUnion_compl` (1.1632) — lex=0.22; ns=1.00; path=1.00; feat=0.60; name=0.33
- `Set.iUnion_eq_compl_iInter_compl` (1.1632) — lex=0.22; ns=1.00; path=1.00; feat=0.60; name=0.33
- `Set.iUnion_of_singleton` (1.1481) — lex=0.23; ns=1.00; path=1.00; feat=0.60; name=0.25
- goal_defs: ['AList.Disjoint', 'Multiset.Disjoint', 'Multiset.Pairwise']

### Set.mapsTo_singleton
- `Set.surjOn_singleton` (1.3239) — lex=0.22; ns=1.00; path=1.00; feat=1.00; name=0.33
- `Set.bijOn_singleton` (1.3023) — lex=0.20; ns=1.00; path=1.00; feat=1.00; name=0.33
- `Set.mem_singleton_iff` (1.2736) — lex=0.25; ns=1.00; path=0.75; feat=1.00; name=0.29
- goal_defs: ['Set.MapsTo']

### Set.mapsTo_inter
- `Set.MapsTo.inter` (1.659) — lex=0.81; ns=0.50; path=1.00; feat=0.50; name=0.75
- `Set.mapsTo_iInter` (1.4159) — lex=0.38; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Set.mapsTo_image_iff` (1.4036) — lex=0.32; ns=1.00; path=1.00; feat=1.00; name=0.29
- goal_defs: ['Set.MapsTo']

### Set.mapsTo_union
- `Set.MapsTo.union` (1.7209) — lex=0.80; ns=0.50; path=1.00; feat=0.67; name=0.75
- `Set.mapsTo_iUnion` (1.4117) — lex=0.37; ns=1.00; path=0.75; feat=1.00; name=0.33
- `Set.mapsTo_sUnion` (1.3926) — lex=0.36; ns=1.00; path=0.75; feat=1.00; name=0.33
- goal_defs: ['Set.MapsTo']

### Set.mapsTo_range_iff
- `Set.mapsTo_singleton` (1.4001) — lex=0.31; ns=1.00; path=1.00; feat=1.00; name=0.29
- `Set.maps_range_to` (1.3663) — lex=0.42; ns=1.00; path=1.00; feat=0.67; name=0.25
- `Set.mapsTo_univ_iff` (1.3326) — lex=0.34; ns=1.00; path=1.00; feat=0.67; name=0.43
- goal_defs: ['Set.MapsTo', 'Set.range', 'Finset.range', 'Multiset.range']

### Set.MapsTo.mem_iff
- `Set.mem_compl_singleton_iff` (1.3578) — lex=0.31; ns=1.00; path=0.75; feat=1.00; name=0.38
- `Set.mapsTo_singleton` (1.3407) — lex=0.35; ns=1.00; path=1.00; feat=0.75; name=0.29
- `Set.mapsTo_univ_iff` (1.2683) — lex=0.34; ns=1.00; path=1.00; feat=0.50; name=0.43
- goal_defs: ['Set.MapsTo']

### Set.bijective_iff_bijective_of_iUnion_eq_univ
- `Set.injective_iff_injective_of_iUnion_eq_univ` (1.4527) — lex=0.41; ns=1.00; path=1.00; feat=0.67; name=0.60
- `Set.surjective_iff_surjective_of_iUnion_eq_univ` (1.4427) — lex=0.40; ns=1.00; path=1.00; feat=0.67; name=0.60
- `Set.iUnion_eq_univ_iff` (1.3239) — lex=0.29; ns=1.00; path=1.00; feat=0.67; name=0.56
- goal_defs: ['Set.restrictPreimage']

### Finset.card_union_eq_card_add_card
- `Finset.card_sdiff_add_card` (1.5459) — lex=0.43; ns=1.00; path=1.00; feat=1.00; name=0.38
- `Finset.card_inter_add_card_union` (1.4597) — lex=0.44; ns=1.00; path=1.00; feat=0.67; name=0.50
- `Finset.card_union_add_card_inter` (1.4597) — lex=0.44; ns=1.00; path=1.00; feat=0.67; name=0.50
- goal_defs: ['Finset.card', 'AList.Disjoint', 'Multiset.Disjoint', 'Multiset.card']

### Finset.disjoint_biUnion_left
- `Finset.disjoint_biUnion_right` (1.6179) — lex=0.49; ns=1.00; path=1.00; feat=1.00; name=0.43
- `Finset.disjoint_union_left` (1.4383) — lex=0.37; ns=1.00; path=0.75; feat=1.00; name=0.43
- `Finset.mem_biUnion` (1.4232) — lex=0.34; ns=1.00; path=1.00; feat=1.00; name=0.29
- goal_defs: ['Finset.biUnion', 'AList.Disjoint', 'Multiset.Disjoint']

### Finset.disjoint_biUnion_right
- `Finset.disjoint_biUnion_left` (1.6177) — lex=0.49; ns=1.00; path=1.00; feat=1.00; name=0.43
- `Finset.disjoint_union_right` (1.4396) — lex=0.37; ns=1.00; path=0.75; feat=1.00; name=0.43
- `Finset.mem_biUnion` (1.4229) — lex=0.34; ns=1.00; path=1.00; feat=1.00; name=0.29
- goal_defs: ['Finset.biUnion', 'AList.Disjoint', 'Multiset.Disjoint']

### Finset.card_filter_le_iff
- `Multiset.card_filter_le_iff` (1.5001) — lex=0.89; path=0.50; feat=0.67; name=0.71
- `Finset.le_card_iff_exists_subset_card` (1.4333) — lex=0.30; ns=1.00; path=1.00; feat=1.00; name=0.44
- `Finset.subset_iff_eq_of_card_le` (1.4257) — lex=0.31; ns=1.00; path=1.00; feat=1.00; name=0.40
- goal_defs: ['Finset.card', 'Finset.filter', 'Multiset.card', 'Multiset.filter']

