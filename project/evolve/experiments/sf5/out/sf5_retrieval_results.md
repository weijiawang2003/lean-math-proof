# SF5 retrieval results

- targets: 20 | index: 5994 lemmas | top-k: 20

## Cluster `Set__iff__iff` (size 16)
Shared retrieved lemmas (≥2 targets):
- `Set.singleton_subset_singleton` — in 7 targets
- `Set.subset_singleton_iff` — in 7 targets
- `Set.singleton_subset_iff` — in 7 targets
- `Set.subset_singleton_iff_eq` — in 5 targets
- `Set.mem_sep_iff` — in 5 targets
- `Set.mem_inter_iff` — in 5 targets
- `Set.not_not_mem` — in 5 targets
- `Set.not_disjoint_iff` — in 5 targets
- `Set.diff_singleton_sSubset` — in 5 targets
- `Set.antitoneOn_iff_antitone` — in 5 targets
- `Set.subset_insert_iff` — in 4 targets
- `Set.ne_univ_iff_exists_not_mem` — in 4 targets
- `Set.subset_insert_diff_singleton` — in 4 targets
- `Set.ssubset_iff_sdiff_singleton` — in 4 targets
- `Set.diff_singleton_subset_iff` — in 4 targets

## Cluster `Set__ite_if__subset` (size 3)
Shared retrieved lemmas (≥2 targets):
- `Set.inter_subset_ite` — in 3 targets
- `Set.ite_inter` — in 3 targets
- `Set.ite` — in 3 targets
- `Set.ite_univ` — in 3 targets
- `Set.ite_mono` — in 3 targets
- `Set.ite_right` — in 3 targets
- `Set.ite_subset_union` — in 3 targets
- `Set.ite_inter_self` — in 3 targets
- `Set.ite_inter_inter` — in 3 targets
- `Set.ite_eq_of_subset_right` — in 2 targets
- `Set.ite_inter_of_inter_eq` — in 2 targets
- `Set.preimage_ite` — in 2 targets
- `Set.subset_ite` — in 2 targets
- `Set.ite_same` — in 2 targets
- `Set.iUnion_ite` — in 2 targets

## Cluster `Multiset__iff__iff` (size 1)
_no lemma retrieved for ≥2 targets_

## Per-target top-5

### Multiset.toFinset_eq_singleton_iff
- `Multiset.toFinset_card_eq_card_iff_nodup` (1.2975) — lexical=0.49; ns=1.00; path=0.75; feat=0.67
- `Multiset.toFinset_singleton` (1.2683) — lexical=0.46; ns=1.00; path=0.75; feat=0.67
- `Multiset.mem_toFinset` (1.248) — lexical=0.44; ns=1.00; path=0.75; feat=0.67
- `Multiset.toFinset_nsmul` (1.2045) — lexical=0.37; ns=1.00; path=0.75; feat=0.75
- `Multiset.toFinset_eq_empty` (1.1975) — lexical=0.46; ns=1.00; path=0.75; feat=0.50

### Set.Nonempty.subset_pair_iff_eq
- `Set.subset_pair_iff_eq` (1.8048) — lexical=0.94; ns=1.00; path=1.00; feat=0.67
- `Set.subset_pair_iff` (1.4947) — lexical=0.57; ns=1.00; path=1.00; feat=0.80
- `Set.pair_eq_pair_iff` (1.392) — lexical=0.55; ns=1.00; path=1.00; feat=0.60
- `Set.subset_singleton_iff_eq` (1.3137) — lexical=0.45; ns=1.00; path=1.00; feat=0.67
- `Set.pair_subset_iff` (1.3105) — lexical=0.39; ns=1.00; path=1.00; feat=0.80

### Set.antitoneOn_iff_antitone
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` (1.1489) — lexical=0.42; ns=1.00; path=1.00; feat=0.33
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` (1.1461) — lexical=0.41; ns=1.00; path=1.00; feat=0.33
- `Set.EqOn.congr_antitoneOn` (1.059) — lexical=0.50; ns=0.50; path=0.75; feat=0.50
- `Set.antitoneOn_singleton` (1.0528) — lexical=0.52; ns=1.00; path=0.75
- `Set.Subsingleton.antitoneOn` (0.9747) — lexical=0.61; ns=0.50; path=0.75

### Set.diff_singleton_subset_iff
- `Set.subset_insert_diff_singleton` (1.5458) — lexical=0.55; ns=1.00; path=1.00; feat=1.00
- `Set.insert_diff_singleton` (1.39) — lexical=0.47; ns=1.00; path=1.00; feat=0.80
- `Set.insert_diff_eq_singleton` (1.3534) — lexical=0.43; ns=1.00; path=1.00; feat=0.80
- `Set.subset_diff_singleton` (1.3483) — lexical=0.43; ns=1.00; path=1.00; feat=0.80
- `Set.diff_singleton_sSubset` (1.3344) — lexical=0.47; ns=1.00; path=1.00; feat=0.67

### Set.ite_eq_of_subset_left
- `Set.ite_left` (1.5335) — lexical=0.53; ns=1.00; path=1.00; feat=1.00
- `Set.ite_eq_of_subset_right` (1.4466) — lexical=0.55; ns=1.00; path=1.00; feat=0.75
- `Set.ite_subset_union` (1.3605) — lexical=0.46; ns=1.00; path=1.00; feat=0.75
- `Set.ite_inter_of_inter_eq` (1.3575) — lexical=0.49; ns=1.00; path=1.00; feat=0.67
- `Set.ite_empty_left` (1.3446) — lexical=0.54; ns=1.00; path=1.00; feat=0.50

### Set.ite_eq_of_subset_right
- `Set.ite_eq_of_subset_left` (1.4477) — lexical=0.55; ns=1.00; path=1.00; feat=0.75
- `Set.ite_left` (1.4442) — lexical=0.44; ns=1.00; path=1.00; feat=1.00
- `Set.ite_inter_of_inter_eq` (1.4389) — lexical=0.57; ns=1.00; path=1.00; feat=0.67
- `Set.ite_right` (1.4204) — lexical=0.55; ns=1.00; path=1.00; feat=0.67
- `Set.ite_subset_union` (1.368) — lexical=0.47; ns=1.00; path=1.00; feat=0.75

### Set.monotoneOn_iff_monotone
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` (1.2619) — lexical=0.40; ns=1.00; path=1.00; feat=0.67
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` (1.2592) — lexical=0.39; ns=1.00; path=1.00; feat=0.67
- `Set.monotoneOn_singleton` (1.1315) — lexical=0.49; ns=1.00; path=0.75; feat=0.25
- `Set.EqOn.congr_monotoneOn` (1.1073) — lexical=0.48; ns=0.50; path=0.75; feat=0.67
- `Set.Subsingleton.monotoneOn` (1.0534) — lexical=0.59; ns=0.50; path=0.75; feat=0.25

### Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` (1.6131) — lexical=0.61; ns=1.00; path=1.00; feat=1.00
- `Set.monotoneOn_iff_monotone` (1.2328) — lexical=0.23; ns=1.00; path=1.00; feat=1.00
- `Set.not_disjoint_iff` (1.2062) — lexical=0.34; ns=1.00; path=1.00; feat=0.67
- `Set.not_subset_iff_exists_mem_not_mem` (1.1821) — lexical=0.38; ns=1.00; path=1.00; feat=0.50
- `Set.ne_univ_iff_exists_not_mem` (1.1422) — lexical=0.28; ns=1.00; path=1.00; feat=0.67

### Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt
- `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` (1.6086) — lexical=0.61; ns=1.00; path=1.00; feat=1.00
- `Set.monotoneOn_iff_monotone` (1.2296) — lexical=0.23; ns=1.00; path=1.00; feat=1.00
- `Set.not_disjoint_iff` (1.2014) — lexical=0.33; ns=1.00; path=1.00; feat=0.67
- `Set.not_subset_iff_exists_mem_not_mem` (1.1768) — lexical=0.38; ns=1.00; path=1.00; feat=0.50
- `Set.ne_univ_iff_exists_not_mem` (1.1383) — lexical=0.27; ns=1.00; path=1.00; feat=0.67

### Set.pair_eq_pair_iff
- `Set.pair_subset_iff` (1.3655) — lexical=0.53; ns=1.00; path=1.00; feat=0.60
- `Set.pair_eq_singleton` (1.3598) — lexical=0.46; ns=1.00; path=1.00; feat=0.75
- `Set.subset_pair_iff_eq` (1.3461) — lexical=0.55; ns=1.00; path=1.00; feat=0.50
- `Set.subset_pair_iff` (1.3422) — lexical=0.50; ns=1.00; path=1.00; feat=0.60
- `Set.pairwise_pair` (1.3003) — lexical=0.50; ns=1.00; path=0.60; feat=0.75

### Set.ssubset_iff_insert
- `Set.ssubset_insert` (1.4047) — lexical=0.54; ns=1.00; path=1.00; feat=0.67
- `Set.ssubset_iff_of_subset` (1.3243) — lexical=0.46; ns=1.00; path=1.00; feat=0.67
- `Set.ssubset_iff_sdiff_singleton` (1.284) — lexical=0.35; ns=1.00; path=1.00; feat=0.83
- `Set.diff_singleton_sSubset` (1.2566) — lexical=0.32; ns=1.00; path=1.00; feat=0.83
- `Set.ssubset_singleton_iff` (1.2454) — lexical=0.36; ns=1.00; path=1.00; feat=0.71

### Set.ssubset_iff_sdiff_singleton
- `Set.diff_singleton_sSubset` (1.5322) — lexical=0.53; ns=1.00; path=1.00; feat=1.00
- `Set.ssubset_singleton_iff` (1.3729) — lexical=0.44; ns=1.00; path=1.00; feat=0.83
- `Set.ssubset_iff_of_subset` (1.2928) — lexical=0.37; ns=1.00; path=1.00; feat=0.80
- `Set.lt_iff_ssubset` (1.2568) — lexical=0.34; ns=1.00; path=1.00; feat=0.80
- `Set.empty_ssubset_singleton` (1.2557) — lexical=0.39; ns=1.00; path=1.00; feat=0.67

### Set.ssubset_singleton_iff
- `Set.ssubset_iff_sdiff_singleton` (1.4567) — lexical=0.46; ns=1.00; path=1.00; feat=1.00
- `Set.diff_singleton_sSubset` (1.4199) — lexical=0.42; ns=1.00; path=1.00; feat=1.00
- `Set.empty_ssubset_singleton` (1.3924) — lexical=0.53; ns=1.00; path=1.00; feat=0.67
- `Finset.ssubset_singleton_iff` (1.3762) — lexical=0.92; path=0.50; feat=0.83
- `Set.eq_empty_of_ssubset_singleton` (1.3391) — lexical=0.47; ns=1.00; path=1.00; feat=0.67

### Set.strictAntiOn_iff_strictAnti
- `Set.antitoneOn_iff_antitone` (0.9006) — lexical=0.10; ns=1.00; path=1.00; feat=0.50
- `Set.mem_def` (0.8817) — lexical=0.08; ns=1.00; path=1.00; feat=0.50
- `Set.mem_inter_iff` (0.8754) — lexical=0.08; ns=1.00; path=1.00; feat=0.50
- `Set.ext_iff` (0.8749) — lexical=0.07; ns=1.00; path=1.00; feat=0.50
- `Set.not_not_mem` (0.8723) — lexical=0.07; ns=1.00; path=1.00; feat=0.50

### Set.strictMonoOn_iff_strictMono
- `strictMono_restrict` (0.9237) — lexical=0.47; path=0.75; feat=0.67
- `Set.EqOn.congr_strictMonoOn` (0.9218) — lexical=0.29; ns=0.50; path=0.75; feat=0.67
- `Set.strictMonoOn_singleton` (0.9198) — lexical=0.28; ns=1.00; path=0.75; feat=0.25
- `Set._root_.StrictMonoOn.strictMono` (0.8911) — lexical=0.45; ns=0.33; path=0.75; feat=0.33
- `Set.ncard_strictMono` (0.8676) — lexical=0.23; ns=1.00; path=0.75; feat=0.25

### Set.subset_insert_iff
- `Set.insert_subset_iff` (1.4111) — lexical=0.49; ns=1.00; path=1.00; feat=0.80
- `Set.diff_singleton_subset_iff` (1.4076) — lexical=0.41; ns=1.00; path=1.00; feat=1.00
- `Set.subset_insert_diff_singleton` (1.3537) — lexical=0.35; ns=1.00; path=1.00; feat=1.00
- `Set.subset_insert_iff_of_not_mem` (1.2664) — lexical=0.35; ns=1.00; path=1.00; feat=0.80
- `Set.insert_subset_insert_iff` (1.2556) — lexical=0.34; ns=1.00; path=1.00; feat=0.80

### Set.subset_ite
- `Set.inter_subset_ite` (1.4656) — lexical=0.63; ns=1.00; path=1.00; feat=0.60
- `Set.ite` (1.4037) — lexical=0.64; ns=1.00; path=1.00; feat=0.40
- `Set.ite_subset_union` (1.3892) — lexical=0.59; ns=1.00; path=1.00; feat=0.50
- `Set.ite_diff_self` (1.3374) — lexical=0.50; ns=1.00; path=1.00; feat=0.60
- `Set.ite_inter_self` (1.2747) — lexical=0.51; ns=1.00; path=1.00; feat=0.40

### Set.subset_pair_iff_eq
- `Set.Nonempty.subset_pair_iff_eq` (1.5658) — lexical=0.87; ns=0.50; path=1.00; feat=0.67
- `Set.subset_pair_iff` (1.5031) — lexical=0.58; ns=1.00; path=1.00; feat=0.80
- `Set.pair_eq_pair_iff` (1.4001) — lexical=0.56; ns=1.00; path=1.00; feat=0.60
- `Set.subset_singleton_iff_eq` (1.3203) — lexical=0.45; ns=1.00; path=1.00; feat=0.67
- `Set.pair_subset_iff` (1.3162) — lexical=0.40; ns=1.00; path=1.00; feat=0.80

### Set.subset_singleton_iff_eq
- `Set.singleton_subset_singleton` (1.3722) — lexical=0.37; ns=1.00; path=1.00; feat=1.00
- `Set.singleton_subset_iff` (1.3622) — lexical=0.36; ns=1.00; path=1.00; feat=1.00
- `Set.subset_singleton_iff` (1.3429) — lexical=0.34; ns=1.00; path=1.00; feat=1.00
- `Set.singleton_eq_singleton_iff` (1.2517) — lexical=0.35; ns=1.00; path=1.00; feat=0.75
- `Set.range_subset_singleton` (1.2442) — lexical=0.31; ns=1.00; path=0.75; feat=1.00

### Set.union_empty_iff
- `Set.mem_diff_singleton_empty` (1.2306) — lexical=0.31; ns=1.00; path=1.00; feat=0.80
- `Set.iUnion_eq_empty` (1.2154) — lexical=0.36; ns=1.00; path=0.75; feat=0.80
- `Set.empty_union` (1.2068) — lexical=0.37; ns=1.00; path=1.00; feat=0.60
- `Set.union_empty` (1.2068) — lexical=0.37; ns=1.00; path=1.00; feat=0.60
- `Set.sUnion_eq_empty` (1.1895) — lexical=0.33; ns=1.00; path=0.75; feat=0.80

