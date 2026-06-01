# RC5V2 attribution

- dynamic attempts: 149 | classifications: {'NO_DYNAMIC_WIN': 141, 'FRESH_TRUE_RC5V2_DELTA': 8}
- **FRESH_TRUE_RC5V2_DELTA: 8** ['Finset.fiber_nonempty_iff_mem_image', "List.attach_map_val'", 'List.choose_mem', 'List.get_pmap', "List.pmap_append'", 'Multiset.map_count_True_eq_filter_card', 'Multiset.mem_bind', 'Set.compl_range_subset_kernImage']
- known/control deltas: 0 | duplicates: 0 | source-specific: 0
- fresh delta by namespace: {'Finset': 1, 'List': 4, 'Multiset': 2, 'Set': 1} | by family: {'d1_simp_lemma': 2, 'd2_simp_aesop': 5, 'def_unfold_simp': 1}

| theorem | ns | class | rank | winning program |
|---|---|---|---|---|
| `Finset.fiber_nonempty_iff_mem_image` | Finset | FRESH_TRUE_RC5V2_DELTA | 3 | `simp [Finset.filter_nonempty_iff]` |
| `List.attach_map_val'` | List | FRESH_TRUE_RC5V2_DELTA | 5 | `simp [List.attach_map_coe']` |
| `List.choose_mem` | List | FRESH_TRUE_RC5V2_DELTA | 5 | `simp [List.choose_spec] <;> aesop` |
| `List.get_pmap` | List | FRESH_TRUE_RC5V2_DELTA | 2 | `simp [List.getElem_pmap] <;> aesop` |
| `List.pmap_append'` | List | FRESH_TRUE_RC5V2_DELTA | 3 | `simp [List.pmap_append] <;> aesop` |
| `Multiset.map_count_True_eq_filter_card` | Multiset | FRESH_TRUE_RC5V2_DELTA | 2 | `simp [Multiset.count_map] <;> aesop` |
| `Multiset.mem_bind` | Multiset | FRESH_TRUE_RC5V2_DELTA | 1 | `simp [Multiset.bind]` |
| `Set.compl_range_subset_kernImage` | Set | FRESH_TRUE_RC5V2_DELTA | 2 | `simp [Set.kernImage_eq_compl] <;> aesop` |
