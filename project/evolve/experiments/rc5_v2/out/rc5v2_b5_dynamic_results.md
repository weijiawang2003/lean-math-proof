# RC5V2 safe dynamic B5 results

- theorems: 149 | dynamic successes: **8** | killed (bounded): 0
- no global stalls: **True** | max wall: 65.0s (cap 60s) | off-policy: 0 | unknown-name: 75
- success targets: ['Finset.fiber_nonempty_iff_mem_image', "List.attach_map_val'", 'List.choose_mem', 'List.get_pmap', "List.pmap_append'", 'Multiset.map_count_True_eq_filter_card', 'Multiset.mem_bind', 'Set.compl_range_subset_kernImage']
- first-success ranks: {'1': 1, '2': 3, '3': 2, '5': 2}

| theorem | success | rank | wall(s) | killed | winning tactic |
|---|---|---|---|---|---|
| `Finset.fiber_nonempty_iff_mem_image` | True | 3 | 3.5 | False | `simp [Finset.filter_nonempty_iff]` |
| `List.attach_map_val'` | True | 5 | 11.36 | False | `simp [List.attach_map_coe']` |
| `List.choose_mem` | True | 5 | 11.35 | False | `simp [List.choose_spec] <;> aesop` |
| `List.get_pmap` | True | 2 | 10.25 | False | `simp [List.getElem_pmap] <;> aesop` |
| `List.pmap_append'` | True | 3 | 10.38 | False | `simp [List.pmap_append] <;> aesop` |
| `Multiset.map_count_True_eq_filter_card` | True | 2 | 11.21 | False | `simp [Multiset.count_map] <;> aesop` |
| `Multiset.mem_bind` | True | 1 | 2.39 | False | `simp [Multiset.bind]` |
| `Set.compl_range_subset_kernImage` | True | 2 | 3.93 | False | `simp [Set.kernImage_eq_compl] <;> aesop` |
