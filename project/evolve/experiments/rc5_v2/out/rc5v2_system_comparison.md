# RC5V2 system comparison

| system | solved | Δ/RC2 | Δ/RC4 | regr |
|---|---|---|---|---|
| RC2 | 67 | — | — | — |
| RC4 static | 67 | 0 | 0 | 0 |
| RC5V2 (RC4+safe B5) | 75 | 8 | **8** | 0 |

- **safe dynamic B5 fresh gain over RC4: 8** ['Finset.fiber_nonempty_iff_mem_image', "List.attach_map_val'", 'List.choose_mem', 'List.get_pmap', "List.pmap_append'", 'Multiset.map_count_True_eq_filter_card', 'Multiset.mem_bind', 'Set.compl_range_subset_kernImage']
- dynamic probes: 698 | probes/fresh delta: 87.2
- fresh delta by namespace: {'List': 4, 'Set': 1, 'Multiset': 2, 'Finset': 1}
- RC4 remains the static core; safe dynamic stage is additive (0 regressions).
