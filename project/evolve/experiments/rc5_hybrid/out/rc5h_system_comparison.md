# RC5H system comparison

| system | solved | new/RC2 | new/RC4 | regr | dyn probes | probes/win |
|---|---|---|---|---|---|---|
| RC2 | 75 | 0 | 0 | 10 | - | - |
| RC4_static | 85 | 10 | 0 | 0 | - | - |
| RC5H_B5 | 88 | 13 | 3 | 0 | 443 | 147.7 |
| RC5H_B10 | 88 | 13 | 3 | 0 | 864 | 288.0 |
| RC5H_B20 | 88 | 13 | 3 | 0 | 1285 | 428.3 |

- RC4 static contribution (new/RC2): **10**
- dynamic stage contribution (new/RC4 @B20): **3**
- hybrid net delta over RC2 @B20: **13**
- floors (n=80): RC2 61 / RC4 61 / RC5H 61
- new/RC4 by namespace: {'Finset': 2, 'Multiset': 1} | by set: {'TR6_dynamic_tail_replay': 2, 'Fresh_dynamic_candidate_frontier': 1}
- **best budget: B5**
