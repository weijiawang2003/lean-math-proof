# TR1 training dataset summary

- examples: **57** (with goal text: 31)
- label types: {'negative': 23, 'triage': 27, 'positive': 7}
- confidence: {'verified': 27, 'strong': 30}
- low-support labels (<3): ['PROOF_SEARCH_DEPTH_GAP', 'WX3_MULTISET_INDUCTION']
- zero-support labels: ['MX2_TOFINSET_AESOP', 'SOURCE_SPECIFIC_OR_REJECTED']

## Label distribution

| label | type | count |
|---|---|---|
| `MISSING_BRIDGE_LEMMA_CANDIDATE` | triage | 19 |
| `BASELINE_DUPLICATE` | negative | 18 |
| `NO_CHEAP_ACTION` | triage | 7 |
| `SET_ITE_SIMP` | positive | 6 |
| `SX3_PRODUCTION_SUBSUMED` | negative | 5 |
| `PROOF_SEARCH_DEPTH_GAP` | triage | 1 |
| `WX3_MULTISET_INDUCTION` | positive | 1 |
| `MX2_TOFINSET_AESOP` | positive | 0 |
| `SOURCE_SPECIFIC_OR_REJECTED` | negative | 0 |

Sources: ['rc2_delta_ledger', 'rc2_delta_ledger_deferred', 'sf4_confirmation', 'sf4_missing_lemma_triage', 'sf4_probe_results', 'sf4_sx4_attribution', 'sx2_set2_relabel', 'sx4_reattribution']

> positives only from production deltas / minimal-relabel-confirmed / accepted RC components; SX3 proxy wins enter only as SX3_PRODUCTION_SUBSUMED (negative).