# TR2 selection-strategy comparison

## Decision: `INCONCLUSIVE_TOO_SMALL`

No strategy can yield a fresh TRUE_DELTA on an exhausted, fully pre-labelled pool; model selection shows a diversity/coverage edge (more namespaces & non-Set cases) but the sample is too small to call it a win on useful-labels-per-probe.

> Pool is ~40 fully pre-labelled theorems; figures are descriptive. Random overlaps model/rule on scarce failures by design (matched failure ratio), so per-probe comparisons control for difficulty but cannot reach significance at this n.

| metric | model | rule | random |
|---|---|---|---|
| selected | 15 | 15 | 15 |
| confirmed_rc2_failures | 13 | 11 | 13 |
| useful_labels | 14 | 14 | 14 |
| true_delta | 0 | 0 | 0 |
| missing_lemma_candidates | 9 | 11 | 8 |
| depth_gap_cases | 2 | 0 | 2 |
| no_cheap_action_confirmations | 1 | 0 | 3 |
| baseline_duplicates | 2 | 3 | 1 |
| open_flakes | 0 | 0 | 0 |
| live_probes | 29 | 27 | 28 |
| useful_per_live_probe | 0.4828 | 0.5185 | 0.5 |
| namespace_diversity | 6 | 2 | 3 |
| non_set_cases | 5 | 2 | 2 |
| underrepresented_cases | 3 | 0 | 2 |

## Per-strategy classification histograms

- **model**: {'PROOF_SEARCH_DEPTH_GAP': 2, 'PRODUCTION_SUBSUMED': 1, 'MISSING_BRIDGE_LEMMA_CANDIDATE': 9, 'BASELINE_DUPLICATE': 2, 'NO_CHEAP_ACTION': 1}
- **rule**: {'MISSING_BRIDGE_LEMMA_CANDIDATE': 11, 'BASELINE_DUPLICATE': 3, 'PRODUCTION_SUBSUMED': 1}
- **random**: {'NO_CHEAP_ACTION': 3, 'MISSING_BRIDGE_LEMMA_CANDIDATE': 8, 'PROOF_SEARCH_DEPTH_GAP': 2, 'PRODUCTION_SUBSUMED': 1, 'BASELINE_DUPLICATE': 1}