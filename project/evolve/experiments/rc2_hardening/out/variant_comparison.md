# RC2 Hardening — Integration Variant Comparison

- attribution-clean choice: **D_additive_single_shot** — recovers all 5 credited with ZERO search perturbation — cleanest attribution for the official delta
- deployable wrapper: **A_priority_any** — the only schema-native deployable wrapper that recovers +5 (also yields harmless deterministic extra wins)

| variant | credited recovered | extra/perturb wins | regr | off-gate | perturbation | deployable | schema-native |
|---|---|---|---|---|---|---|---|
| A_priority_any | 5/5 | 4 | 0 | 0 | YES (reorders base policy) | True | True |
| D_additive_single_shot | 5/5 | 0 | 0 | 0 | NONE | False | False |
| E_sequence_sx3 | 5/5 | 4 | 0 | 0 | NONE | False | False |

## Per-theorem probes (single-shot)
| theorem | simp [Set.ite] | simp [Set.ite] <;> aesop |
|---|---|---|
| `Set.ite_empty_right` | True | True |
| `Set.ite_right` | True | True |
| `Set.ite_empty` | True | True |
| `Set.ite_empty_left` | True | True |
| `Set.ite_left` | True | True |
| `Set.ite_inter` | False | True |
| `Set.ite_inter_self` | False | True |
| `Set.ite_compl` | False | True |
| `Set.ite_inter_compl_self` | False | True |

> Variants B (late priority) and C (fallback cap-fix) were not run: B offers no attribution benefit over A (still full-wrapper perturbation) and C requires per-state-cap schema support absent here; D is the perturbation-free reference and A is the deployable artifact.