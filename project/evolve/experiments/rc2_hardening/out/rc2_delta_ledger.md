# RC2 Credited-Delta Ledger

- **credited delta = 5**: ['Set.ite_empty', 'Set.ite_empty_left', 'Set.ite_empty_right', 'Set.ite_left', 'Set.ite_right']
- deferred (SX3 sequence candidates): ['Set.ite_compl', 'Set.ite_inter', 'Set.ite_inter_compl_self', 'Set.ite_inter_self']
- excluded (search perturbation): []
- category histogram: `{'credited_SET_ITE_SIMP': 5, 'SX3_sequence_candidate': 4}`
- policy: Official RC2 credited delta counts ONLY credited_SET_ITE_SIMP (single-shot, literal-RC1-confirmed, minimal-relabel TRUE). SX3_sequence_candidate -> deferred to SX3. search_perturbation -> excluded.

| theorem | category | minimal_relabel | win tactic (steps) | decision | reason |
|---|---|---|---|---|---|
| `Set.ite_empty` | credited_SET_ITE_SIMP | TRUE_SET_ITE_SIMP_WIN | `simp [Set.ite]` (1) | **credit** | single-shot simp [Set.ite] closes it; literal RC1 and all ba |
| `Set.ite_empty_left` | credited_SET_ITE_SIMP | TRUE_SET_ITE_SIMP_WIN | `simp [Set.ite]` (1) | **credit** | single-shot simp [Set.ite] closes it; literal RC1 and all ba |
| `Set.ite_empty_right` | credited_SET_ITE_SIMP | TRUE_SET_ITE_SIMP_WIN | `simp [Set.ite]` (1) | **credit** | single-shot simp [Set.ite] closes it; literal RC1 and all ba |
| `Set.ite_left` | credited_SET_ITE_SIMP | TRUE_SET_ITE_SIMP_WIN | `simp [Set.ite]` (1) | **credit** | single-shot simp [Set.ite] closes it; literal RC1 and all ba |
| `Set.ite_right` | credited_SET_ITE_SIMP | TRUE_SET_ITE_SIMP_WIN | `simp [Set.ite]` (1) | **credit** | single-shot simp [Set.ite] closes it; literal RC1 and all ba |
| `Set.ite_compl` | SX3_sequence_candidate | UNEXPECTED_WIN_NEEDS_REVIEW | `aesop` (2) | **defer** | simp [Set.ite] <;> aesop/simp_all closes it (depth-2); bare  |
| `Set.ite_inter` | SX3_sequence_candidate | UNEXPECTED_WIN_NEEDS_REVIEW | `aesop` (2) | **defer** | simp [Set.ite] <;> aesop/simp_all closes it (depth-2); bare  |
| `Set.ite_inter_compl_self` | SX3_sequence_candidate | UNEXPECTED_WIN_NEEDS_REVIEW | `aesop` (2) | **defer** | simp [Set.ite] <;> aesop/simp_all closes it (depth-2); bare  |
| `Set.ite_inter_self` | SX3_sequence_candidate | UNEXPECTED_WIN_NEEDS_REVIEW | `aesop` (2) | **defer** | simp [Set.ite] <;> aesop/simp_all closes it (depth-2); bare  |