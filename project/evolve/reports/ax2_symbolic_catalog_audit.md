# AX2 — fresh Option/List symbolic-action catalog audit

Goal: find fresh (unused, confirmed-available) Option/List theorems to mine for new symbolic-action labels under the AX1 wrapper. `used` = union of all registered theorem sets (CX3/WX1/WX2/AX1-equivalence/demo_v1).

| namespace | available unique | fresh unused | discovered-only (unverified) | buckets |
|---|---:|---:|---:|---|
| Option | 46 | 0 | 0 | {} |
| List | 260 | 76 | 0 | {'list_induction_simp': 10, 'list_cases_simp': 51, 'list_hard_unknown': 12, 'list_simp_only': 3} |

**Verdict:** Option is **exhausted** (0 fresh — all 46 available Option lemmas consumed by CX3/WX1), with no additional available candidate in the broader scan. The only fresh symbolic-mining surface is **List (76 fresh)**. AX2 dataset growth is therefore List-only; Option contribution stays at the AX1 baseline. This matches the WX2 finding that List is the sole remaining cases/induction-friendly surface.

List bucket detail (classification is a prior; the Stage 3 probe decides the actual winning tactic):

- `list_cases_simp`: 51
- `list_hard_unknown`: 12
- `list_induction_simp`: 10
- `list_simp_only`: 3
