# RC5S strict policy diff vs RC5H

- RC5H plan classification under strict grammar: {'POLICY_ALLOWED': 1269, 'REMOVED_STALL_RISK': 366, 'REMOVED_OFF_POLICY': 73, 'REMOVED_NAMESPACE_DISABLED': 84}
- would remove 523 of 1792 RC5H programs

## Removed tactic families (rationale)
- **simp_all**: any tactic containing simp_all (simp [L] <;> simp_all, rw [L] <;> simp_all, bare simp_all) — top Dojo-stall cause in RC5H (230+ programs, ignores SIGALRM)
- **depth3_try_chains**: simp [L] <;> try aesop <;> try simp_all and similar — 112 programs, depth-3 try chains stall and are off-grammar
- **depth3_other**: any program with >=2 `<;>` except the single `constructor <;> intro h <;> aesop` pattern (e.g. ext x <;> simp [L] <;> aesop)
- **bare_tactics**: aesop / omega / nlinarith / tauto / decide with no lemma — off-policy TR6 grammar leakage (74 off-policy programs in RC5H)

## Allowed grammar (strict, 8 patterns + simpa variants)
- `exact L` (exact_L)
- `simpa using L` (simpa_using_L)
- `simpa [L]` (simpa_L)
- `simp [L]` (simp_L)
- `rw [L]` (rw_L)
- `simp [L] <;> aesop` (simp_L_aesop)
- `rw [L] <;> aesop` (rw_L_aesop)
- `ext x <;> simp [L]` (ext_simp_L)
- `constructor <;> intro h <;> aesop` (constructor_aesop)

## Budgets
- default B5; B10 = safe non-aesop families only; **B20 disabled**

## Timeout policy
- per-theorem wall cap **60s** (process-group kill); per-tactic 8s
- aesop-tail namespace gate: ['Set', 'Finset', 'List', 'Multiset'] (not Nat)

## RC5H winners preserved

- Finset.biUnion_subset_iff_forall_subset (simp [Finset.biUnion_subset] <;> aesop)
- Multiset.add_bind (simp [Multiset.bind])
- Finset.image_subset_iff (simp [Finset.subset_iff])
