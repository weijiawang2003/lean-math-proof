# TR5 RC4B / RC4C evidence

## RC4B — `Set.disjoint_left` bridge
- decision: **READY_FOR_RC4B_VALIDATION**
- true wins: 3 → ['Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_singleton_left']
- reproduced TR3: ['Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_singleton_left'] | fresh: []
- fire stats (top-5): {'attempted_in_top5': 9, 'closed_as_credited': 3}
- off-gate risk: low — simp[Set.disjoint_left] is a single named-lemma rewrite gated to Set goals; only fires when the lemma is retrieved
- candidate policy: narrow allowlist gate: add `simp [Set.disjoint_left]` (and the d2 `simp [Set.disjoint_left] <;> aesop`) to the Set route battery, gated to goals mentioning Disjoint; off-by-default, additive over RC2 (SET_ITE_SIMP / RC4A pattern)
- validation set: the live Set.disjoint_* wins + held-out Disjoint/disjoint_left Set theorems from the discovered catalog as fresh holdouts

## RC4C — `d2_simp_aesop` (`simp [L] <;> aesop`)
- decision: **READY_FOR_RC4C_VALIDATION**
- true wins: 3 → ['Set.Nonempty.subset_pair_iff_eq', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right']
- reproduced TR3: ['Set.Nonempty.subset_pair_iff_eq', 'Set.disjoint_iff_forall_ne', 'Set.disjoint_right'] | fresh: []
- fire stats (top-5): {'attempted_in_top5': 170, 'closed_as_credited': 3}
- false positives (live B5 d2_simp_aesop fails): 153
- source-specific risk: medium — the win depends on the retrieved lemma L being the right bridge; aesop after simp[L] can also close goals where plain aesop times out, so the credit is the simp[L] enabling step (SX4 PRODUCTION_SUBSUMED guard already applied — RC2's best-first search does NOT reach the simp[L]-advanced state)
- separate validation warranted: True
