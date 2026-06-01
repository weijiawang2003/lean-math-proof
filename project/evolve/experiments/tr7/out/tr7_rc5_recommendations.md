# TR7 RC5 recommendations

- **primary: RC5_HYBRID_STATIC_PLUS_RANKER**
- static-compatible 78% / dynamic-only 22% / distribution PARTIAL_DISTRIBUTION_MISMATCH

> The headline 'RC4R 0 fresh delta' is substantially a SELECTION ARTIFACT: 0/18 TR6 wins are in the RC4R fresh frontier (it excluded all RC4D-used theorems, which is where 14/18 TR6 wins went), and the fresh set over-samples the loose RC4A gate. RC4 actually covers 10/18 TR6 wins as its known wins.

## RC5_HYBRID_STATIC_PLUS_RANKER (primary)
- **rationale:** 78% of TR6 fresh wins are static-compatible (already RC4 or fixable by allowlist/gate/schema work) but 22% require theorem-specific retrieved lemmas (tauto/rw/exact families) that cannot become a small static allowlist. RC4 static is safe and reproduces the validated component wins; the fresh out-of-sample delta needs the TR6 ranker-guided dynamic retrieval that found per-theorem lemmas. Keep RC4 static as the deterministic core and add a gated ranker-guided dynamic retrieval stage for the dynamic tail.
- **expected benefit:** recovers the fresh-frontier generalization RC4 static lacks without losing RC4's safety/determinism on the static core.
- **risk:** dynamic retrieval reintroduces nondeterminism + probe cost; must be gated and owner-billed; ranker is namespace-specific (TR4/TR6 caveat).
- **required validation:** RC5 hybrid benchmark = RC4 static floors/known-wins preserved + ranker-guided dynamic stage measured on a FRESH frontier with SX4-style attribution; determinism scoped to the static core.
- **next task:** RC5H — Hybrid static+ranker wrapper prototype & fresh-frontier benchmark

## RC4A_TIGHTEN_MONO_GATE (gate refinement, not a release) (secondary)
- **rationale:** RC4A def-unfold gate fires 76× and closes only 7 (precision 0.092); it fires on every monotone/antitone theorem. Tighten to the iff-unfold shape (require `_iff_` in the name) to cut wasted emissions. Additive/safe today, so low urgency, but it is the loosest component.
- **expected benefit:** less wasted probe budget; cleaner precision before any expansion.
- **risk:** tightening could drop a future iff-unfold win — validate additively.
- **required validation:** re-run RC4A external-additive eval with the tightened gate; confirm 0 lost wins, lower fire count.
- **next task:** folded into RC5H or a standalone RC4A-gate patch

## TR8_MORE_FRONTIER_DATA (secondary)
- **rationale:** 3 TR6 wins use clean single-occurrence lemmas (['Finset.biUnion_subset', 'Finset.subset_iff', 'Set.MapsTo']) that are allowlist-expansion candidates but lack repeat / namespace-parametric evidence. A larger fresh sweep would tell whether they recur (→ static) or stay one-off (→ dynamic).
- **expected benefit:** resolves the allowlist-expansion-vs-dynamic question for the tail.
- **risk:** more compute; may still be inconclusive.
- **required validation:** TR6-style ranked live sweep over a larger fresh pool, count recurrence of these lemmas.
- **next task:** TR8 — larger fresh-frontier sweep focused on the candidate lemmas

