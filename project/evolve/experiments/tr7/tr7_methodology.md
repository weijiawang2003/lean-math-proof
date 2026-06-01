# TR7 — Fresh Delta Gap Analysis: TR6 Ranker Search vs RC4 Static Wrapper

## What TR7 is

TR7 is a **diagnostic** task. It does **not** discover a new candidate. It explains a single
apparent contradiction:

- **TR6** (ranker-guided live search: retrieval + model-ranked program selection + live probes)
  found **18 fresh TRUE_DELTA** wins on a fresh multi-namespace frontier.
- **RC4R** (the static RC4 release wrapper: fixed allowlist + fixed name-prefix gates +
  schema-native wrapper) found **0 fresh out-of-sample delta** on its fresh frontier.

TR7 asks: *where does the generalization disappear?*

## The two systems

| | TR6 dynamic | RC4 static |
|---|---|---|
| lemma source | **retrieval** (per-theorem candidate lemmas) | **fixed allowlist** (14 promoted lemmas/defs) |
| program selection | **model ranker** over a program grammar | fixed priority tactics |
| gating | implicit (retrieval relevance + ranker score) | **name-prefix `startswith` gates** |
| emission | top-B ranked programs probed live | gated priority tactics in best-first search |

## Hypotheses to discriminate (the task's 7 candidate explanations)

1. theorem-set **distribution mismatch** (the two "fresh" sets are different cohorts)
2. **missing allowlisted lemmas** (TR6's winning lemma not in the static allowlist)
3. static **gate too narrow** (relevant action exists but gate doesn't fire)
4. static **gate too broad but wrong target** (fires a lot, closes little)
5. **wrapper representation mismatch** (program works externally, search can't reproduce)
6. ranker-selected **theorem-specific lemmas** that cannot become static tactics
7. **dynamic retrieval required** (static abstraction is wrong for these cases)

## Definitions

- **TR6_FRESH_WIN** — a theorem with TR6 `FRESH_TRUE_DELTA` (18 total).
- **RC4R_FRESH_CASE** — a theorem in the RC4R fresh out-of-sample frontier (125).
- **STATIC_COVERAGE** — whether any RC4 static action would even be emitted for a theorem
  (its name-prefix gate fires).
- **DYNAMIC_ONLY_WIN** — a TR6 fresh win no RC4 static component covers.
- **STATIC_GATE_MISS** — a relevant RC4 action exists but its gate does not fire.
- **ALLOWLIST_MISS** — the winning TR6 lemma is not in the RC4 static allowlist.
- **WRAPPER_REPRESENTATION_MISS** — the external program works but the schema-native wrapper
  search cannot reproduce it.
- **DISTRIBUTION_MISMATCH** — the RC4R fresh frontier does not contain analogues of TR6 wins.
- **RC5_DYNAMIC_RETRIEVAL_CANDIDATE** — a case arguing the static wrapper is the wrong
  abstraction and dynamic retrieval should be used.

## Key prior fact (the temporal cohort)

The 18 TR6 fresh wins were found **before** RC4B/RC4C/RC4D existed; **14 of the 18 were then
folded into RC4D validation as RC4B/RC4C/RC4A evidence**, and 11 are RC4R's *known wins* (part
of RC4's +22). RC4R's fresh out-of-sample frontier **explicitly excluded every RC4D-used
theorem**, so **0 of the 18 TR6 wins are in the RC4R fresh set by construction**. TR7 quantifies
how much of the "0 fresh delta" is this cohort/selection artifact vs a genuine static-abstraction
limitation, and what remains (allowlist misses, dynamic-retrieval-only cases) to inform RC5.

## Constraints

Diagnostic only. No wrapper modification, no allowlist additions, no RC5 release, no promotion,
no production routing change. One small live step (replay ~18 TR6 wins through RC2 / RC4 / the
exact TR6 program). Protected: RC1/RC2/NS24, NS9, REL1/RC1/RC2 reports, TR1–TR6 datasets,
RC4A/B/C/D + RC4R artifacts. No commit.
