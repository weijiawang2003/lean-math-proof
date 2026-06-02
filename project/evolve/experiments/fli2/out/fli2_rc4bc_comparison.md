# FLI2 vs RC4B/RC4C

- RC4B decision: `RC4B_CANDIDATE_CONFIRMED` | RC4C decision: `RC4C_CONFIRMED_WITH_RC4B_OVERLAP`
- RC4 method: manual hand-built static wrapper + literal-RC2 validation
- FLI2 method: automated failure-analysis → gated retrieved-lemma deployment → at-position rescue
- FLI2 true rescues: 11
- overlap with RC4 families: 0 | new families: 11 ['Finset:card', 'Finset:filterMap', 'Finset:map', 'Finset:preimage', 'Finset:subtype', 'List:bidirectionalRec']

## Overlap cases


## New (beyond RC4B/RC4C)

- `Finset.card_le_one_iff` via `simp [Finset.card_le_one] <;> aesop` (lemma `Finset.card_le_one`)
- `Finset.mem_filterMap` via `simp [Finset.filterMap]` (lemma `Finset.filterMap`)
- `Finset.mem_filterMap` via `simp [Finset.filterMap] <;> aesop` (lemma `Finset.filterMap`)
- `Finset.card_subtype` via `simp [Finset.subtype]` (lemma `Finset.subtype`)
- `Finset.card_subtype` via `simp [Finset.subtype] <;> aesop` (lemma `Finset.subtype`)
- `Finset.mem_map` via `simp [Finset.map]` (lemma `Finset.map`)
- `Finset.mem_map` via `simp [Finset.map] <;> aesop` (lemma `Finset.map`)
- `Finset.mem_preimage` via `simp [Finset.preimage]` (lemma `Finset.preimage`)
- `Finset.mem_preimage` via `simp [Finset.preimage] <;> aesop` (lemma `Finset.preimage`)
- `List.bidirectionalRec_singleton` via `simp [List.bidirectionalRec]` (lemma `List.bidirectionalRec`)
- `List.bidirectionalRec_singleton` via `simp [List.bidirectionalRec] <;> aesop` (lemma `List.bidirectionalRec`)

## Assessment

FLI2 discovers the same KIND of object RC4B/RC4C were hand-built for — a small gated `simp [L]`/closer action that deploys an existing lemma — but sourced automatically from failure analysis rather than manual curation. 11 at-position rescue(s) found; 0 overlap RC4-style families, 11 are new. Whether it becomes an RC-candidate generator depends on each rescue passing the full literal-RC2 additive validation (off-gate/floors/determinism) used for RC4A–RC4D; FLI2 only produces candidates, it does not validate or promote them.
