# CX1 — current catalog audit

**Catalog:** `project/discovered_theorems.json`
**Mathlib commit:** `29dcec074de168ac2bf835a77ef68bbe069194c5`
**Total theorems:** **527**
**By difficulty:** easy=420, medium=60, hard=47

## Source files scanned

| file | theorems |
|---|---:|
| `Mathlib/Data/Finset/Basic.lean` | 228 |
| `Mathlib/Data/Nat/Defs.lean` | 208 |
| `Mathlib/Data/Set/Basic.lean` | 91 |

## Per-namespace counts

| namespace | total | easy | medium | hard |
|---|---:|---:|---:|---:|
| `Nat` | 208 | 147 | 28 | 33 |
| `Finset` | 200 | 171 | 21 | 8 |
| `Set` | 89 | 76 | 10 | 3 |
| `List` | 13 | 12 | 1 | 0 |
| `Multiset` | 13 | 10 | 0 | 3 |
| `Bool` | 0 | 0 | 0 | 0 |
| `Option` | 0 | 0 | 0 | 0 |
| `Int` | 0 | 0 | 0 | 0 |
| _other_ | 4 | – | – | – |

## Theorem-set coverage

**Total distinct theorems used across all `THEOREM_SETS`:** 508
**Used but absent from the discovered catalog:** 31 (e.g. hand-written tasks)

Examples of used-but-not-in-catalog:
- `Finset.insert_comm`
- `Finset.mem_insert`
- `Finset.mem_singleton`
- `Nat.add_mod`
- `Nat.mod_add_mod`
- `Nat.mul_mod`
- `Set.diff_empty`
- `Set.diff_self`

## Per-namespace usage / remainder

| namespace | catalog | used | unused | exhaustion |
|---|---:|---:|---:|---|
| `Nat` | 208 | 208 | 0 | EXHAUSTED (208/208 used) |
| `Finset` | 200 | 200 | 0 | EXHAUSTED (200/200 used) |
| `Set` | 89 | 46 | 43 | HAS REMAINING SURFACE (43/89 unused) |
| `List` | 13 | 13 | 0 | EXHAUSTED (13/13 used) |
| `Multiset` | 13 | 10 | 3 | HAS REMAINING SURFACE (3/13 unused) |
| `Bool` | 0 | 0 | 0 | ABSENT from catalog |
| `Option` | 0 | 0 | 0 | ABSENT from catalog |
| `Int` | 0 | 0 | 0 | ABSENT from catalog |

## Exhaustion summary

- Nat: EXHAUSTED (208/208 used)
- Finset: EXHAUSTED (200/200 used)
- Set: HAS REMAINING SURFACE (43/89 unused)
- List: EXHAUSTED (13/13 used)
- Multiset: HAS REMAINING SURFACE (3/13 unused)
- Bool: ABSENT from catalog
- Option: ABSENT from catalog
- Int: ABSENT from catalog

## CX1 implications

The current catalog was built by scanning only 3 Mathlib source files: `Mathlib/Data/Nat/Defs.lean`, `Mathlib/Data/Set/Basic.lean`, `Mathlib/Data/Finset/Basic.lean`. Nat is fully exhausted; Finset is nearly so. Set has remaining surface but the Set base also looks largely used. List/Multiset are only 13 thms each — most of the namespace surface is outside the scanned files. Bool/Option/Int are absent.

**`extract_theorems.py` already lists 16 EXTENDED_FILES that have not yet been scanned**, including `Finset/Image.lean`, `Finset/Card.lean`, `Nat/GCD/Basic.lean`, `List/Basic.lean`, `Bool/Basic.lean`, and `Int/Basic.lean`. Scanning these is the natural CX1 Stage 2.

