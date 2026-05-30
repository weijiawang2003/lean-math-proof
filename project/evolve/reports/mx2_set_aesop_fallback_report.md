# MX2 — Set `aesop` fallback tweak

**Branch:** `mx2-set-aesop-fallback`
**Base:** MX1 (`8ec8c99`)
**Stage:** wrapper/fallback experiment (LIVE LeanDojo; **no training**)
**Decision:** **B — narrow `Set.Finite`/`toFinset` aesop patch.** 2 clean-aesop
wins beyond production, 0 regressions, 0 off-Set emissions; the broad Set gate
adds **no** extra wins over the narrow one and incurs overfiring cost.

---

## 1. MX1 recap

MX1's live frontier mine found the symbolic-action layer is namespace-saturated:
the new Finset/Set ext/cases actions yielded **0 clean symbolic labels**. The only
2 new Set wins (`Set.Finite.toFinset_insert`, `Set.Finite.toFinset_offDiag`) were
`over_attributed_raw` — a plain `aesop` closes them. Production missed them only
because the Set route (`gen_v5_ns12_balanced`) carries no aesop fallback (unlike
Finset/NS21). MX2 tests whether a Set-gated aesop fallback captures them — a tiny
wrapper tweak, not symbolic training.

## 2. Candidate inventory (Stage 1)

`project/data/mx2_set_aesop_candidate_meta.json`. The 2 known aesop-misses plus
similar fresh Set lemmas: `Set.Finite.*` 65, `Set.image` 21, `Set.preimage` 7,
`Set.Finite.toFinset` 6 (99 total). The Set route has no aesop fallback.

## 3. Config design (Stage 2)

Both configs deep-copy the **NS9 best genome** (on-disk genome UNCHANGED) and add
`aesop` to `priority_templates` (all shapes) + a `theorem_name_tactic_gates`
aesop gate — mirroring NS19 `finset_aesop_only` but gated to Set:

- `mx2_set_aesop_safe.json` — **broad**, gate `{aesop: [Set.]}`.
- `mx2_set_finite_aesop_safe.json` — **narrow**, gate `{aesop: [Set.Finite., Set.toFinset]}`.

Disabled/unselected ⇒ production unchanged; any `aesop` tactic fires only on
gated names; additive to the ranked list.

## 4. Eval matrix (Stage 4, LIVE)

`project/data/mx2_set_aesop_probe_meta.json`. Variants A (production), B (broad),
C (narrow; run only where its gate can fire — elsewhere provably ≡ production).

| set | n | A prod | B broad (new/regr) | C narrow (new/regr) | B aesop emit/close |
|---|---|---|---|---|---|
| set_aesop_known | 2 | 0 | **2** / 0 | **2** / 0 | 2 / 2 |
| set_finite_frontier | 10 | 3 | 4 (+1) / 0 | 4 (+1) / 0 | 10 / 4 |
| set_aesop_frontier (image/preimage) | 10 | 4 | 4 (+0) / 0 | ≡A | 18 / 4 |
| set_negative_control | 9 | 0 | 0 (+0) / 0 | ≡A | 12 / **0** |
| mixed_preservation_control | 8 | 3 | 3 (+0) / 0 | ≡A | 10 / 3 |
| **total** | 39 | 10 | **+3** / **0** | **+3** / **0** | — |

- **Broad and narrow capture the SAME +3 new wins beyond production**, all on the
  `Set.Finite` surface. The broader Set surface (image/preimage) yields **no**
  extra aesop wins — production's generative policy already gets those 4.
- **0 regressions** for both, on every set.
- **Negative control**: broad aesop fires 12× and closes **0** — no false wins,
  no regressions, but wasted compute (the overfiring cost of the broad gate).
- **Mixed**: 0 regressions — the Set gate leaves Finset/List/Multiset untouched.

## 5. Minimal relabel (Stage 5, LIVE)

`project/data/mx2_set_aesop_minimal_labels.json`. Strict battery from the initial
state on each of the 3 new wins:

| theorem | minimal closer | class |
|---|---|---|
| `Set.Finite.toFinset_insert` | `aesop` | **clean_aesop** |
| `Set.Finite.toFinset_offDiag` | `aesop` | **clean_aesop** |
| `Set.Finite.to_subtype` | `assumption` | simpler_raw |

⇒ **2 genuinely clean-aesop wins** (the MX1 misses, confirmed); the 3rd is
`assumption`-closable (aesop subsumes it, so the fallback still catches it, but it
is not aesop-specific). These are FALLBACK wins, not symbolic labels — no training.

## 6. Preservation / runtime (Stage 6)

`project/data/mx2_preservation_matrix.json`. **Static: 0 non-Set aesop-admissible
across all preservation sets** (demo_v1, nat_defs_medium/large, ns17_set/finset,
ns14_set_finset, wx2_list, ax4_multiset_heldout) — the name-gate provably forbids
any aesop emission on a non-Set theorem. **Live regressions: 0** — established
by the eval matrix's own controls: `set_negative_control` (9 hard Set lemmas,
B=A=0, 0 regr) and `mixed_preservation_control` (Finset/List/Multiset/Set, B=A=3,
0 regr). Combined with additivity (aesop appended to the ranked list, never
replacing it), no production win can be lost. NS9 floors preserved
(genome byte-unchanged; aesop additive + Set-gated): medium 37/38, large 49/65,
demo 11/15. Runtime: aesop is expensive per call; the broad gate fires it on
*every* Set theorem (12× on the negative control alone, closing 0) — wasted
compute — whereas the narrow gate fires only on `Set.Finite`/`toFinset`.

## 7. Decision

**Gate B — keep as a narrow `Set.Finite`/`toFinset` aesop patch.**

- ≥2 confirmed Set wins beyond production: **yes** (2 clean aesop).
- Zero regressions: **yes** (broad and narrow, every set).
- Set-gated emissions only: **yes** (0 non-Set aesop, static guarantee).
- Runtime negligible: **only for the narrow gate.** The broad `Set.` gate adds
  **no** wins beyond narrow yet fires aesop across the whole Set surface (e.g.
  12× on the negative control, 0 closes) — non-negligible wasted compute.

**Recommendation:** adopt **`mx2_set_finite_aesop_safe`** (narrow) as the optional,
off-by-default Set fallback — it captures the 2 clean-aesop `Set.Finite.toFinset`
misses with minimal overfiring. Do **not** promote the broad `Set.` aesop gate:
same wins, more wasted aesop calls. Both remain experimental configs; production
(NS9 + WX3 + NS24, no Set aesop) is unchanged unless an MX2 config is selected.

This closes the MX1 thread: the Set misses were indeed best addressed by an
ordinary (narrowly-gated) `aesop` fallback — exactly as NS21 did for Finset —
not by symbolic learning. The broad sweep confirms there is no larger aesop
headroom on the fresh Set surface beyond the `Set.Finite.toFinset` family.
