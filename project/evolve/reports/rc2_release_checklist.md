# RC2 Release Checklist

| item | status |
|---|---|
| RC1 wrapper preserved (untouched) | ✅ `rc1_production_wrapper.json` empty diff |
| NS24 router preserved | ✅ `ns24_router.json` empty diff |
| NS9 genome/checkpoints preserved | ✅ untouched |
| REL1 release reports preserved | ✅ untouched |
| RC2 wrapper created as NEW artifact | ✅ `project/evolve/experiments/rc2_release/rc2_production_wrapper.json` |
| Exactly one added component (SET_ITE_SIMP) | ✅ `simp [Set.ite]` in priority_templates["any"] + name-gate |
| Speculative gates disabled | ✅ none present (SET_EXT_SIMP/SUBSET_ANTISYMM/IFF_CONSTRUCTOR/EXT_BYCASES/RW_BRIDGE/SOURCE_SPECIFIC) |
| Canonical floors pass | ✅ demo_v1 11/15, nat_defs_medium 37/38, nat_defs_large_v5 49/65 |
| Credited delta confirmed | ✅ +5 single-shot SET_ITE_SIMP, minimal-relabel 5/5 TRUE |
| Off-gate emissions zero | ✅ 0 (gate name-prefixed to Set.ite) |
| Regressions zero | ✅ 0 (RC2≡RC1 on non-Set.ite by construction) |
| Determinism confirmed | ✅ hash-stable across runs |
| SX3 deferred (not in RC2) | ✅ 4 depth-2 sequence candidates deferred |
| README updated (RC2 recommended) | ✅ section added; RC1 history preserved |
| Reproduction commands verified | ✅ `rc2_reproduction_commands.md` + final_verification |
| Committed | ❌ NOT committed (working tree only) |

Release status: **RC2 release frozen, commit-ready, not committed.**
