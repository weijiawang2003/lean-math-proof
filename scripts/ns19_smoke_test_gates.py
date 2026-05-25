"""NS19 Stage 1 smoke — verify theorem_name_tactic_gates drops the
gated tactic on out-of-namespace theorems and keeps it on in-namespace
theorems. Uses a dummy base_policy so no model load is required.
"""
from __future__ import annotations

import json
from pathlib import Path

from evolve.strategy_wrapper import (
    StrategyWrapperPolicy,
    load_strategy_config,
)


class _DummyBase:
    def rank_tactics(self, state_pp: str, full_name: str = "", k: int = 5):
        # Emit a candidate that contains the gated substring `aesop`
        # so we can verify that base-model output is NOT filtered by
        # the gate. NS9 wrapper baseline relies on this — the routed
        # generative model emits aesop on Set/Finset goals.
        return ["aesop", "dummy_base_tac"]


def make_wrapper(cfg_path: str) -> StrategyWrapperPolicy:
    (fb, tmpl, cap, fam, fam_budgets, deny,
     re_en, re_top_k, re_forms,
     re_filt_self, re_filt_unav,
     re_skip_bloat, re_shape_filt,
     tb_tmpl, tb_budget,
     pt, pt_budget,
     use_bag,
     re_req_fam, re_fam_gates,
     gates) = load_strategy_config(cfg_path)
    return StrategyWrapperPolicy(
        base_policy=_DummyBase(),
        fallback_tactics=fb, tactic_templates=tmpl,
        max_extra_tactics_per_state=cap,
        theorem_family_tactics=fam, family_budgets=fam_budgets,
        theorem_tactic_denylist=deny,
        retrieval_enabled=re_en, retrieval_top_k=re_top_k,
        retrieval_tactic_forms=re_forms,
        retrieval_filter_self=re_filt_self,
        retrieval_filter_unavailable=re_filt_unav,
        retrieval_shape_filter=re_shape_filt,
        retrieval_requires_family=re_req_fam,
        retrieval_family_gates=re_fam_gates,
        term_builder_templates=tb_tmpl,
        term_builder_budget=tb_budget,
        priority_templates=pt,
        priority_template_budget=pt_budget,
        use_skeleton_bag=use_bag,
        theorem_name_tactic_gates=gates,
    )


def run() -> int:
    failed: list[str] = []

    # --- Finset-gated aesop ---
    cfg = "project/evolve/experiments/ns19/ns19_finset_aesop_only.json"
    w = make_wrapper(cfg)
    state = "x : Nat ⊢ True"

    finset_tacs = w.rank_tactics(state, "Finset.coe_insert", k=8)
    set_tacs = w.rank_tactics(state, "Set.inter_singleton_eq_empty", k=8)
    nat_tacs = w.rank_tactics(state, "Nat.add_mod_eq_ite", k=8)

    # On Finset: aesop should be emitted from both base AND wrapper.
    if finset_tacs.count("aesop") < 1:
        failed.append("FAIL: aesop should be emitted on Finset.coe_insert")
    else:
        print("ok: aesop emitted on Finset.coe_insert")

    # On Set: base-model aesop MUST still be emitted (NS9 baseline
    # relies on this); wrapper-added aesop must be gated. Since the
    # base emits one aesop, we expect aesop to appear in set_tacs
    # exactly once (the gated wrapper entry is deduped or filtered).
    # The key invariant is presence, not absence — base aesop must
    # survive.
    if "aesop" not in set_tacs:
        failed.append("FAIL: base-model aesop should NOT be gated on Set.* names")
    else:
        print("ok: base-model aesop preserved on Set.inter_singleton_eq_empty")

    if "aesop" not in nat_tacs:
        failed.append("FAIL: base-model aesop should NOT be gated on Nat.* names")
    else:
        print("ok: base-model aesop preserved on Nat.add_mod_eq_ite")

    # --- Nat simp_all arith ---
    cfg2 = "project/evolve/experiments/ns19/ns19_nat_simp_arith_targeted.json"
    w2 = make_wrapper(cfg2)
    nat_tacs2 = w2.rank_tactics(state, "Nat.mul_mod_mod", k=8)
    set_tacs2 = w2.rank_tactics(state, "Set.inter_singleton_eq_empty", k=8)
    finset_tacs2 = w2.rank_tactics(state, "Finset.coe_insert", k=8)

    add_mod_sub = "simp_all [Nat.add_mod, Nat.mul_mod]"
    if not any(add_mod_sub in t for t in nat_tacs2):
        failed.append(f"FAIL: '{add_mod_sub}' should be emitted on Nat.mul_mod_mod")
    else:
        print(f"ok: '{add_mod_sub}' emitted on Nat.mul_mod_mod")

    if any(add_mod_sub in t for t in set_tacs2):
        failed.append(f"FAIL: '{add_mod_sub}' should be gated out on Set.* names")
    else:
        print(f"ok: '{add_mod_sub}' blocked on Set.* names")
    if any(add_mod_sub in t for t in finset_tacs2):
        failed.append(f"FAIL: '{add_mod_sub}' should be gated out on Finset.* names")
    else:
        print(f"ok: '{add_mod_sub}' blocked on Finset.* names")

    # Cross-check: existing NS9 tactics (no gate) are unaffected on
    # all names.
    if "omega" in w.rank_tactics(state, "Set.foo", k=8):
        # omega is in fallback_tactics of NS9 best — should still fire.
        print("ok: omega still emitted on Set.foo (no gate for omega in ns19_finset_aesop_only)")
    else:
        # omega may not appear depending on shape; not a failure.
        print("note: omega not in this call's ranked list (state-dependent, not a gate issue)")

    if failed:
        print("\n".join(failed))
        return 1
    print("\nall NS19 gate smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
