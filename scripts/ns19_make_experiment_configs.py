"""NS19 Stage 1 + 3 — emit two name-gated wrapper variants as additive
deltas on the NS9 best genome.

- ns19_finset_aesop_only: same priority_templates as ns18_aesop_wrapper,
  but `aesop` is gated to theorems whose full_name starts with
  `Finset.`. The NS18 -1 regression on `Set.inter_singleton_eq_empty`
  should be eliminated; the +3 Finset wins should be preserved.

- ns19_nat_simp_arith_targeted: extends the NS18 nat_simp_arith
  variant with extra Nat-arithmetic simp_all bundles (Nat.add_*,
  Nat.mul_*, Nat.add_mod, Nat.mul_mod, Nat.mod_mul_left_mod) plus
  omega. All new tactics are gated to `Nat.` theorem names so the
  variant cannot pollute Set/Finset surfaces.

Both configs are experimental, additive, and never overwrite the NS9
best genome.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

NS18_DIR = Path("project/evolve/experiments/ns18")
NS19_DIR = Path("project/evolve/experiments/ns19")
NS18_DIR.mkdir(parents=True, exist_ok=True)
NS19_DIR.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict:
    return json.loads((NS18_DIR / name).read_text(encoding="utf-8"))


def variant_finset_aesop_only() -> dict:
    """Copy ns18_aesop_wrapper and add a Finset-only name gate for aesop.

    `aesop` is already present across all shapes in ns18_aesop_wrapper.
    The gate filters it back to Finset theorems only. No change to
    priority_templates, family tactics, or retrieval.
    """
    g = copy.deepcopy(load("ns18_aesop_wrapper.json"))
    gates = dict(g.get("theorem_name_tactic_gates") or {})
    gates["aesop"] = ["Finset."]
    g["theorem_name_tactic_gates"] = gates
    g["_ns19_variant"] = "finset_aesop_only"
    g["_ns19_gates"] = {"aesop": ["Finset."]}
    return g


def variant_nat_simp_arith_targeted() -> dict:
    """Extend ns18_nat_simp_arith with more Nat arithmetic bundles, all
    gated to `Nat.` names so other namespaces are unaffected.
    """
    g = copy.deepcopy(load("ns18_nat_simp_arith.json"))
    # Add additional simp_all bundles to priority_templates.any. We
    # avoid re-ordering existing templates and we keep all new tactics
    # gated by name prefix.
    extra_tactics = [
        "simp_all [Nat.add_comm, Nat.add_assoc, Nat.add_left_comm]",
        "simp_all [Nat.mul_comm, Nat.mul_assoc, Nat.mul_left_comm]",
        "simp_all [Nat.add_mod, Nat.mod_eq_of_lt]",
        "simp_all [Nat.mul_mod, Nat.mod_mul_left_mod]",
        "simp_all [Nat.add_mod, Nat.mul_mod]",
        "omega",
    ]
    pt = g.setdefault("priority_templates", {})
    any_slot = list(pt.get("any") or [])
    for t in extra_tactics:
        if t not in any_slot:
            any_slot.append(t)
    pt["any"] = any_slot
    g["priority_templates"] = pt
    g["priority_template_budget"] = max(
        int(g.get("priority_template_budget", 0) or 0), 32
    )
    # Gate every new tactic substring to Nat.* names. Pre-existing
    # tactics from NS9 / NS18 are untouched (their substrings are not
    # in the gate dict, so they fire on every name).
    gates = dict(g.get("theorem_name_tactic_gates") or {})
    for t in extra_tactics:
        gates[t] = ["Nat."]
    g["theorem_name_tactic_gates"] = gates
    g["_ns19_variant"] = "nat_simp_arith_targeted"
    g["_ns19_added_any"] = extra_tactics
    g["_ns19_gates"] = {t: ["Nat."] for t in extra_tactics}
    return g


def write(name: str, g: dict) -> Path:
    p = NS19_DIR / name
    p.write_text(json.dumps(g, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def main() -> None:
    p1 = write("ns19_finset_aesop_only.json", variant_finset_aesop_only())
    p2 = write("ns19_nat_simp_arith_targeted.json",
               variant_nat_simp_arith_targeted())
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
