"""NS18 Stage 1 — emit experimental wrapper strategy configs.

For each NS18 variant (A–F), this script reads the NS9 best genome
(``project/evolve/best/ns9_best_genome.json``), applies a small
additive patch, and writes the result to
``project/evolve/experiments/ns18/ns18_<variant>.json``.

The NS9 best genome is **not** modified. Every variant is a delta
on top of it, so any variant that fails smoke test or regresses
the canonical benchmarks can be discarded without touching the
production genome.

Variants:

  A. ns18_constructor_omega     — adds ``constructor <;> omega``
                                  templates to "iff".
  B. ns18_split_ifs_omega       — adds ``split_ifs <;> simp_all``
                                  to "any".
  C. ns18_nat_simp_arith        — adds compact Nat-simp bundles
                                  to "eq" / "any".
  D. ns18_aesop_wrapper         — adds ``aesop`` to every shape.
  E. ns18_bool_option_cases     — adds ``cases _ <;> simp_all``
                                  and ``decide``.
  F. ns18_combined_safe         — union of A + B + C + E (no
                                  aesop until smoke confirmed
                                  safe).

Usage::
    python scripts/ns18_make_experiment_configs.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


BASE_PATH = Path("project/evolve/best/ns9_best_genome.json")
OUT_DIR = Path("project/evolve/experiments/ns18")


def load_base() -> dict:
    return json.loads(BASE_PATH.read_text(encoding="utf-8"))


def add_priority(g: dict, shape: str, tactics: list[str]) -> None:
    g.setdefault("priority_templates", {})
    existing = g["priority_templates"].get(shape, [])
    seen = set(existing)
    for t in tactics:
        if t in seen:
            continue
        existing.append(t)
        seen.add(t)
    g["priority_templates"][shape] = existing


def add_family(g: dict, family: str, tactics: list[str]) -> None:
    g.setdefault("theorem_family_tactics", {})
    existing = g["theorem_family_tactics"].get(family, [])
    seen = set(existing)
    for t in tactics:
        if t in seen:
            continue
        existing.append(t)
        seen.add(t)
    g["theorem_family_tactics"][family] = existing


def add_fallback(g: dict, tactics: list[str]) -> None:
    existing = g.get("fallback_tactics", [])
    seen = set(existing)
    for t in tactics:
        if t in seen:
            continue
        existing.append(t)
        seen.add(t)
    g["fallback_tactics"] = existing


def variant_constructor_omega(base: dict) -> dict:
    g = copy.deepcopy(base)
    add_priority(g, "iff", [
        "constructor <;> omega",
        "constructor <;> simp_all",
        "constructor <;> (intro _ <;> omega)",
    ])
    g["_ns18_variant"] = "constructor_omega"
    g["_ns18_added"] = {
        "iff": [
            "constructor <;> omega",
            "constructor <;> simp_all",
            "constructor <;> (intro _ <;> omega)",
        ],
    }
    # Increase priority_template_budget so the new candidates get
    # surfaced. NS9 best is 18; we bump to 24 to allow ~6 extra slots.
    g["priority_template_budget"] = max(g.get("priority_template_budget", 0), 24)
    return g


def variant_split_ifs_omega(base: dict) -> dict:
    g = copy.deepcopy(base)
    add_priority(g, "any", [
        "split_ifs <;> simp_all",
        "split_ifs with h <;> omega",
        "split_ifs with h <;> simp_all [h]",
    ])
    g["_ns18_variant"] = "split_ifs_omega"
    g["_ns18_added"] = {
        "any": [
            "split_ifs <;> simp_all",
            "split_ifs with h <;> omega",
            "split_ifs with h <;> simp_all [h]",
        ],
    }
    g["priority_template_budget"] = max(g.get("priority_template_budget", 0), 24)
    return g


def variant_nat_simp_arith(base: dict) -> dict:
    g = copy.deepcopy(base)
    add_priority(g, "eq", [
        "simp_all [Nat.add_comm, Nat.add_assoc, Nat.add_left_comm]",
        "simp_all [Nat.mul_comm, Nat.mul_assoc, Nat.mul_left_comm]",
        "simp_all [Nat.add_mod, Nat.mod_eq_of_lt]",
    ])
    add_priority(g, "any", [
        "simp_all",
    ])
    add_family(g, "mod", [
        "simp_all [Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_lt]",
    ])
    g["_ns18_variant"] = "nat_simp_arith"
    g["_ns18_added"] = {
        "eq": [
            "simp_all [Nat.add_comm, Nat.add_assoc, Nat.add_left_comm]",
            "simp_all [Nat.mul_comm, Nat.mul_assoc, Nat.mul_left_comm]",
            "simp_all [Nat.add_mod, Nat.mod_eq_of_lt]",
        ],
        "any": ["simp_all"],
        "mod_family": ["simp_all [Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_lt]"],
    }
    g["priority_template_budget"] = max(g.get("priority_template_budget", 0), 24)
    return g


def variant_aesop(base: dict) -> dict:
    g = copy.deepcopy(base)
    for shape in ("iff", "eq", "lt", "le", "any"):
        add_priority(g, shape, ["aesop"])
    g["_ns18_variant"] = "aesop_wrapper"
    g["_ns18_added"] = {"all_shapes": ["aesop"]}
    g["priority_template_budget"] = max(g.get("priority_template_budget", 0), 24)
    return g


def variant_bool_option_cases(base: dict) -> dict:
    g = copy.deepcopy(base)
    # These templates target Bool/Option/List by structure. The
    # state pp will only match these for the right hyps; mistargets
    # are gated by Lean error.
    add_priority(g, "any", [
        "decide",
        "rfl",
    ])
    add_priority(g, "eq", [
        "rfl",
    ])
    g["_ns18_variant"] = "bool_option_cases"
    g["_ns18_added"] = {
        "any": ["decide", "rfl"],
        "eq": ["rfl"],
    }
    g["priority_template_budget"] = max(g.get("priority_template_budget", 0), 24)
    return g


def variant_combined_safe(base: dict) -> dict:
    """Union of A + B + C + E (skip D until smoke-passed)."""
    g = copy.deepcopy(base)
    # A
    add_priority(g, "iff", [
        "constructor <;> omega",
        "constructor <;> simp_all",
        "constructor <;> (intro _ <;> omega)",
    ])
    # B
    add_priority(g, "any", [
        "split_ifs <;> simp_all",
        "split_ifs with h <;> omega",
    ])
    # C
    add_priority(g, "eq", [
        "simp_all [Nat.add_comm, Nat.add_assoc, Nat.add_left_comm]",
        "simp_all [Nat.add_mod, Nat.mod_eq_of_lt]",
    ])
    add_priority(g, "any", ["simp_all"])
    add_family(g, "mod", [
        "simp_all [Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_lt]",
    ])
    # E
    add_priority(g, "any", ["decide", "rfl"])
    g["_ns18_variant"] = "combined_safe"
    g["_ns18_added"] = {
        "from_variants": ["constructor_omega", "split_ifs_omega",
                          "nat_simp_arith", "bool_option_cases"],
    }
    # Bigger budget — more candidates need more slots.
    g["priority_template_budget"] = max(g.get("priority_template_budget", 0), 30)
    return g


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base()
    variants = {
        "constructor_omega": variant_constructor_omega,
        "split_ifs_omega": variant_split_ifs_omega,
        "nat_simp_arith": variant_nat_simp_arith,
        "aesop_wrapper": variant_aesop,
        "bool_option_cases": variant_bool_option_cases,
        "combined_safe": variant_combined_safe,
    }
    for name, fn in variants.items():
        g = fn(base)
        out_path = OUT_DIR / f"ns18_{name}.json"
        out_path.write_text(json.dumps(g, indent=2), encoding="utf-8")
        added = g.get("_ns18_added", {})
        n_added = sum(len(v) for v in added.values() if isinstance(v, list))
        print(f"  ns18_{name}: +{n_added} tactic candidates → {out_path}")


if __name__ == "__main__":
    main()
