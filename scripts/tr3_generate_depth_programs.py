#!/usr/bin/env python3
"""TR3 Part 6 — generate gated retrieval-aware bounded-depth proof programs.

Per confirmed RC2 failure, emit deterministic depth-1/2/3 programs seeded with
retrieved lemmas, gated by goal shape:
  Set equality        -> ext / Subset.antisymm probes
  Set iff             -> constructor / intro probes
  Nat / arithmetic    -> omega / nlinarith
  Multiset.toFinset   -> toFinset / mem_toFinset simp
Plus a definitional-unfold family (`simp [Def, DefOn]` over goal-named defs — the
proven SF5 win mechanism) and lemma-free depth-only controls.

Limits: <=10 retrieved lemmas, <=60 programs/target, depth<=3, deterministic order,
no source-specific scripts. Emission order favours cheap/high-yield families first so
`--stop-after-win` terminates early while still recording skipped programs.
"""
from __future__ import annotations

import argparse
import json
import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.]*$")


def _p(*a):
    return os.path.join(_REPO, *a)


def _malformed(name):
    return not _NAME_RE.match(name or "")


def _rw_safe(stmt):
    if not stmt:
        return False
    return ("↔" in stmt) or (" = " in stmt) or stmt.strip().endswith("=")


def _shape(goal, namespace, features):
    g = goal or ""
    low = g.lower()
    f = features or {}
    has_iff = ("↔" in g) or f.get("has_iff")
    # set equality: two Set-typed sides of '='
    has_set = f.get("has_set") or ("set " in low) or namespace == "Set"
    has_eq = (" = " in g) or f.get("has_eq")
    is_set_eq = bool(has_set and has_eq and not has_iff)
    is_iff = bool(has_iff)
    is_subset = bool(("⊆" in g) or f.get("has_subset"))
    is_nat = bool(namespace == "Nat" or f.get("has_nat") or f.get("has_arith"))
    is_multiset_tofinset = bool(f.get("has_tofinset") or namespace == "Multiset"
                                or "tofinset" in low)
    return {"set_eq": is_set_eq, "iff": is_iff, "subset": is_subset,
            "nat": is_nat, "multiset_tofinset": is_multiset_tofinset}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--confirmation", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--max-lemmas", type=int, default=10)
    ap.add_argument("--max-programs", type=int, default=60)
    args = ap.parse_args()

    ret = {r["target"]: r for r in json.load(open(_p(args.retrieval)))["results"]}
    conf = json.load(open(_p(args.confirmation)))
    failures = [r for r in conf["results"] if r["classification"] == "CONFIRMED_RC2_FAILURE"]

    targets_out = []
    total = 0
    fam_counter = {}
    for fr in failures:
        fn = fr["full_name"]
        r = ret.get(fn, {})
        goal = fr.get("goal_text") or ""
        shape = _shape(goal, fr.get("namespace"), fr.get("features") or {})
        programs = []
        seen = set()

        def add(tactic, family, depth, lemmas, risk, gate_reason):
            if tactic in seen or len(programs) >= args.max_programs:
                return
            seen.add(tactic)
            programs.append({
                "program_id": f"{fn}::{len(programs):02d}",
                "target": fn, "family": family, "depth": depth,
                "lemmas": lemmas, "tactic": tactic,
                "gate": {"shape": [k for k, v in shape.items() if v], "reason": gate_reason},
                "risk": risk,
            })
            fam_counter[family] = fam_counter.get(family, 0) + 1

        # retrieved theorem-kind lemmas (skip defs; defs go to def-unfold)
        lemmas = [t for t in r.get("top_lemmas", [])
                  if not _malformed(t["lemma"]) and t.get("decl_kind") != "def"
                  and t["lemma"] != fn][: args.max_lemmas]
        goal_defs = [d for d in r.get("goal_defs", []) if not _malformed(d) and d != fn][:4]

        # ---- (1) definitional unfold (proven SF5 family) ----
        if goal_defs:
            add("simp [" + ", ".join(goal_defs) + "]", "def_unfold_simp", 1,
                goal_defs, "low", "goal-named definitions")

        # ---- (2) depth-1 retrieval ----
        for t in lemmas[:6]:
            L = t["lemma"]
            add(f"exact {L}", "d1_exact", 1, [L], "low", "direct term")
            add(f"simpa using {L}", "d1_simpa_using", 1, [L], "low", "simp-normalize")
            add(f"simp [{L}]", "d1_simp_lemma", 1, [L], "low", "simp with lemma")
            add(f"simpa [{L}]", "d1_simpa_lemma", 1, [L], "low", "simpa with lemma")
            if _rw_safe(t.get("statement_text")):
                add(f"rw [{L}]", "d1_rw_lemma", 1, [L], "low", "iff/eq rewrite")

        # ---- (3) lemma-free depth-only controls (gated) ----
        if shape["set_eq"]:
            add("ext x <;> aesop", "d2_ext_aesop", 2, [], "medium", "set equality")
            add("apply Set.Subset.antisymm <;> intro x <;> aesop", "d3_antisymm_aesop", 3,
                [], "high", "set equality via antisymm")
        if shape["iff"]:
            add("constructor <;> intro h <;> aesop", "d3_constructor_aesop", 3, [], "high",
                "iff split")
            add("constructor <;> intro h <;> simp_all", "d3_constructor_simp_all", 3, [], "high",
                "iff split")
        if shape["nat"]:
            add("omega", "d1_omega", 1, [], "low", "nat/arith")
            add("nlinarith", "d1_nlinarith", 1, [], "medium", "nat/arith")
        if shape["multiset_tofinset"]:
            add("simp [Multiset.toFinset, Multiset.mem_toFinset]", "d1_tofinset_simp", 1, [],
                "low", "multiset toFinset")
        add("aesop", "d1_aesop", 1, [], "low", "control")
        add("simp_all", "d1_simp_all", 1, [], "low", "control")
        add("tauto", "d1_tauto", 1, [], "low", "control")

        # ---- (4) depth-2 retrieval-aware ----
        for t in lemmas[:3]:
            L = t["lemma"]
            add(f"simp [{L}] <;> aesop", "d2_simp_aesop", 2, [L], "medium", "simp+aesop")
            add(f"simp [{L}] <;> simp_all", "d2_simp_simpall", 2, [L], "medium", "simp+simp_all")
            if _rw_safe(t.get("statement_text")):
                add(f"rw [{L}] <;> aesop", "d2_rw_aesop", 2, [L], "medium", "rw+aesop")
                add(f"rw [{L}] <;> simp_all", "d2_rw_simpall", 2, [L], "medium", "rw+simp_all")
            if shape["iff"]:
                add(f"constructor <;> intro h <;> simpa using {L}", "d2_constructor_simpa", 2,
                    [L], "medium", "iff split + lemma")
            if shape["set_eq"]:
                add(f"ext x <;> simp [{L}]", "d2_ext_simp", 2, [L], "medium", "ext + lemma")

        # ---- (5) depth-3 conservative ----
        for t in lemmas[:2]:
            L = t["lemma"]
            if shape["set_eq"]:
                add(f"ext x <;> simp [{L}] <;> aesop", "d3_ext_simp_aesop", 3, [L], "high",
                    "ext + lemma + aesop")
            add(f"simp [{L}] <;> try aesop <;> try simp_all", "d3_simp_try", 3, [L], "high",
                "simp + fallbacks")

        total += len(programs)
        targets_out.append({
            "full_name": fn, "file_path": fr.get("file_path"), "namespace": fr.get("namespace"),
            "cluster_id": fr.get("cluster_id"), "goal_text": goal,
            "shape": shape, "num_programs": len(programs), "programs": programs,
        })

    out = {"generated_by": "scripts/tr3_generate_depth_programs.py",
           "limits": {"max_lemmas": args.max_lemmas, "max_programs": args.max_programs,
                      "max_depth": 3},
           "num_targets": len(targets_out), "total_programs": total,
           "family_histogram": dict(sorted(fam_counter.items(), key=lambda kv: -kv[1])),
           "theorems": targets_out}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR3 depth-program plan", "",
          f"- targets: {len(targets_out)} | total programs: {total}", "",
          "## Family histogram"]
    for k, v in out["family_histogram"].items():
        md.append(f"- {k}: {v}")
    md += ["", "## Sample (first 3 targets)"]
    for t in targets_out[:3]:
        md.append(f"### {t['full_name']} ({t['num_programs']} programs, shape "
                  f"{[k for k,v in t['shape'].items() if v]})")
        for pgm in t["programs"][:10]:
            md.append(f"- d{pgm['depth']} `{pgm['tactic']}` [{pgm['family']}]")
        md.append("")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr3-programs] {len(targets_out)} targets, {total} programs; "
          f"families={out['family_histogram']}")


if __name__ == "__main__":
    main()
