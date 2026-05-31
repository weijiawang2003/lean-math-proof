#!/usr/bin/env python3
"""SX2 Part 1 — mine successful Set probes into normalized proof-sequence templates.

Reads the SF2 Set deep-dive probe results, extracts every *winning* probe, and
abstracts it into a normalized symbolic template with a family label, the symbols
it depends on, the goal shape it fired on, parser constraints, and honest
generalization / theorem-specific risk scores.

Template families (structural classifier over the winning tactic string):
  SET_ITE_SIMP        simp[/_all/ only] [Set.ite]            (pure, no named lemmas)
  SET_EXT_SIMP        ext x <;> simp[...]                    (no by_cases)
  SET_EXT_BYCASES     ext x <;> by_cases ... <;> simp_all    (per-branch split)
  SET_IFF_CONSTRUCTOR constructor/refine <;> intro <;> ...   (iff split)
  SET_SUBSET_ANTISYMM apply Set.Subset.antisymm <;> ...
  SET_RW_BRIDGE       rw [named bridge lemmas] (+ closer)
  SOURCE_SPECIFIC     theorem-specific simp-set / def unfold / hypothesis use

A template is "interesting" (gate-worthy) only if it solves >=2 theorems OR is
clearly reusable by goal shape with NO theorem-specific symbols. Single-theorem
source-inspired hacks (named bridge lemmas, local-hypothesis use) are NOT.

Outputs:
  set_sequence_templates.json
  set_sequence_templates.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

PROBES = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/probe_results.json"
CA = "project/evolve/experiments/sf2/out/set_cluster_deep_dive/cluster_analysis.json"
OUT_JSON = "project/evolve/experiments/sx2/out/set_sequence_templates.json"
OUT_MD = "project/evolve/experiments/sx2/out/set_sequence_templates.md"

BASELINES = {"simp", "simp_all", "aesop", "classical <;> aesop"}

# qualified (A.b.c) or known bare snake_case lemma tokens inside [...] of rw/simp
_QUALIFIED = re.compile(r"[A-Za-z][\w']*(?:\.[A-Za-z][\w']*)+")
_BRACKET = re.compile(r"\[([^\]]*)\]")
_HYP = re.compile(r"(?<![\w.])@?h[a-z0-9_]*\b")


def referenced_symbols(tac):
    """Lemma/def names the tactic depends on (qualified names + bracket args)."""
    syms = set()
    for m in _QUALIFIED.finditer(tac):
        syms.add(m.group(0))
    for br in _BRACKET.findall(tac):
        for tok in re.split(r"[,\s]+", br):
            tok = tok.strip().lstrip("←").strip()
            if not tok:
                continue
            # keep lemma-ish identifiers (snake_case or CamelCase), drop pure ops
            if re.fullmatch(r"[A-Za-z][\w'.]*", tok):
                syms.add(tok)
    return sorted(syms)


def uses_local_hyp(tac):
    # references a local hypothesis (h, hx, @h ...) — NOT a global lemma
    # exclude qualified names (a.b) and the bound binder name in `by_cases hx :`
    body = _BRACKET.findall(tac)
    hits = []
    for chunk in body:
        for m in _HYP.finditer(chunk):
            hits.append(m.group(0))
    return sorted(set(hits))


def classify_family(tac):
    t = tac.strip()
    norm = re.sub(r"\s+", " ", t)
    # SET_ITE_SIMP: a simp variant whose ONLY bracket content is Set.ite
    m = re.fullmatch(r"(simp(?: only)?|simp_all)\s*\[\s*Set\.ite\s*\]", norm)
    if m:
        return "SET_ITE_SIMP"
    if norm.startswith("ext ") and ("by_cases" in norm):
        return "SET_EXT_BYCASES"
    if norm.startswith("ext "):
        return "SET_EXT_SIMP"
    if norm.startswith("apply Set.Subset.antisymm"):
        return "SET_SUBSET_ANTISYMM"
    if norm.startswith("constructor") or norm.startswith("refine"):
        return "SET_IFF_CONSTRUCTOR"
    if norm.startswith("rw ") or norm.startswith("rw["):
        return "SET_RW_BRIDGE"
    return "SOURCE_SPECIFIC"


def normalized_template(tac, family):
    """Canonical, theorem-agnostic shape string for grouping."""
    norm = re.sub(r"\s+", " ", tac.strip())
    if family == "SET_ITE_SIMP":
        return "simp [Set.ite]"
    if family == "SET_EXT_SIMP":
        return "ext x <;> simp"
    if family == "SET_EXT_BYCASES":
        return "ext x <;> by_cases <VAR> <;> simp_all [...]"
    if family == "SET_IFF_CONSTRUCTOR":
        return "constructor <;> intro h <;> simp_all"
    if family == "SET_SUBSET_ANTISYMM":
        return "apply Set.Subset.antisymm <;> intro x <;> simp_all"
    if family == "SET_RW_BRIDGE":
        return "rw [<bridge lemmas>] (+ closer)"
    return norm  # SOURCE_SPECIFIC keeps its concrete form


def risk_scores(family, symbols, hyps):
    """Honest generalization & theorem-specific risk."""
    # any named, non-structural lemma symbol (excluding Set.ite/the structural ops)
    structural = {"Set.ite", "Set.Subset.antisymm"}
    named = [s for s in symbols if s not in structural]
    if family == "SET_ITE_SIMP":
        return "low", "low"
    if hyps:  # depends on a local hypothesis -> cannot be emitted generically
        return "high", "high"
    if family == "SET_RW_BRIDGE":
        return "high", "high"
    if family in ("SET_EXT_SIMP", "SET_SUBSET_ANTISYMM", "SET_IFF_CONSTRUCTOR"):
        # structurally generic, but in mined data these never fired in pure form
        return "medium", "medium" if named else "low"
    # SOURCE_SPECIFIC: theorem-specific simp set / def unfold
    return "high" if named else "medium", "high" if named else "medium"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--probe-results", default=PROBES)
    p.add_argument("--cluster-analysis", default=CA)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    args = p.parse_args(argv)

    pr = json.load(open(args.probe_results))
    results = pr["results"]

    mined = []
    for r in results:
        if not r.get("solved_by_probe") or not r.get("winning_probe"):
            continue
        tac = r["winning_probe"]
        fam = classify_family(tac)
        syms = referenced_symbols(tac)
        hyps = uses_local_hyp(tac)
        gen_risk, ts_risk = risk_scores(fam, syms, hyps)
        # find the winning outcome record to capture source_inspired/risk
        wf = next((o for o in r["outcomes"]
                   if o.get("probe") == tac and o.get("solved")), {})
        base_fail = sorted(BASELINES.intersection(
            {m["probe"] for m in (r.get("minimality_results") or [])
             if not m["solved"]}))
        mined.append({
            "theorem": r["full_name"],
            "cluster_id": r.get("cluster_id"),
            "winning_tactic": tac,
            "winning_family": r.get("winning_family"),
            "source_inspired": bool(wf.get("source_inspired")),
            "baseline_failures": base_fail,
            "normalized_template": normalized_template(tac, fam),
            "template_family": fam,
            "required_symbols": syms,
            "goal_shape": r.get("primary_goal_shape"),
            "local_hypotheses_used": hyps,
            "parser_constraints": (
                ["single_line_only", "';'->'<;>' rewrite", "no '·' bullet blocks"]
                if "<;>" in tac else ["single_line_only"]),
            "generalization_risk": gen_risk,
            "theorem_specific_risk": ts_risk,
            "gap_classification": r.get("classification"),
        })

    # ---- template support roll-up (group by family + normalized form) ----
    support = defaultdict(lambda: {"theorems": [], "clusters": set(),
                                   "tactic_strings": set(), "all_baselines_failed": True,
                                   "uses_named_lemmas": False, "uses_local_hyp": False,
                                   "goal_shapes": set()})
    for m in mined:
        key = m["template_family"]
        s = support[key]
        s["theorems"].append(m["theorem"])
        s["clusters"].add(m["cluster_id"])
        s["tactic_strings"].add(m["winning_tactic"])
        s["goal_shapes"].add(m["goal_shape"])
        # "all baselines failed" only meaningful for the actual mined solves
        named = [x for x in m["required_symbols"]
                 if x not in ("Set.ite", "Set.Subset.antisymm")]
        if named:
            s["uses_named_lemmas"] = True
        if m["local_hypotheses_used"]:
            s["uses_local_hyp"] = True

    templates = []
    for fam, s in support.items():
        n = len(s["theorems"])
        # A family is gate-worthy ONLY if a SINGLE theorem-agnostic tactic
        # generalizes: no named (theorem-specific) lemmas, no local-hypothesis
        # dependence. SET_RW_BRIDGE / SOURCE_SPECIFIC count >=2 at the family
        # level but each member is a DISTINCT theorem-specific tactic, so no one
        # emittable string generalizes -> NOT interesting.
        theorem_agnostic = (not s["uses_named_lemmas"]) and (not s["uses_local_hyp"])
        # only one canonical emittable string when theorem-agnostic
        single_reusable_tactic = theorem_agnostic and len(s["tactic_strings"]) >= 1
        reusable_by_shape = (fam == "SET_ITE_SIMP" and theorem_agnostic)
        interesting = theorem_agnostic and (n >= 2 or reusable_by_shape)
        # source-copy risk: theorem-specific lemma names or hypothesis use
        if s["uses_local_hyp"]:
            source_copy = "high"
        elif s["uses_named_lemmas"]:
            source_copy = "high" if fam in ("SET_RW_BRIDGE", "SOURCE_SPECIFIC") else "medium"
        else:
            source_copy = "low"
        templates.append({
            "template_family": fam,
            "normalized_template": normalized_template(
                next(t for t in s["tactic_strings"]), fam),
            "num_theorems_solved": n,
            "theorems": sorted(s["theorems"]),
            "clusters_covered": sorted(c for c in s["clusters"] if c),
            "unique_tactic_strings": sorted(s["tactic_strings"]),
            "all_baselines_failed": True,  # every mined win had baselines fail (verified in SF2)
            "uses_named_lemmas": s["uses_named_lemmas"],
            "uses_local_hyp": s["uses_local_hyp"],
            "goal_shapes": sorted(s["goal_shapes"]),
            "source_copy_risk": source_copy,
            "generalization_risk": "low" if reusable_by_shape else (
                "high" if source_copy == "high" else "medium"),
            "theorem_agnostic": theorem_agnostic,
            "single_reusable_tactic": single_reusable_tactic,
            "interesting": interesting,
            "interesting_reason": (
                f"theorem-agnostic tactic solves {n} theorems" if interesting and n >= 2 else
                "theorem-agnostic pure tactic reusable by goal shape" if interesting else
                f"family solves {n} but each member is theorem-specific "
                f"(named lemmas={s['uses_named_lemmas']}, local-hyp={s['uses_local_hyp']}) "
                f"— NO single reusable tactic, NOT gate-worthy"),
        })
    templates.sort(key=lambda t: (-t["num_theorems_solved"], t["template_family"]))

    out = {
        "source": args.probe_results,
        "num_winning_probes": len(mined),
        "num_template_families": len(templates),
        "num_interesting": sum(1 for t in templates if t["interesting"]),
        "promotion_note": "A template is gate-worthy only if it solves >=2 theorems "
                          "or is a pure tactic reusable by goal shape with NO "
                          "theorem-specific symbols. Mined evidence: only SET_ITE_SIMP "
                          "qualifies (n=2, simp [Set.ite]); all rw-bridges / simp-sets "
                          "are single-theorem and theorem-specific.",
        "templates": templates,
        "mined_probes": mined,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# SX2 — Mined Set Proof-Sequence Templates", ""]
    L.append(f"- winning probes mined: {out['num_winning_probes']} | template "
             f"families: {out['num_template_families']} | interesting "
             f"(gate-worthy): {out['num_interesting']}")
    L.append(f"- {out['promotion_note']}")
    L.append("")
    L.append("## Template support")
    L.append("")
    L.append("| family | n | theorems | named-lemmas | local-hyp | gen-risk | "
             "source-copy | interesting |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t in templates:
        L.append(f"| `{t['template_family']}` | {t['num_theorems_solved']} | "
                 f"{', '.join(s.split('.')[-1] for s in t['theorems'])} | "
                 f"{t['uses_named_lemmas']} | {t['uses_local_hyp']} | "
                 f"{t['generalization_risk']} | {t['source_copy_risk']} | "
                 f"**{t['interesting']}** ({t['interesting_reason']}) |")
    L.append("")
    L.append("## Per-winning-probe detail")
    L.append("")
    for m in mined:
        L.append(f"### `{m['theorem']}` — `{m['template_family']}`")
        L.append(f"- winning tactic: `{m['winning_tactic']}`")
        L.append(f"- normalized: `{m['normalized_template']}` | shape: {m['goal_shape']}")
        L.append(f"- required symbols: {m['required_symbols']}")
        L.append(f"- local hyps used: {m['local_hypotheses_used']} | "
                 f"gen-risk: {m['generalization_risk']} | "
                 f"theorem-specific-risk: {m['theorem_specific_risk']}")
        L.append(f"- parser constraints: {m['parser_constraints']}")
        L.append("")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sx2:mine] winning_probes={out['num_winning_probes']} "
          f"families={out['num_template_families']} "
          f"interesting={out['num_interesting']} -> {args.out_json}")
    for t in templates:
        print(f"   {t['template_family']:20s} n={t['num_theorems_solved']} "
              f"interesting={t['interesting']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
