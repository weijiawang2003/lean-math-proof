#!/usr/bin/env python3
"""SF5 Part 5 — generate retrieval-guided proof probes.

For each target and its retrieved lemmas, emit conservative single-line probes:
  exact <lemma> / simpa using <lemma> / simp [<lemma>] / simpa [<lemma>] /
  rw [<lemma>] (iff/eq-shaped only) / aesop (add simp [<lemma>]) (parse-risk flagged)

Per-target diagnostics that ask "does *any* existing lemma close this":
  exact? / apply?    (library search; flagged diagnostic)

Cluster-level probes:
  simp only [l1, l2, ...]   over the cluster's top shared retrieved lemmas

Limits: ≤10 lemmas/target, ≤40 probes/target. Malformed names skipped; parse risk
recorded so the live runner / attribution can discount parse_error noise.
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


def _rw_safe(stmt):
    """rw is only safe for iff/eq-shaped rewrite lemmas."""
    if not stmt:
        return False
    s = stmt
    # crude top-level shape test: contains ↔ or a top-level '='
    return ("↔" in s) or (" = " in s) or s.strip().endswith("=")


def _malformed(name):
    return not _NAME_RE.match(name or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--targets",
                    default="project/evolve/experiments/sf5/cases/"
                            "sf5_missing_bridge_targets.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--max-lemmas", type=int, default=10)
    ap.add_argument("--max-probes", type=int, default=40)
    args = ap.parse_args()

    ret = json.load(open(_p(args.retrieval)))
    tg_meta = {t["full_name"]: t for t in json.load(open(_p(args.targets)))}

    targets_out = []
    total_probes = 0
    for r in ret["results"]:
        target = r["target"]
        probes = []
        seen = set()

        def add(tactic, family, lemma, parse_risk="low", diagnostic=False):
            if tactic in seen:
                return False
            if len(probes) >= args.max_probes:
                return False
            seen.add(tactic)
            probes.append({
                "tactic": tactic, "family": family, "lemma": lemma,
                "parse_risk": parse_risk, "diagnostic": diagnostic,
            })
            return True

        cid = r.get("cluster_id")
        # reserve high-value slots up front so the 40-cap never crowds them out:
        # diagnostics (decisive "does any existing lemma close it") + cluster probes.
        add("exact?", "diagnostic_search", None, parse_risk="low", diagnostic=True)
        add("apply?", "diagnostic_search", None, parse_risk="low", diagnostic=True)
        shared = [s["lemma"] for s in
                  ret.get("cluster_shared_lemmas", {}).get(cid, {})
                  .get("shared_retrieved_lemmas", [])
                  if not _malformed(s["lemma"]) and s["lemma"] != target][:6]
        if shared:
            add("simp only [" + ", ".join(shared) + "]", "cluster_simp_only",
                None, parse_risk="low")
            add("simp [" + ", ".join(shared[:4]) + "]", "cluster_simp",
                None, parse_risk="low")

        # definitional-unfold family: `simp [Def1, Def2, ...]` over retrieved *definitions*.
        # Many iff-of-predicate goals (e.g. MonotoneOn f s ↔ Monotone ...) close by
        # unfolding the predicate definitions, which are not @[simp] so bare simp fails.
        # precise channel first: defs whose name literally appears in the goal
        # (e.g. MonotoneOn ↔ Monotone -> simp [MonotoneOn, Monotone]); then fall back
        # to lexically-retrieved defs.
        goal_defs = list(r.get("goal_defs", []))
        retr_defs = [c["lemma"] for c in r.get("retrieved_defs", [])] + [
            c["lemma"] for c in r["retrieved"] if c.get("decl_kind") in ("def", "abbrev")]
        seen_defs, defs = set(), []
        for l in goal_defs + retr_defs:
            if l not in seen_defs and not _malformed(l) and l != target:
                seen_defs.add(l)
                defs.append(l)
        # the goal-driven combined unfold is the high-value probe
        if goal_defs:
            gd = [d for d in goal_defs if not _malformed(d) and d != target][:4]
            if gd:
                add("simp [" + ", ".join(gd) + "]", "def_unfold_simp",
                    "+".join(gd), parse_risk="low")
        for d in defs[:3]:
            add(f"simp [{d}]", "def_unfold_simp", d, parse_risk="low")

        lemmas = [c for c in r["retrieved"] if not _malformed(c["lemma"])][: args.max_lemmas]
        for rank, c in enumerate(lemmas):
            lemma = c["lemma"]
            stmt = c.get("statement_text")
            add(f"exact {lemma}", "exact", lemma)
            add(f"simpa using {lemma}", "simpa_using", lemma)
            add(f"simp [{lemma}]", "simp_lemma", lemma)
            if _rw_safe(stmt):
                add(f"rw [{lemma}]", "rw_lemma", lemma)
            # aesop(add simp) is slow + parse-risky: restrict to the top-3 lemmas
            if rank < 3:
                add(f"aesop (add simp [{lemma}])", "aesop_add_simp", lemma,
                    parse_risk="medium")

        total_probes += len(probes)
        meta = tg_meta.get(target, {})
        targets_out.append({
            "full_name": target,
            "file_path": meta.get("file_path"),
            "namespace": r.get("namespace"),
            "cluster_id": cid,
            "goal_text": r.get("goal_text"),
            "num_probes": len(probes),
            "num_lemmas_used": len(lemmas),
            "probes": probes,
        })

    # backfill file_path from the targets cases file (retrieval json lacks it directly)
    out = {
        "generated_by": "scripts/sf5_generate_retrieval_probes.py",
        "limits": {"max_lemmas": args.max_lemmas, "max_probes": args.max_probes},
        "num_targets": len(targets_out),
        "total_probes": total_probes,
        "theorems": targets_out,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# SF5 retrieval probe plan", "",
          f"- targets: {len(targets_out)} | total probes: {total_probes}",
          f"- limits: ≤{args.max_lemmas} lemmas, ≤{args.max_probes} probes per target", ""]
    fam = {}
    for t in targets_out:
        for p in t["probes"]:
            fam[p["family"]] = fam.get(p["family"], 0) + 1
    md.append("## Probe families")
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        md.append(f"- {k}: {v}")
    md.append("")
    for t in targets_out[:6]:
        md.append(f"### {t['full_name']} ({t['num_probes']} probes)")
        for p in t["probes"][:8]:
            tag = " [diag]" if p["diagnostic"] else ""
            md.append(f"- `{p['tactic']}`{tag}")
        md.append("")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")

    print(f"[sf5-probes] {len(targets_out)} targets, {total_probes} probes, families={fam}")


if __name__ == "__main__":
    main()
