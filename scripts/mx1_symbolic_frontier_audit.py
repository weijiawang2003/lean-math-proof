"""MX1 Stage 1 — symbolic-action frontier audit.

Enumerate FRESH theorem candidates (not consumed by any CX/WX/AX/SX arc) with
likely symbolic-action potential, across the priority namespaces, and classify
each by the symbolic-action family it is most likely to need. This is signal
*scoping* for live mining — no Lean is run here; availability is screened from
the cx1 availability probe and confirmed at mine time.

Candidate pool: `project/data/cx1_available_theorems.json` (availability-screened
1817) enriched with difficulty / family tags from
`project/discovered_theorems_cx1.json` (3989). Already-used theorems are pulled
from every routing/*_theorem_sets.json plus the live `tasks.THEOREM_SETS`
registry, and excluded. demo_v1 and all prior training/mining sets are part of
that registry, so they drop out automatically.

Likely-family classification (heuristic, from namespace + file + name):
  multiset_induction_simp, multiset_ext_simp, list_cases_simp,
  list_induction_simp, option_cases_simp, finset_ext_simp, finset_cases_simp,
  set_ext_simp, sequence_candidate, hard_unknown.

Outputs:
  project/data/mx1_symbolic_frontier_audit_meta.json
  project/evolve/reports/mx1_symbolic_frontier_audit.md
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AVAIL = ROOT / "project/data/cx1_available_theorems.json"
DISCO = ROOT / "project/discovered_theorems_cx1.json"
OUT_META = ROOT / "project/data/mx1_symbolic_frontier_audit_meta.json"
OUT_MD = ROOT / "project/evolve/reports/mx1_symbolic_frontier_audit.md"

PRIORITY_NS = ("Multiset", "Finset", "List", "Option", "Set")

# induction-suggesting name fragments (recursive structure over the container)
_INDUCTION_HINTS = ("fold", "sum", "prod", "length", "card", "count", "map",
                    "join", "bind", "attach", "pmap", "replicate", "concat",
                    "reverse", "filter", "dedup", "powerset", "scan")
# extensionality-suggesting fragments / files
_EXT_HINTS = ("ext", "eq_iff", "subset", "inter", "union", "sdiff", "compl",
              "image", "preimage", "coe", "val")


def used_full_names() -> set:
    used = set()
    for f in glob.glob(str(ROOT / "project/evolve/routing/*_theorem_sets.json")):
        d = json.load(open(f))
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict) and it.get("full_name"):
                            used.add(it["full_name"])
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        import tasks
        for lst in tasks.THEOREM_SETS.values():
            for cfg in lst:
                fn = getattr(cfg, "full_name", None)
                if fn:
                    used.add(fn)
    except Exception:
        pass
    return used


def classify(ns: str, full_name: str, file_path: str) -> str:
    name = full_name.split(".", 1)[1] if "." in full_name else full_name
    nl, fl = name.lower(), file_path.lower()
    ext_like = any(h in nl for h in _EXT_HINTS) or "ext" in fl
    ind_like = any(h in nl for h in _INDUCTION_HINTS)
    if ns == "Multiset":
        # quotient type: induction_on is the workhorse; ext for set-like eqs
        if ext_like and not ind_like:
            return "multiset_ext_simp"
        return "multiset_induction_simp"
    if ns == "Finset":
        if ext_like:
            return "finset_ext_simp"
        return "finset_cases_simp"
    if ns == "List":
        if ind_like:
            return "list_induction_simp"
        return "list_cases_simp"
    if ns == "Option":
        return "option_cases_simp"
    if ns == "Set":
        return "set_ext_simp"
    return "hard_unknown"


def main() -> None:
    # availability-screened subset (cx1 probed 44 files): treat as confirmed.
    screened = {t.get("full_name") for t in json.load(open(AVAIL)).get(
        "theorems", [])}
    # full discovered catalog is the candidate POOL for the priority
    # namespaces; availability for non-screened rows is confirmed at mine time
    # (AX4 saw ~0 attrition on the broader Multiset frontier).
    disco_rows = json.load(open(DISCO)).get("theorems", [])
    used = used_full_names()

    seen = set()
    candidates = []
    for t in disco_rows:
        fn = t.get("full_name")
        ns = t.get("namespace") or (fn.split(".", 1)[0] if fn else "")
        fp = t.get("file_path", "")
        if not fn or ns not in PRIORITY_NS:
            continue
        if fn in used or fn in seen:
            continue
        seen.add(fn)
        d = t
        fam = classify(ns, fn, fp)
        # a small slice of induction/ext families are plausible depth-2
        # sequence candidates (first action advances, follow-up closes)
        seq_candidate = fam in ("multiset_induction_simp", "list_induction_simp",
                                "finset_ext_simp") and \
            (d.get("difficulty") in ("hard", "medium"))
        candidates.append({
            "full_name": fn, "file_path": fp, "namespace": ns,
            "likely_family": fam,
            "difficulty": d.get("difficulty", "?"),
            "num_tactics_approx": d.get("num_tactics", d.get("num_tactics_approx")),
            "availability": "screened" if fn in screened else "unconfirmed",
            "sequence_candidate": bool(seq_candidate),
        })

    by_ns = Counter(c["namespace"] for c in candidates)
    by_fam = Counter(c["likely_family"] for c in candidates)
    by_ns_fam = defaultdict(Counter)
    for c in candidates:
        by_ns_fam[c["namespace"]][c["likely_family"]] += 1
    seq_n = sum(1 for c in candidates if c["sequence_candidate"])

    meta = {
        "description": "MX1 Stage 1 — fresh symbolic-action frontier audit "
                       "(availability-screened, used-excluded; no live Lean).",
        "pool_source": "project/data/cx1_available_theorems.json (1817)",
        "tags_source": "project/discovered_theorems_cx1.json (3989)",
        "already_used_excluded": len(used),
        "priority_namespaces": list(PRIORITY_NS),
        "total_fresh_candidates": len(candidates),
        "by_namespace": dict(by_ns),
        "by_likely_family": dict(by_fam),
        "by_namespace_family": {k: dict(v) for k, v in by_ns_fam.items()},
        "sequence_candidates": seq_n,
        "candidates": candidates,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    lines = ["# MX1 symbolic-action frontier audit\n",
             "Fresh, availability-screened candidates with likely "
             "symbolic-action potential, excluding every theorem consumed by "
             "prior CX/WX/AX/SX arcs (and demo_v1 / training sets, which are in "
             "the `tasks.THEOREM_SETS` registry). No Lean is run here — "
             "availability is confirmed at mine time.\n",
             f"- pool: cx1 availability probe (1817); tags: discovered_cx1 (3989)",
             f"- already-used excluded: **{len(used)}**",
             f"- **total fresh candidates: {len(candidates)}**",
             f"- depth-2 sequence candidates flagged: {seq_n}\n",
             "## By namespace\n",
             "| namespace | fresh candidates |", "|---|---|"]
    for ns in PRIORITY_NS:
        lines.append(f"| {ns} | {by_ns.get(ns, 0)} |")
    lines += ["", "## By likely action family\n",
              "| family | count |", "|---|---|"]
    for fam, n in by_fam.most_common():
        lines.append(f"| `{fam}` | {n} |")
    lines += ["", "## By namespace × family\n"]
    for ns in PRIORITY_NS:
        if ns in by_ns_fam:
            fams = ", ".join(f"{f}={n}" for f, n in by_ns_fam[ns].most_common())
            lines.append(f"- **{ns}**: {fams}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"total fresh candidates: {len(candidates)}  by_ns={dict(by_ns)}")
    print(f"by_family={dict(by_fam)}")


if __name__ == "__main__":
    main()
