"""AX2 Stage 3 — extract raw vs NS9 vs AX1-symbolic signal on fresh sets.

Reads the eval-run metrics for the three AX2 configs (all on ns24_router,
top-k 8 max-steps 8) over the fresh List sets:

  raw    : ax2_raw_<set>     (routed_generative, no wrapper)
  ns9    : ax2_ns9_<set>     (NS9 best genome — no symbolic block)
  ax1sym : ax2_ax1sym_<set>  (NS9 genome + AX1 symbolic_actions block)

Computes per-set wins, symbolic-only wins beyond NS9 and beyond raw,
regressions, and per-win attribution (winning tactic, symbolic action id
[template_source slot], family source, variable, namespace).

Output: project/data/ax2_symbolic_mining_probe_meta.json
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AX2_SETS = ["ax2_list_cases_fresh", "ax2_list_induction_fresh",
            "ax2_option_list_mixed_fresh"]
OUT = ROOT / "project/data/ax2_symbolic_mining_probe_meta.json"

_VAR_RE = re.compile(r"^\s*(?:cases|induction)\s+(\S+)\s+<;>")


def first(pat):
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def metrics(tag, s):
    f = first(str(ROOT / f"project/evolve/eval_runs/ax2_{tag}_{s}/eval-*/metrics.json"))
    return json.load(open(f)) if f else None


def pt(d):
    return {t["full_name"]: t for t in d.get("per_theorem", [])} if d else {}


def wins(d):
    return {n for n, t in pt(d).items() if t.get("finished")}


def var_from_tactic(tac):
    m = _VAR_RE.match(tac or "")
    return m.group(1) if m else None


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import tasks
    name_to_file = {}
    for _s, thms in tasks.THEOREM_SETS.items():
        for t in thms:
            name_to_file.setdefault(t.full_name, t.file_path)

    per_set = []
    sym_only_ns9_all = []
    sym_only_raw_all = []
    regressions_all = []
    origin_counts = Counter()
    action_id_counts = Counter()
    fam_counts = Counter()

    for s in AX2_SETS:
        raw_m, ns9_m, sym_m = metrics("raw", s), metrics("ns9", s), metrics("ax1sym", s)
        raw_w, ns9_w, sym_w = wins(raw_m), wins(ns9_m), wins(sym_m)
        sym_pt = pt(sym_m)
        sym_only_ns9 = sym_w - ns9_w
        sym_only_raw = sym_w - raw_w
        regress = ns9_w - sym_w

        for n in sorted(sym_only_ns9):
            b = sym_pt.get(n, {})
            origin = b.get("winning_tactic_origin") or "?"
            action_id = b.get("winning_tactic_template_source")
            fam = b.get("winning_tactic_family_source") or "?"
            tac = b.get("winning_tactic")
            origin_counts[origin] += 1
            if action_id:
                action_id_counts[action_id] += 1
            fam_counts[fam] += 1
            sym_only_ns9_all.append({
                "full_name": n, "file_path": name_to_file.get(n, ""),
                "namespace": n.split(".")[0], "set": s,
                "winning_tactic": tac,
                "winning_origin": origin,
                "symbolic_action_id": action_id,
                "winning_family": fam,
                "variable_selected": var_from_tactic(tac),
                "also_beyond_raw": n not in raw_w,
            })
        for n in sorted(sym_only_raw):
            sym_only_raw_all.append({"full_name": n, "set": s})
        for n in sorted(regress):
            regressions_all.append({"full_name": n, "set": s})

        per_set.append({
            "set": s,
            "available": sym_m.get("available") if sym_m else None,
            "total_theorems": sym_m.get("total_theorems") if sym_m else None,
            "raw_wins": len(raw_w), "ns9_wins": len(ns9_w),
            "symbolic_wins": len(sym_w),
            "symbolic_only_beyond_ns9": len(sym_only_ns9),
            "symbolic_only_beyond_raw": len(sym_only_raw),
            "regressions_vs_ns9": len(regress),
            "symbolic_only_beyond_ns9_theorems": sorted(sym_only_ns9),
            "regression_theorems": sorted(regress),
        })

    out = {
        "router": "ns24_router",
        "configs": {
            "raw": "routed_generative (no wrapper)",
            "ns9": "project/evolve/best/ns9_best_genome.json",
            "ax1sym": "project/evolve/experiments/ax1/ax1_symbolic_option_list_cases.json",
        },
        "per_set_summary": per_set,
        "totals": {
            "symbolic_only_beyond_ns9": len(sym_only_ns9_all),
            "symbolic_only_beyond_raw": len(sym_only_raw_all),
            "regressions_vs_ns9": len(regressions_all),
        },
        "symbolic_only_origin_counts": dict(origin_counts.most_common()),
        "symbolic_only_action_id_counts": dict(action_id_counts.most_common()),
        "symbolic_only_family_counts": dict(fam_counts.most_common()),
        "symbolic_only_by_namespace": dict(
            Counter(w["namespace"] for w in sym_only_ns9_all).most_common()),
        "regression_theorems": regressions_all,
        "symbolic_only_beyond_ns9_theorems": sym_only_ns9_all,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"{'set':28s} {'raw':>4} {'ns9':>4} {'sym':>4} "
          f"{'s>ns9':>6} {'s>raw':>6} {'regr':>5}")
    for r in per_set:
        print(f"{r['set']:28s} {r['raw_wins']:>4} {r['ns9_wins']:>4} "
              f"{r['symbolic_wins']:>4} {r['symbolic_only_beyond_ns9']:>6} "
              f"{r['symbolic_only_beyond_raw']:>6} {r['regressions_vs_ns9']:>5}")
    print(f"\nTOTAL symbolic-only beyond NS9: {len(sym_only_ns9_all)}  "
          f"beyond raw: {len(sym_only_raw_all)}  "
          f"regressions: {len(regressions_all)}")
    print(f"action ids: {dict(action_id_counts)}")
    print(f"origins: {dict(origin_counts)}")


if __name__ == "__main__":
    main()
