"""CX3 Stage 4 — extract raw-vs-wrapper Bool/Option probe signal.

Reads the cx3 eval runs (raw_routed vs wrap_routed under the NS24
router) for all five CX3 sets, computes per-set raw / wrapper /
wrapper-only wins, and emits the probe metadata plus a wrapper-only
theorem list (with file_path resolved from tasks) for Stage 5 relabel.

Writes:
  - project/data/cx3_bool_option_probe_meta.json
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CX3_SETS = [
    "cx3_bool_decide_easy",
    "cx3_bool_simp_medium",
    "cx3_option_simp_easy",
    "cx3_option_cases_medium",
    "cx3_bool_option_mixed",
]


def fam(t: str) -> str:
    """Coarse family label from a tactic string (Bool/Option flavored)."""
    t = re.sub(r"\s+", " ", (t or "").strip())
    if not t:
        return "empty"
    if t == "decide":
        return "fallback_decide"
    if t == "rfl":
        return "fallback_rfl"
    if t == "aesop":
        return "aesop"
    if t == "omega":
        return "fallback_omega"
    if t == "tauto":
        return "tauto"
    if t == "norm_num":
        return "norm_num"
    if t.startswith("simp_all"):
        return "simp_all"
    if t.startswith("simp"):
        return "simp_other"
    if t.startswith("cases") or t.startswith("rcases") or t.startswith("rintro"):
        return "cases_simp"
    if t.startswith("constructor"):
        return "constructor"
    return "other"


def first_match(pat: str) -> str | None:
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def load_per_thm(p: str | None) -> dict[str, dict]:
    if not p:
        return {}
    return {t["full_name"]: t
            for t in json.load(open(p)).get("per_theorem", [])}


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tasks
    name_to_file: dict[str, str] = {}
    for _set, thms in tasks.THEOREM_SETS.items():
        for t in thms:
            name_to_file.setdefault(t.full_name, t.file_path)

    # bucket hint from the theorem-set config (Stage 3)
    bucket_of: dict[str, str] = {}
    cfg_path = Path("project/evolve/routing/cx3_theorem_sets.json")
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        for _s, items in cfg.items():
            for it in items:
                bucket_of[it["full_name"]] = it.get("expected_bucket", "?")

    per_set_summary: list[dict] = []
    wrapper_only: list[dict] = []
    relabel_candidates: list[dict] = []
    all_raw_wins: list[str] = []
    all_wrap_wins: list[str] = []
    unavailable: list[str] = []
    raw_origin_counts: Counter = Counter()
    wrap_origin_counts: Counter = Counter()

    for s in CX3_SETS:
        raw = first_match(
            f"project/evolve/eval_runs/cx3_rawrouted_ns24_router_{s}/"
            "eval-*/metrics.json")
        wrap = first_match(
            f"project/evolve/eval_runs/cx3_wraprouted_ns24_router_{s}/"
            "eval-*/metrics.json")
        raw_pt = load_per_thm(raw)
        wrap_pt = load_per_thm(wrap)

        # availability: a theorem is unavailable if it didn't load in
        # EITHER run (available flag False in both).
        names = set(raw_pt) | set(wrap_pt)
        set_unavail = []
        for n in names:
            a = (raw_pt.get(n, {}).get("available")
                 or wrap_pt.get(n, {}).get("available"))
            if a is False or a is None:
                # only mark unavailable if explicitly False in the run(s)
                ra = raw_pt.get(n, {}).get("available")
                wa = wrap_pt.get(n, {}).get("available")
                if (ra is False or ra is None) and (wa is False or wa is None):
                    set_unavail.append(n)
        unavailable.extend(set_unavail)

        raw_win = {n for n, t in raw_pt.items() if t.get("finished")}
        wrap_win = {n for n, t in wrap_pt.items() if t.get("finished")}
        all_raw_wins.extend(raw_win)
        all_wrap_wins.extend(wrap_win)
        wonly = wrap_win - raw_win

        for n, t in raw_pt.items():
            if t.get("finished"):
                raw_origin_counts[t.get("winning_tactic_origin") or "?"] += 1
        for n, t in wrap_pt.items():
            if t.get("finished"):
                wrap_origin_counts[t.get("winning_tactic_origin") or "?"] += 1

        per_set_summary.append({
            "set": s,
            "total": len(tasks.get_theorems(s)),
            "available": len(names) - len(set_unavail),
            "unavailable": len(set_unavail),
            "raw_wins": len(raw_win),
            "wrap_wins": len(wrap_win),
            "wrapper_only": len(wonly),
            "raw_metrics": raw,
            "wrap_metrics": wrap,
        })

        for n in sorted(wonly):
            blob = wrap_pt.get(n, {})
            tac = (blob.get("winning_tactic") or blob.get("last_tactic")
                   or blob.get("tactic") or "")
            wrapper_only.append({
                "full_name": n,
                "file_path": name_to_file.get(n, ""),
                "namespace": n.split(".")[0],
                "original_family": fam(tac),
                "wrapper_tactic": tac,
                "wrapper_tactic_origin": blob.get("winning_tactic_origin"),
                "first_seen_set": s,
            })

        # All AVAILABLE theorems become relabel candidates, tagged by the
        # current routed-model solve status. With wrapper-only == 0 the
        # gate-relevant question is headroom: are the UNSOLVED theorems
        # closed by a short tactic the model simply isn't emitting?
        for n in sorted(names - set(set_unavail)):
            rb = raw_pt.get(n, {})
            wb = wrap_pt.get(n, {})
            tac = (wb.get("winning_tactic") or rb.get("winning_tactic") or "")
            relabel_candidates.append({
                "full_name": n,
                "file_path": name_to_file.get(n, ""),
                "namespace": n.split(".")[0],
                "original_family": fam(tac) if tac else "unsolved",
                "wrapper_tactic": tac,
                "currently_solved_raw": n in raw_win,
                "currently_solved_wrap": n in wrap_win,
                "expected_bucket": bucket_of.get(n, "?"),
                "first_seen_set": s,
            })

    # Family distribution over wrapper-only wins (pre-relabel).
    by_family = Counter(w["original_family"] for w in wrapper_only)
    by_ns = Counter(w["namespace"] for w in wrapper_only)

    out = {
        "router_used": "ns24_router",
        "wrapper_genome": "project/evolve/best/ns9_best_genome.json",
        "eval_settings": {"top_k": 8, "max_steps": 8},
        "per_set_summary": per_set_summary,
        "totals": {
            "raw_wins": len(set(all_raw_wins)),
            "wrap_wins": len(set(all_wrap_wins)),
            "wrapper_only": len(wrapper_only),
            "unavailable": len(set(unavailable)),
        },
        "raw_win_origin_counts": dict(raw_origin_counts.most_common()),
        "wrap_win_origin_counts": dict(wrap_origin_counts.most_common()),
        "wrapper_only_by_family_prerelabel": dict(by_family.most_common()),
        "wrapper_only_by_namespace": dict(by_ns.most_common()),
        "unavailable_theorems": sorted(set(unavailable)),
        "wrapper_only_theorems": wrapper_only,
        "relabel_candidates": relabel_candidates,
    }
    Path("project/data/cx3_bool_option_probe_meta.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print("wrote project/data/cx3_bool_option_probe_meta.json")
    print()
    print(f"{'set':28s} {'tot':>4} {'avail':>5} {'raw':>4} "
          f"{'wrap':>4} {'w-only':>6}")
    for r in per_set_summary:
        print(f"{r['set']:28s} {r['total']:>4} {r['available']:>5} "
              f"{r['raw_wins']:>4} {r['wrap_wins']:>4} "
              f"{r['wrapper_only']:>6}")
    print()
    print(f"TOTAL wrapper-only: {len(wrapper_only)}  "
          f"unavailable: {len(set(unavailable))}")
    print(f"wrapper-only by family (pre-relabel): {dict(by_family)}")
    print(f"wrapper-only by namespace: {dict(by_ns)}")


if __name__ == "__main__":
    main()
