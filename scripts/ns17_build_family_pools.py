"""NS17 Stage 6 — family-pool aggregation + Stage 7 readiness gate.

Combines the family audit (``project/data/ns17_family_audit.json``)
with the per-surface wrapper-only diagnostics from Stage 5 to emit:

  - ``project/data/ns17_family_pools_meta.json`` — per-family aggregate
    with suggested oversample factor and NS18 readiness.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path


AUDIT_PATH = Path("project/data/ns17_family_audit.json")
NS17_SURFACES = (
    "ns17_nat_remaining", "ns17_set_extra",
    "ns17_finset_extra", "ns17_list_multiset",
)


def first_match(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_solved(path: str) -> set[str]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t["full_name"] for t in m.get("per_theorem", [])
            if t.get("finished")}


def compute_ns17_wrapper_only() -> dict[str, dict]:
    """Per NS17 surface: raw / wrapper / wrapper-only theorem lists."""
    out: dict[str, dict] = {}
    for s in NS17_SURFACES:
        raw = first_match(f"project/evolve/eval_runs/ns17_ns15routed_raw_{s}/eval-*/metrics.json")
        wrap = first_match(f"project/evolve/eval_runs/ns17_ns15routed_wrapper_{s}/eval-*/metrics.json")
        if not raw or not wrap:
            continue
        raw_solved = load_solved(raw)
        wrap_solved = load_solved(wrap)
        wo = sorted(wrap_solved - raw_solved)
        out[s] = {
            "raw_solved": len(raw_solved),
            "wrap_solved": len(wrap_solved),
            "wrapper_only_count": len(wo),
            "wrapper_only_theorems": wo,
        }
    return out


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    evolved = audit["evolved"]["families"]
    traces = audit["traces_close_only"]["families"]

    ns17_wo = compute_ns17_wrapper_only()
    total_ns17_wrapper_only = sum(v["wrapper_only_count"] for v in ns17_wo.values())

    pools: dict[str, dict] = {}
    for fam in set(list(evolved) + list(traces)):
        ev = evolved.get(fam, {})
        tr = traces.get(fam, {})
        ev_rows = ev.get("rows", 0)
        ev_thms = ev.get("unique_theorems", 0)
        ev_wo = ev.get("wrapper_only_rows", 0)
        tr_rows = tr.get("rows", 0)
        tr_thms = tr.get("unique_theorems", 0)
        # Suggested oversample factor to hit ~100 examples in a
        # combined pool of (ev_rows + tr_rows). If pool is already
        # >= 100, factor is 1.
        pool = ev_rows + tr_rows
        suggested_oversample = max(1, (100 + pool - 1) // max(1, pool))
        # Strong NS18 candidate if either: ≥ 10 wrapper-only rows
        # in evolved OR ≥ 20 unique theorems in evolved+trace
        # combined AND pool size ≥ 10.
        unique_thms = ev_thms + tr_thms  # may double-count if same thm
        gate_pass = (
            (ev_wo >= 10) or
            (pool >= 20 and unique_thms >= 20)
        )
        pools[fam] = {
            "evolved_rows": ev_rows,
            "evolved_unique_theorems": ev_thms,
            "evolved_wrapper_only": ev_wo,
            "evolved_held_out": ev.get("held_out_rows", 0),
            "trace_rows": tr_rows,
            "trace_unique_theorems": tr_thms,
            "combined_pool_size": pool,
            "suggested_oversample_factor": suggested_oversample,
            "ns18_gate_pass": gate_pass,
            "example_tactic": (ev.get("example_tactics") or
                               tr.get("example_tactics") or [""])[0],
            "by_namespace": ev.get("by_namespace") or tr.get("by_namespace") or {},
        }

    meta = {
        "ns17_surface_wrapper_only": ns17_wo,
        "ns17_wrapper_only_total": total_ns17_wrapper_only,
        "decision_gate_criteria": {
            "rule_1": "evolved wrapper-only rows >= 10",
            "rule_2": "pool size >= 20 AND unique theorems >= 20",
        },
        "family_pools": pools,
        "families_passing_gate": sorted(
            f for f, info in pools.items() if info["ns18_gate_pass"]
        ),
    }
    out_path = Path("project/data/ns17_family_pools_meta.json")
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("NS17 surfaces (raw vs wrapper):")
    for s, info in ns17_wo.items():
        print(f"  {s:>26}: raw={info['raw_solved']:>2} "
              f"wrap={info['wrap_solved']:>2} "
              f"WO={info['wrapper_only_count']:>2}")
    print(f"  TOTAL wrapper-only on NS17 surfaces: {total_ns17_wrapper_only}")
    print()

    print(f"{'family':>26} {'evRows':>7} {'evThms':>6} {'evWO':>5} "
          f"{'trRows':>6} {'pool':>5} {'over':>5} {'gate':>6}")
    for fam in sorted(pools, key=lambda x: -pools[x]["combined_pool_size"]):
        info = pools[fam]
        gate = "PASS" if info["ns18_gate_pass"] else "fail"
        print(f"{fam:>26} {info['evolved_rows']:>7} "
              f"{info['evolved_unique_theorems']:>6} "
              f"{info['evolved_wrapper_only']:>5} "
              f"{info['trace_rows']:>6} "
              f"{info['combined_pool_size']:>5} "
              f"{info['suggested_oversample_factor']:>5}x "
              f"{gate:>5}")

    print()
    print(f"families passing gate: {meta['families_passing_gate']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
