"""NS20 Stage 4 — extract the consolidated Finset/aesop wrapper-only
training pool across NS18 + NS19 + NS20.

For each truly-new wrapper-only theorem (variant proved AND NS9
wrap did NOT prove AND family=='aesop'), record:
  - full_name
  - source_set (which surface produced the win)
  - source_arc (NS18/NS19/NS20)
  - winning_tactic
  - raw_solved (was raw NS15 routed able to solve it without wrapper?)
  - wrap_solved (was NS9 wrap baseline able?)
  - variant_solved (yes — that's why it's here)

Aggregate into project/data/ns20_finset_aesop_pool_meta.json:
  unique_count, row_count, recommended_oversample_factor,
  whether the NS20 training gate (>=5 unique) is met, namespace
  breakdown, source-arc breakdown.

The script does NOT build training JSONL — per NS20 spec, training
is deferred to NS21 unless the pool is small and trivial.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path


OUT_PATH = Path("project/data/ns20_finset_aesop_pool_meta.json")


def first_match(pattern: str) -> str | None:
    ms = sorted(glob.glob(pattern))
    return ms[0] if ms else None


def load_solved(path: str | None) -> set[str]:
    if not path:
        return set()
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t["full_name"] for t in m.get("per_theorem", []) if t.get("finished")}


def load_per_thm(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    return {t["full_name"]: t for t in
            json.loads(Path(path).read_text(encoding="utf-8")).get("per_theorem", [])}


def winning_tactic(blob: dict) -> str:
    return (
        blob.get("winning_tactic")
        or blob.get("last_tactic")
        or blob.get("tactic")
        or ""
    )


# (arc_tag, variant_dir_pattern_template, raw_pattern_template, wrap_pattern_template, set_name)
# We declare every (variant, set) pair that could carry a Finset/aesop win.
SOURCES = [
    # NS18 aesop_wrapper on its evaluated sets
    ("NS18", "ns18_aesop_wrapper_wrapper", "ns17_finset_extra",
     "ns17_ns15routed_raw_ns17_finset_extra",
     "ns17_ns15routed_wrapper_ns17_finset_extra"),
    # NS19 finset_aesop_only on its evaluated sets
    ("NS19", "ns19_finset_aesop_only_wrapper", "ns19_finset_aesop_surface",
     "ns19_raw_ns19_finset_aesop_surface",
     "ns19_ns9wrap_ns19_finset_aesop_surface"),
    ("NS19", "ns19_finset_aesop_only_wrapper", "ns17_finset_extra",
     "ns17_ns15routed_raw_ns17_finset_extra",
     "ns17_ns15routed_wrapper_ns17_finset_extra"),
    # NS20 finset_aesop_only on the new surfaces
    ("NS20", "ns20_finset_aesop_only_wrapper", "ns20_finset_aesop_extra_easy",
     "ns20_raw_ns20_finset_aesop_extra_easy",
     "ns20_ns9wrap_ns20_finset_aesop_extra_easy"),
    ("NS20", "ns20_finset_aesop_only_wrapper", "ns20_finset_aesop_extra_medium",
     "ns20_raw_ns20_finset_aesop_extra_medium",
     "ns20_ns9wrap_ns20_finset_aesop_extra_medium"),
    ("NS20", "ns20_finset_aesop_only_wrapper", "ns20_finset_aesop_extra_hard",
     "ns20_raw_ns20_finset_aesop_extra_hard",
     "ns20_ns9wrap_ns20_finset_aesop_extra_hard"),
]


def main() -> None:
    rows: list[dict] = []
    seen: set[str] = set()
    for arc, var_pat, set_name, raw_pat, wrap_pat in SOURCES:
        vp = first_match(
            f"project/evolve/eval_runs/{var_pat}_{set_name}/eval-*/metrics.json"
        )
        if not vp:
            continue
        rp = first_match(f"project/evolve/eval_runs/{raw_pat}/eval-*/metrics.json")
        wp = first_match(f"project/evolve/eval_runs/{wrap_pat}/eval-*/metrics.json")
        v_thms = load_per_thm(vp)
        v_solved = {n for n, t in v_thms.items() if t.get("finished")}
        raw_solved = load_solved(rp)
        wrap_solved = load_solved(wp)
        truly_new = sorted(v_solved - wrap_solved)
        for thm in truly_new:
            if not thm.startswith("Finset."):
                continue
            tac = winning_tactic(v_thms.get(thm) or {})
            if tac.strip() != "aesop":
                continue  # only collect bare-aesop wins
            if thm in seen:
                # Already accounted from an earlier surface; record
                # additional source for traceability but don't double-
                # count for the gate.
                for r in rows:
                    if r["full_name"] == thm:
                        r["also_proved_in"].append({"arc": arc, "set": set_name})
                continue
            seen.add(thm)
            rows.append({
                "full_name": thm,
                "winning_tactic": "aesop",
                "source_arc": arc,
                "source_set": set_name,
                "raw_solved": thm in raw_solved,
                "wrap_solved": thm in wrap_solved,
                "variant_solved": True,
                "also_proved_in": [],
            })

    unique = len(rows)
    rows_total = unique + sum(len(r["also_proved_in"]) for r in rows)
    if unique <= 1:
        os_f = 20
    elif unique <= 3:
        os_f = 15
    elif unique <= 6:
        os_f = 10
    elif unique <= 12:
        os_f = 5
    else:
        os_f = 2

    out = {
        "family": "aesop",
        "namespace": "Finset",
        "training_gate_unique_required": 5,
        "unique_count": unique,
        "rows_total": rows_total,
        "trainable": unique >= 5,
        "recommended_oversample_factor": os_f,
        "tactic_strings_sample": ["aesop"],
        "rows": rows,
        "namespace_breakdown": {"Finset": unique},
        "source_arc_breakdown": _count_by(rows, "source_arc"),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"\nFinset/aesop wrapper-only pool (NS18+NS19+NS20):")
    print(f"  unique theorems: {unique}")
    print(f"  rows (incl. duplicates across surfaces): {rows_total}")
    print(f"  recommended oversample: {os_f}x")
    print(f"  gate met (>=5): {out['trainable']}")
    print(f"  source-arc breakdown: {out['source_arc_breakdown']}")
    print("\n  theorems:")
    for r in rows:
        also = " | also: " + ", ".join(f"{x['arc']}:{x['set']}" for x in r["also_proved_in"]) if r["also_proved_in"] else ""
        print(f"    [{r['source_arc']}] {r['full_name']} (from {r['source_set']}){also}")


def _count_by(rows: list[dict], key: str) -> dict[str, int]:
    d: dict[str, int] = {}
    for r in rows:
        d[r[key]] = d.get(r[key], 0) + 1
    return d


if __name__ == "__main__":
    main()
