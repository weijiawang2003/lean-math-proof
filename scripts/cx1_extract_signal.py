"""CX1 Stage 6 — wrapper-only signal extraction.

For each (variant, CX1 set) pair, compares variant proofs against
raw NS15 routed and NS9 best-wrapper baselines. Emits per-theorem
metadata for every wrapper-only win:

  - theorem
  - namespace
  - theorem_set
  - raw_solved (bool)
  - wrap_solved (bool)
  - variant_solved (bool — True by construction)
  - winning_tactic
  - tactic_origin (when traces expose it; otherwise null)
  - family (greedy classifier)
  - shape (from theorem-name pattern matching)
  - homogeneous_candidate (bool — family has ≥3 wrapper-only wins
    in the same namespace across CX1)
  - trainable (bool — homogeneous + variant_solved + not raw_solved)

Aggregates to project/data/cx1_wrapper_only_signal_meta.json and
writes project/evolve/reports/cx1_wrapper_only_signal_summary.md.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path


SURFACES_ALL = (
    "cx1_finset_image_filter",
    "cx1_nat_gcd_dvd_mod",
    "cx1_list_multiset",
    "cx1_bool_option_int",
    "cx1_mixed_easy",
    "cx1_mixed_medium",
)

# (variant_label, eval_dir_template, applies_to_sets)
VARIANTS = [
    ("finset_aesop_only",
     "cx1_finset_aesop_only_wrapper",
     ("cx1_finset_image_filter", "cx1_mixed_easy", "cx1_mixed_medium")),
]


def first_match(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_per_thm(path: str) -> dict[str, dict]:
    return {t["full_name"]: t for t in
            json.loads(Path(path).read_text(encoding="utf-8")).get("per_theorem", [])}


def load_solved(path: str) -> set[str]:
    return {t["full_name"] for t in
            json.loads(Path(path).read_text(encoding="utf-8")).get("per_theorem", [])
            if t.get("finished")}


def family_of(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip())
    if not t: return "empty"
    if t == "omega": return "fallback_omega"
    if t == "aesop": return "aesop"
    if t == "decide": return "fallback_decide"
    if t == "rfl": return "fallback_rfl"
    if t.startswith("constructor") and "omega" in t: return "constructor_omega"
    if t.startswith("split_ifs"): return "split_ifs"
    if "fun h => by omega" in t and t.count("by omega") >= 2: return "iff_omega_pair"
    if t.startswith("simp_all"): return "simp_all"
    if t.startswith("simp"): return "simp_other"
    if t.startswith("rw"): return "rw_named"
    if t.startswith("exact"): return "exact_named"
    if t.startswith("apply"): return "apply_named"
    return "other"


def shape_of(name: str) -> str:
    last = name.split(".", 1)[1].lower() if "." in name else name.lower()
    if "iff" in last: return "iff"
    if "_eq_" in last or last.endswith("_eq") or last.startswith("eq_"): return "eq"
    if "_le_" in last or last.startswith("le_") or last.endswith("_le"): return "le"
    if "_lt_" in last or last.startswith("lt_") or last.endswith("_lt"): return "lt"
    if "mem" in last: return "mem"
    if "subset" in last: return "subset"
    if "card" in last: return "card"
    if "comm" in last: return "comm"
    if "assoc" in last: return "assoc"
    if "image" in last: return "image"
    if "filter" in last: return "filter"
    if "empty" in last: return "empty"
    return "other"


def winning_tactic(blob: dict) -> str:
    return (
        blob.get("winning_tactic")
        or blob.get("last_tactic")
        or blob.get("tactic")
        or blob.get("proof")
        or blob.get("result_tactic")
        or ""
    )


def main() -> None:
    rows: list[dict] = []
    raw_paths: dict[str, str | None] = {}
    wrap_paths: dict[str, str | None] = {}
    var_paths: dict[tuple[str, str], str | None] = {}

    for s in SURFACES_ALL:
        raw_paths[s] = first_match(
            f"project/evolve/eval_runs/cx1_raw_{s}/eval-*/metrics.json"
        )
        wrap_paths[s] = first_match(
            f"project/evolve/eval_runs/cx1_ns9wrap_{s}/eval-*/metrics.json"
        )
    for variant, dir_tpl, applies in VARIANTS:
        for s in applies:
            var_paths[(variant, s)] = first_match(
                f"project/evolve/eval_runs/{dir_tpl}_{s}/eval-*/metrics.json"
            )

    # For each (variant, set), enumerate truly-new-vs-raw wins.
    for variant, _dir_tpl, applies in VARIANTS:
        for s in applies:
            vp = var_paths.get((variant, s))
            rp = raw_paths.get(s)
            wp = wrap_paths.get(s)
            if not vp:
                continue
            v_per_thm = load_per_thm(vp)
            v_solved = {n for n, t in v_per_thm.items() if t.get("finished")}
            raw_solved = load_solved(rp) if rp else set()
            wrap_solved = load_solved(wp) if wp else set()
            wrapper_only = sorted(v_solved - raw_solved)
            for thm in wrapper_only:
                blob = v_per_thm.get(thm) or {}
                tac = winning_tactic(blob)
                fam = family_of(tac) if tac else "unknown"
                ns = thm.split(".", 1)[0] if "." in thm else "?"
                rows.append({
                    "theorem": thm,
                    "namespace": ns,
                    "theorem_set": s,
                    "variant": variant,
                    "raw_solved": thm in raw_solved,
                    "wrap_solved": thm in wrap_solved,
                    "variant_solved": True,
                    "winning_tactic": tac,
                    "tactic_origin": blob.get("winning_tactic_origin"),
                    "family": fam,
                    "shape": shape_of(thm),
                })

    # Also collect NS9-wrap-only wins (raw didn't solve, NS9 did) —
    # these are part of the "wrapper-only signal" the spec asks for,
    # independent of CX1 variants.
    for s in SURFACES_ALL:
        wp = wrap_paths.get(s)
        rp = raw_paths.get(s)
        if not wp or not rp:
            continue
        wrap_per_thm = load_per_thm(wp)
        raw_solved = load_solved(rp)
        wrap_only = {n for n, t in wrap_per_thm.items()
                     if t.get("finished") and n not in raw_solved}
        for thm in sorted(wrap_only):
            blob = wrap_per_thm.get(thm) or {}
            tac = winning_tactic(blob)
            fam = family_of(tac) if tac else "unknown"
            ns = thm.split(".", 1)[0] if "." in thm else "?"
            rows.append({
                "theorem": thm,
                "namespace": ns,
                "theorem_set": s,
                "variant": "ns9_wrap",
                "raw_solved": False,
                "wrap_solved": True,
                "variant_solved": True,
                "winning_tactic": tac,
                "tactic_origin": blob.get("winning_tactic_origin"),
                "family": fam,
                "shape": shape_of(thm),
            })

    # Per-(family, namespace) homogeneity tally.
    family_ns_count: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        family_ns_count[(r["family"], r["namespace"])].add(r["theorem"])

    for r in rows:
        unique = len(family_ns_count[(r["family"], r["namespace"])])
        r["homogeneous_candidate"] = unique >= 3
        r["family_ns_unique_count"] = unique
        r["trainable"] = bool(r["homogeneous_candidate"]
                              and not r["raw_solved"]
                              and r["variant_solved"])

    out = {
        "rows": rows,
        "row_count": len(rows),
        "unique_theorems": len({r["theorem"] for r in rows}),
        "family_ns_breakdown": {
            f"{f}|{ns}": len(thms)
            for (f, ns), thms in family_ns_count.items()
        },
        "raw_paths": raw_paths,
        "wrap_paths": wrap_paths,
        "variant_paths": {f"{k[0]}|{k[1]}": v for k, v in var_paths.items()},
    }
    out_path = Path("project/data/cx1_wrapper_only_signal_meta.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines: list[str] = ["# CX1 — wrapper-only signal summary\n"]
    lines.append(f"**Rows**: {out['row_count']}   "
                 f"**Unique theorems**: {out['unique_theorems']}\n")
    lines.append("## Per-(family, namespace) wrapper-only counts\n")
    lines.append("| family | namespace | unique wins | homogeneous? |")
    lines.append("|---|---|---:|:---:|")
    ordered = sorted(family_ns_count.items(),
                     key=lambda kv: -len(kv[1]))
    for (f, ns), thms in ordered:
        homo = "✔" if len(thms) >= 3 else ""
        lines.append(f"| `{f}` | `{ns}` | {len(thms)} | {homo} |")
    lines.append("")
    md_path = Path("project/evolve/reports/cx1_wrapper_only_signal_summary.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"wrote {md_path}")
    print(f"\nTop families × namespaces by unique wins:")
    for (f, ns), thms in ordered[:8]:
        print(f"  {f}/{ns}: {len(thms)}")


if __name__ == "__main__":
    main()
